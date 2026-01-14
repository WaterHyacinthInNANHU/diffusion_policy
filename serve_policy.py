#!/usr/bin/env python3
"""
WebSocket server for Diffusion Policy inference.

This server wraps a trained diffusion policy checkpoint and exposes it
via a WebSocket protocol compatible with OpenPI/Polaris.

Usage:
    python serve_policy.py --checkpoint <path_to_checkpoint> --port 8000
"""

import argparse
import asyncio
import http
import logging
import time
import traceback
from typing import Dict, Any

import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames

from omegaconf import OmegaConf
import hydra

# msgpack for serialization (compatible with openpi-client)
try:
    from openpi_client import msgpack_numpy
except ImportError:
    # Fallback if openpi_client is not installed
    import msgpack
    import msgpack_numpy as m
    m.patch()

    class msgpack_numpy:
        @staticmethod
        def unpackb(data):
            return msgpack.unpackb(data, raw=False)

        class Packer:
            def pack(self, data):
                return msgpack.packb(data, use_bin_type=True)


logger = logging.getLogger(__name__)


class DiffusionPolicyServer:
    """
    WebSocket server that wraps a trained diffusion policy checkpoint.

    Protocol:
    - On connection: sends metadata dict
    - On each request: receives observation dict, returns action dict

    Expected observation format (from Polaris):
        {
            "observation/exterior_image_1_left": np.ndarray (H, W, 3) uint8,
            "observation/wrist_image_left": np.ndarray (H, W, 3) uint8,
            "observation/joint_position": np.ndarray (7,) float32,
            "observation/gripper_position": np.ndarray (1,) float32,
            "prompt": str (language instruction, currently unused)
        }

    Response format:
        {
            "actions": np.ndarray (horizon, 8) float32,
            "server_timing": {"infer_ms": float}
        }
    """

    def __init__(
        self,
        checkpoint_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "cuda:0",
        image_size: int = 224,
    ):
        """
        Initialize the diffusion policy server.

        Args:
            checkpoint_path: Path to trained checkpoint (.ckpt)
            host: Host to bind to
            port: Port to listen on
            device: Device to run inference on
            image_size: Expected image size (must match training)
        """
        self.host = host
        self.port = port
        self.device = torch.device(device)
        self.image_size = image_size

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        self.cfg = checkpoint['cfg']
        logger.info("Policy configuration:")
        logger.info(OmegaConf.to_yaml(self.cfg.policy))

        # Create policy
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        self.policy.load_state_dict(checkpoint['state_dicts']['model'])
        self.policy.to(self.device)
        self.policy.eval()

        # Load normalizer
        self.normalizer = checkpoint['state_dicts']['normalizer']
        self.normalizer.to(self.device)

        # Policy parameters
        self.n_obs_steps = self.cfg.n_obs_steps
        self.n_action_steps = self.cfg.n_action_steps
        self.horizon = self.cfg.horizon

        # Observation buffer (per connection)
        self.obs_buffer = []

        # Metadata to send on connection
        self.metadata = {
            "policy_type": "diffusion_policy",
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "horizon": self.horizon,
            "image_size": self.image_size,
            "action_dim": 8,
        }

        logger.info(f"Policy loaded successfully")
        logger.info(f"  n_obs_steps: {self.n_obs_steps}")
        logger.info(f"  n_action_steps: {self.n_action_steps}")
        logger.info(f"  horizon: {self.horizon}")

    def reset(self):
        """Reset observation buffer for new episode."""
        self.obs_buffer = []

    def preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess observation from Polaris format to policy input format.

        Args:
            obs: Observation dict from client

        Returns:
            Preprocessed observation dict for policy
        """
        # Extract image - handle both naming conventions
        if "observation/exterior_image_1_left" in obs:
            image = obs["observation/exterior_image_1_left"]
        elif "external_image" in obs:
            image = obs["external_image"]
        else:
            raise KeyError(f"No external image found in obs. Keys: {obs.keys()}")

        # Image: (H, W, C) uint8 -> (1, C, H, W) float32 in [0, 1]
        if image.shape[:2] != (self.image_size, self.image_size):
            import cv2
            image = cv2.resize(image, (self.image_size, self.image_size))
        image = np.moveaxis(image, -1, 0) / 255.0  # (C, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, C, H, W)

        # Extract state
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            gripper_pos = obs["observation/gripper_position"]
        elif "joint_position" in obs:
            joint_pos = obs["joint_position"]
            gripper_pos = obs["gripper_position"]
        else:
            raise KeyError(f"No joint position found in obs. Keys: {obs.keys()}")

        # Ensure correct shapes
        joint_pos = np.asarray(joint_pos).flatten()[:7]
        gripper_pos = np.asarray(gripper_pos).flatten()[:1]

        state = np.concatenate([joint_pos, gripper_pos], axis=0)  # (8,)
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, 8)

        return {
            'image': image.to(self.device),
            'state': state.to(self.device)
        }

    @torch.no_grad()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on observation.

        Args:
            obs: Observation dict from client

        Returns:
            Action dict with "actions" key containing action sequence
        """
        # Check for reset signal
        if obs.get("reset", False):
            self.reset()
            return {"actions": np.zeros((self.n_action_steps, 8), dtype=np.float32)}

        # Preprocess observation
        processed_obs = self.preprocess_obs(obs)

        # Add to buffer
        self.obs_buffer.append(processed_obs)

        # Keep only last n_obs_steps observations
        if len(self.obs_buffer) > self.n_obs_steps:
            self.obs_buffer.pop(0)

        # If buffer not full yet, repeat the first observation
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.insert(0, self.obs_buffer[0])

        # Stack observations
        obs_dict = {
            'image': torch.cat([o['image'] for o in self.obs_buffer], dim=0),  # (n_obs_steps, C, H, W)
            'state': torch.cat([o['state'] for o in self.obs_buffer], dim=0)   # (n_obs_steps, 8)
        }

        # Add batch dimension
        obs_dict = {
            'image': obs_dict['image'].unsqueeze(0),  # (1, n_obs_steps, C, H, W)
            'state': obs_dict['state'].unsqueeze(0)   # (1, n_obs_steps, 8)
        }

        # Normalize
        obs_dict = self.normalizer.normalize(obs_dict)

        # Predict action sequence
        action_seq = self.policy.predict_action(obs_dict)  # (1, horizon, 8)

        # Return action chunk
        actions = action_seq[0, :self.n_action_steps].cpu().numpy()  # (n_action_steps, 8)

        return {"actions": actions.astype(np.float32)}

    async def _handler(self, websocket: _server.ServerConnection):
        """Handle WebSocket connection."""
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Reset state for new connection
        self.reset()

        # Send metadata
        await websocket.send(packer.pack(self.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                self.reset()
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def serve_forever(self):
        """Start the server."""
        asyncio.run(self._run())

    async def _run(self):
        """Async server main loop."""
        async def health_check(connection, request):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        logger.info(f"Starting diffusion policy server on {self.host}:{self.port}")
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Serve Diffusion Policy via WebSocket")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Expected image size (default: 224)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run server
    server = DiffusionPolicyServer(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        image_size=args.image_size,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
