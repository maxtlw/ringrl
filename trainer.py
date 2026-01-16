from copy import deepcopy
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import huber_loss
from torch.optim import Optimizer

from conv_qnet import QNet
from logs import get_logger
from replay_ring import SharedReplayRing

_LOGGER = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        network: QNet,
        replay_ring: SharedReplayRing,
        updates_queue: Queue,
        target_sync_every_steps: int,
        save_checkpoint_every_steps: int,
        update_actor_every_steps: int,
        warmup_complete_signal: EventType,
        global_gradient_steps: Synchronized,
        update_metrics_every_steps: int = 100,
    ):
        self.replay_ring = replay_ring
        self.updates_queue = updates_queue
        self.target_sync_every_steps = target_sync_every_steps
        self.save_checkpoint_every_steps = save_checkpoint_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.warmup_complete_signal = warmup_complete_signal
        self.update_metrics_every_steps = update_metrics_every_steps

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        _LOGGER.info(f"Using device: {self.device}")

        self.network = deepcopy(network)
        self.network.train()
        self.network.to(self.device)
        self.target_network = deepcopy(network)
        self.target_network.eval()
        self.target_network.to(self.device)

        self.gradient_steps = 0
        self.global_gradient_steps = global_gradient_steps

    def run(
        self,
        gamma: float,
        eta: float,
        batch_size: int,
        run_name: str,
        metrics_queue: Queue,
    ):
        self.warmup_complete_signal.wait()

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=eta,
        )

        _LOGGER.info("🏋️ Warmup complete, starting training")

        self.checkpoints_path = Path("checkpoints") / run_name.replace("Run: ", "")
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        while True:
            # Sync target network if necessary
            if (
                self.gradient_steps > 0
                and self.gradient_steps % self.target_sync_every_steps == 0
            ):
                _LOGGER.info("🔄 Syncing target actor")
                self.target_network.load_state_dict(self.network.state_dict())

            # Train one step
            self._train_step(
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                metrics_queue=metrics_queue,
            )

            # Save checkpoint if necessary
            if (
                self.gradient_steps > 0
                and self.gradient_steps % self.save_checkpoint_every_steps == 0
            ):
                torch.save(
                    self.network.state_dict(),
                    self.checkpoints_path / f"checkpoint_{self.gradient_steps}.pth",
                )

            # Update actor if necessary
            if (
                self.gradient_steps > 0
                and self.gradient_steps % self.update_actor_every_steps == 0
            ):
                self._push_actor_updates()

            self.gradient_steps += 1
            self.global_gradient_steps.value += 1

            _LOGGER.info(f"🏋️ Gradient step {self.gradient_steps}")

    def _push_actor_updates(self):
        _LOGGER.debug("Pushing actor updates")
        cpu_state_dict = {
            k: v.detach().cpu() for k, v in self.network.state_dict().items()
        }
        self.updates_queue.put(cpu_state_dict)
        _LOGGER.debug("Pushed actor updates")

    def _train_step(
        self,
        optimizer: Optimizer,
        batch_size: int,
        gamma: float,
        metrics_queue: Queue,
    ):
        n_written = self.replay_ring.write_pos.value
        n_ready = min(n_written, self.replay_ring.capacity)
        idxs = np.random.choice(n_ready, size=batch_size, replace=False)

        (
            obs_batch,
            act_batch,
            rew_batch,
            next_obs_batch,
            terminated_batch,
            truncated_batch,
        ) = self.replay_ring.read_batch_safe(idxs)

        terminated_t = torch.as_tensor(terminated_batch, dtype=torch.bool).to(
            self.device
        )
        truncated_t = torch.as_tensor(truncated_batch, dtype=torch.bool).to(self.device)
        done_t = terminated_t | truncated_t
        rew_t = torch.as_tensor(rew_batch, dtype=torch.float32).to(self.device)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        q_values = self.network(obs_t)
        act_t = torch.as_tensor(act_batch, dtype=torch.long).to(self.device)
        q_values_actions = torch.gather(q_values, 1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_obs_t = torch.as_tensor(next_obs_batch, dtype=torch.float32).to(
                self.device
            )
            q_values_next = self.target_network(next_obs_t)
            q_values_next_max = torch.max(q_values_next, dim=1).values
            targets = rew_t + gamma * (~done_t).float() * q_values_next_max

        loss = huber_loss(q_values_actions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.gradient_steps % self.update_metrics_every_steps == 0:
            metrics_queue.put(
                {
                    "type": "track",
                    "name": "loss",
                    "value": loss.item(),
                    "step": self.gradient_steps,
                    "context": {"subset": "train"},
                }
            )
            metrics_queue.put(
                {
                    "type": "track",
                    "name": "q_values_mean",
                    "value": q_values.mean().item(),
                    "step": self.gradient_steps,
                    "context": {"subset": "train"},
                }
            )
            metrics_queue.put(
                {
                    "type": "track",
                    "name": "mean_absolute_td_error",
                    "value": (targets - q_values_actions).abs().mean().item(),
                    "step": self.gradient_steps,
                    "context": {"subset": "train"},
                }
            )
