from copy import deepcopy
from multiprocessing import Queue
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import huber_loss
from torch.optim import Optimizer

from convnet import DQN
from logs import get_logger
from params_store import SharedParamsStore
from replay_ring import SharedReplayRing

_LOGGER = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        network: DQN,
        replay_ring: SharedReplayRing,
        params_store: SharedParamsStore,
        batch_size: int,
        target_sync_every_steps: int,
        save_checkpoint_every_steps: int,
        update_actor_every_steps: int,
        warmup_complete_signal: EventType,
        global_gradient_steps: Synchronized,
        update_metrics_every_steps: int = 100,
    ):
        self.replay_ring = replay_ring
        self.target_sync_every_steps = target_sync_every_steps
        self.params_store = params_store
        self.batch_size = batch_size
        self.save_checkpoint_every_steps = save_checkpoint_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.warmup_complete_signal = warmup_complete_signal
        self.update_metrics_every_steps = update_metrics_every_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        run_name: str,
        metrics_queue: Queue,
    ):
        self.warmup_complete_signal.wait()

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=eta, betas=(0.9, 0.999), eps=0.00015
        )

        if self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = None

        _LOGGER.info("🏋️ Warmup complete, starting training")

        self.checkpoints_path = Path("checkpoints") / run_name.replace("Run: ", "")
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        while True:
            # Sync target network if necessary
            if (
                self.gradient_steps > 0
                and self.gradient_steps % self.target_sync_every_steps == 0
            ):
                _LOGGER.info(f"🔄 Syncing target network (step: {self.gradient_steps})")
                self.target_network.load_state_dict(self.network.state_dict())

            # Train one step
            self._train_step(
                optimizer=optimizer,
                gamma=gamma,
                metrics_queue=metrics_queue,
                scaler=scaler,
            )

            # Save checkpoint if necessary
            if self.gradient_steps > 0 and (
                self.gradient_steps % self.save_checkpoint_every_steps == 0
            ):
                _LOGGER.info(f"💾 Saving checkpoint (step: {self.gradient_steps})")
                torch.save(
                    self.network.state_dict(),
                    self.checkpoints_path / f"checkpoint_{self.gradient_steps}.pth",
                )

            # Update actor if necessary
            if (
                self.gradient_steps > 0
                and self.gradient_steps % self.update_actor_every_steps == 0
            ):
                _LOGGER.info(
                    f"🔄 Publishing actor parameters to shared memory (step: {self.gradient_steps})"
                )
                self.params_store.publish(self.network)

            self.gradient_steps += 1
            self.global_gradient_steps.value += 1

            _LOGGER.debug(f"🏋️ Gradient step {self.gradient_steps}")

    def _train_step(
        self,
        optimizer: Optimizer,
        gamma: float,
        metrics_queue: Queue,
        scaler: torch.amp.GradScaler | None,
    ):
        optimizer.zero_grad()

        n_written = self.replay_ring.write_pos.value
        n_ready = min(n_written, self.replay_ring.capacity)
        idxs = np.random.randint(
            0, n_ready, size=self.batch_size, dtype=np.int64
        )  # Allow duplicates for better performance

        (
            obs_batch,
            act_batch,
            rew_batch,
            next_obs_batch,
            terminated_batch,
            truncated_batch,
        ) = self.replay_ring.read_batch_safe(idxs)

        terminated_t = torch.from_numpy(terminated_batch)
        truncated_t = torch.from_numpy(truncated_batch)
        done_t = (terminated_t | truncated_t).to(self.device).bool()
        rew_t = torch.from_numpy(rew_batch).to(self.device)

        with torch.no_grad():
            next_obs_t = torch.from_numpy(next_obs_batch).to(self.device).float()
            next_action = self.network(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_obs_t).gather(1, next_action).squeeze(1)
            targets = rew_t + gamma * (~done_t).float() * next_q

        obs_t = torch.from_numpy(obs_batch).to(self.device).float()
        act_t = torch.from_numpy(act_batch).to(self.device).long()

        if scaler is not None:
            with torch.autocast(self.device.type, dtype=torch.float16):
                q_values = self.network(obs_t)
                q_values_actions = torch.gather(
                    q_values, 1, act_t.unsqueeze(1)
                ).squeeze(1)
                loss = huber_loss(q_values_actions, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    max_norm=10.0,
                )
                scaler.step(optimizer)
                scaler.update()
        else:
            q_values = self.network(obs_t)
            q_values_actions = torch.gather(q_values, 1, act_t.unsqueeze(1)).squeeze(1)
            loss = huber_loss(q_values_actions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                max_norm=10.0,
            )
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
                    "name": "mean_q_values",
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
