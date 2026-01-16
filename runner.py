import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventType

import gymnasium as gym
import numpy as np
import torch

from conv_qnet import QNet
from logs import get_logger
from params_store import SharedParamsStore
from replay_ring import SharedReplayRing
from utils import get_breakout_env, sample_action

_LOGGER = get_logger(__name__)


class Runner:
    def __init__(
        self,
        actor: QNet,
        replay_ring: SharedReplayRing,
        params_store: SharedParamsStore,
        initial_epsilon: float,
        epsilon_decay_rate: float,
        epsilon_min: float,
        validate_every_episodes: int,
        validation_episodes: int,
        warmup_complete_signal: EventType,
        stacked_frames: int = 4,
    ):
        self.actor = actor
        self.replay_ring = replay_ring
        self.params_store = params_store
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.validate_every_episodes = validate_every_episodes
        self.validation_episodes = validation_episodes
        self.warmup_complete_signal = warmup_complete_signal
        self.stacked_frames = stacked_frames

        if not validate_every_episodes > validation_episodes:
            raise ValueError(
                "validate_every_episodes must be greater than validation_episodes"
            )

        self.process_name = mp.current_process().name

        self.tmp_cpu = torch.empty_like(self.params_store.shared)
        self.current_actor_version = 0

    def run(self, metrics_queue: Queue):
        process_rank = int(self.process_name.split("-")[-1])

        episode = 0
        while True:
            if (
                episode >= self.validation_episodes
                and (episode % self.validate_every_episodes < self.validation_episodes)
                and self.warmup_complete_signal.is_set()
                and episode % (process_rank + 1) == 0  # only validate on one worker
            ):
                subset = "val"
                epsilon = 0.01
            else:
                subset = "train"
                epsilon = max(
                    self.epsilon_min,
                    (self.epsilon_decay_rate**episode) * self.initial_epsilon,
                )

            self._maybe_update_actor()

            env = get_breakout_env(stacked_frames=self.stacked_frames)

            self._rollout_episode(
                env=env,
                actor=self.actor,
                epsilon=epsilon,
                subset=subset,
                episode=episode,
                metrics_queue=metrics_queue,
            )
            _LOGGER.debug(f"🏃 [{self.process_name}] Episode {episode}")

            episode += 1

    def _maybe_update_actor(self):
        if (
            self.params_store.version.value % 2 == 0
            and self.params_store.version.value > self.current_actor_version
        ):
            self.params_store.read_into(self.tmp_cpu)
            torch.nn.utils.vector_to_parameters(self.tmp_cpu, self.actor.parameters())
            self.current_actor_version = self.params_store.version.value

    def _rollout_episode(
        self,
        env: gym.Env,
        actor: QNet,
        epsilon: float,
        subset: str,
        episode: int,
        metrics_queue: Queue,
    ):
        episode_return = 0.0
        episode_length = 0

        state, _ = env.reset(seed=42 + episode)
        terminated, truncated = False, False

        while not (terminated or truncated):
            state_t = torch.as_tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = actor(state_t.unsqueeze(0)).squeeze(0)
            action = sample_action(q_values, epsilon, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action)

            state_u8 = np.ascontiguousarray(state, dtype=np.uint8)
            next_state_u8 = np.ascontiguousarray(next_state, dtype=np.uint8)

            self.replay_ring.write_slot(
                obs_u8=state_u8,
                act=action,
                rew=float(reward),
                next_obs_u8=next_state_u8,
                terminated=terminated,
                truncated=truncated,
            )

            state = next_state

            episode_return += float(reward)
            episode_length += 1

        metrics_queue.put(
            {
                "type": "track",
                "name": "episode_return",
                "value": episode_return,
                "step": episode,
                "context": {"subset": subset, "worker": self.process_name},
            }
        )
        metrics_queue.put(
            {
                "type": "track",
                "name": "episode_length",
                "value": episode_length,
                "step": episode,
                "context": {"subset": subset, "worker": self.process_name},
            }
        )
        metrics_queue.put(
            {
                "type": "track",
                "name": "epsilon",
                "value": epsilon,
                "step": episode,
                "context": {"subset": subset, "worker": self.process_name},
            }
        )
        metrics_queue.put(
            {
                "type": "track",
                "name": "actor_version",
                "value": self.current_actor_version,
                "step": episode,
                "context": {"worker": self.process_name},
            }
        )
