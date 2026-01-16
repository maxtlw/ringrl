import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventType
from typing import Literal

import gymnasium as gym
import numpy as np
import torch

from conv_qnet import QNet
from logs import get_logger
from replay_ring import SharedReplayRing
from utils import get_breakout_env, sample_action

_LOGGER = get_logger(__name__)


class Runner:
    def __init__(
        self,
        actor: QNet,
        replay_ring: SharedReplayRing,
        updates_queue: Queue,
        initial_epsilon: float,
        epsilon_decay_rate: float,
        epsilon_min: float,
        validate_every_episodes: int,
        validation_episodes: int,
        stacked_frames: int = 4,
    ):
        self.actor = actor
        self.replay_ring = replay_ring
        self.updates_queue = updates_queue
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.validate_every_episodes = validate_every_episodes
        self.validation_episodes = validation_episodes
        self.stacked_frames = stacked_frames

        if not validate_every_episodes > validation_episodes:
            raise ValueError(
                "validate_every_episodes must be greater than validation_episodes"
            )

        self.process_name = mp.current_process().name

    def run(self, metrics_queue: Queue):
        episode = 0
        while True:
            self._maybe_update_actor()

            env = get_breakout_env(stacked_frames=self.stacked_frames)

            epsilon = max(
                self.epsilon_min,
                (self.epsilon_decay_rate**episode) * self.initial_epsilon,
            )

            self._rollout_episode(
                env=env,
                actor=self.actor,
                epsilon=epsilon,
                subset="train",
                episode=episode,
                metrics_queue=metrics_queue,
            )
            _LOGGER.info(f"🏃 [{self.process_name}] Episode {episode}")

            episode += 1

    def _maybe_update_actor(self):
        if self.updates_queue.empty():
            return
        _LOGGER.debug("Updating actor")
        self.actor.load_state_dict(self.updates_queue.get())
        _LOGGER.debug("Updated actor")

    def _rollout_episode(
        self,
        env: gym.Env,
        actor: QNet,
        epsilon: float,
        subset: Literal["train", "val"],
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
