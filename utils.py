import random
import time

import gymnasium as gym
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class UTDCalculator:
    def __init__(self, current_produced_transitions: int, current_gradient_steps: int):
        self.last_produced_transitions = current_produced_transitions
        self.last_gradient_steps = current_gradient_steps

        self.last_update_time = time.time()

    def compute(
        self, current_produced_transitions: int, current_gradient_steps: int
    ) -> dict[str, float]:
        now = time.time()
        delta_time = now - self.last_update_time

        delta_transitions = (
            current_produced_transitions - self.last_produced_transitions
        )
        delta_gradient_steps = current_gradient_steps - self.last_gradient_steps

        gradient_steps_per_second = delta_gradient_steps / delta_time
        transitions_per_second = delta_transitions / delta_time
        utd = gradient_steps_per_second / transitions_per_second

        self.last_produced_transitions = current_produced_transitions
        self.last_gradient_steps = current_gradient_steps
        self.last_update_time = now

        return {
            "gradient_steps_per_second": gradient_steps_per_second,
            "transitions_per_second": transitions_per_second,
            "utd": utd,
        }


def sample_action(
    q_values: torch.Tensor,
    epsilon: float,
    action_space: gym.spaces.Space,
) -> int:
    if random.random() < epsilon:
        action = action_space.sample()
    else:
        action = torch.argmax(q_values.detach()).item()
    return action


def get_breakout_env(stacked_frames: int = 4):
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=stacked_frames,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
    )
    env = FrameStackObservation(env, stacked_frames)
    return env
