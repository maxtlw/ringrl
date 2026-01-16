from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    hidden_dim: int = Field(default=128, ge=1)


class RunnerConfig(BaseModel):
    starting_epsilon: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_decay_rate: float = Field(
        default=0.99999,  # 1 - 1e-5
        ge=0.0,
        le=1.0,
        description="Rate at which epsilon decays, epsilon = starting_epsilon * epsilon_decay_rate^episode",
    )
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0)
    validate_every_episodes: int = Field(
        default=100, ge=1, description="Number of episodes between validation runs"
    )
    validation_episodes: int = Field(
        default=20, ge=1, description="Number of episodes to run for validation"
    )
    num_workers: int = Field(default=4, ge=1)


class TrainerConfig(BaseModel):
    eta: float = Field(default=1e-3, ge=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    batch_size: int = Field(default=64, ge=1)
    target_sync_every_steps: int = Field(
        default=5_000,
        ge=1,
        description="Number of gradient steps between target network updates",
    )
    save_checkpoint_every_steps: int = Field(
        default=10_000,
        ge=1,
        description="Number of gradient steps between checkpoint saves",
    )
    update_actor_every_steps: int = Field(
        default=1_000,
        ge=1,
        description="Number of gradient steps between actor updates",
    )
    stacked_frames: int = Field(default=4, ge=1)


class Config(BaseModel):
    replay_buffer_size: int = Field(default=100_000, ge=1)
    replay_buffer_warmup_size: int = Field(
        default=10_000,
        ge=1,
        description="Number of transitions to collect before training starts, to warm up the replay buffer",
    )
    model: ModelConfig
    runner: RunnerConfig
    trainer: TrainerConfig
