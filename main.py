import multiprocessing as mp
import random
import time
from copy import deepcopy
from multiprocessing import Event, Process, Queue
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import Lock as LockType
from queue import Empty
from typing import Any

import ale_py  # noqa: F401  (import registers ALE envs via side effects)
import torch
from aim import Distribution, Run
from yaml import FullLoader, load

from config import Config
from convnet import DQN
from params_store import SharedParamsStore
from replay_ring import SharedReplayRing
from runner import Runner
from trainer import Trainer
from utils import UTDCalculator

torch.manual_seed(42)
random.seed(42)


def _runner_worker(
    actor_network: DQN,
    params_store: SharedParamsStore,
    replay_ring_handles: dict,
    replay_ring_write_pos: Synchronized,
    metrics_queue: Queue,
    warmup_complete_signal: EventType,
    config: Config,
):
    replay_ring = SharedReplayRing.attach(
        handles=replay_ring_handles,
        write_pos=replay_ring_write_pos,
    )

    actor_network = deepcopy(actor_network)
    actor_network.eval()

    runner = Runner(
        actor=actor_network,
        params_store=params_store,
        replay_ring=replay_ring,
        initial_epsilon=config.runner.starting_epsilon,
        epsilon_decay_rate=config.runner.epsilon_decay_rate,
        epsilon_min=config.runner.epsilon_min,
        validate_every_episodes=config.runner.validate_every_episodes,
        validation_episodes=config.runner.validation_episodes,
        warmup_complete_signal=warmup_complete_signal,
        stacked_frames=config.trainer.stacked_frames,
    )
    try:
        runner.run(metrics_queue=metrics_queue)
    finally:
        replay_ring.close_only()


def _trainer_worker(
    params_store: SharedParamsStore,
    replay_ring_handles: dict,
    replay_ring_write_pos: Synchronized,
    metrics_queue: Queue,
    warmup_complete_signal: EventType,
    config: Config,
    run_name: str,
    global_gradient_steps: Synchronized,
):
    replay_ring = SharedReplayRing.attach(
        handles=replay_ring_handles,
        write_pos=replay_ring_write_pos,
    )

    base_network = DQN(
        observations_channels=config.trainer.stacked_frames,
        action_space_dim=4,
    )

    trainer = Trainer(
        network=base_network,
        params_store=params_store,
        replay_ring=replay_ring,
        batch_size=config.trainer.batch_size,
        target_sync_every_steps=config.trainer.target_sync_every_steps,
        save_checkpoint_every_steps=config.trainer.save_checkpoint_every_steps,
        update_actor_every_steps=config.trainer.update_actor_every_steps,
        warmup_complete_signal=warmup_complete_signal,
        global_gradient_steps=global_gradient_steps,
    )

    try:
        trainer.run(
            gamma=config.trainer.gamma,
            eta=config.trainer.eta,
            run_name=run_name,
            metrics_queue=metrics_queue,
        )
    finally:
        replay_ring.close_only()


if __name__ == "__main__":
    # Match macOS default behavior and avoid CUDA+fork pitfalls on Linux.
    mp.set_start_method("spawn", force=True)

    config = Config(**load(open("config.yml", "r"), Loader=FullLoader))

    run = Run(experiment="breakout-dqn")
    run["hparams"] = {
        "eta": config.trainer.eta,
        "gamma": config.trainer.gamma,
        "epsilon": config.runner.starting_epsilon,
        "epsilon_decay_rate": config.runner.epsilon_decay_rate,
        "epsilon_min": config.runner.epsilon_min,
        "replay_buffer_warmup_size": config.replay_buffer_warmup_size,
        "batch_size": config.trainer.batch_size,
    }

    base_network = DQN(
        observations_channels=config.trainer.stacked_frames,
        action_space_dim=4,
    )

    replay_ring = SharedReplayRing(
        capacity=config.replay_buffer_size,
        obs_shape=(config.trainer.stacked_frames, 84, 84),
    )
    replay_ring_write_pos = mp.Value("q", 0, lock=True)  # int64

    params_store = SharedParamsStore(model=base_network)

    global_gradient_steps = mp.Value("q", 0)  # int64

    metrics_queue: MPQueue[dict[str, Any]] = Queue()
    warmup_complete_signal = Event()

    # Spawn a set of runner workers
    runner_processes = [
        Process(
            name=f"runner-{i}",
            target=_runner_worker,
            args=(
                base_network,
                params_store,
                replay_ring.export_handles(),
                replay_ring_write_pos,
                metrics_queue,
                warmup_complete_signal,
                config,
            ),
        )
        for i in range(config.runner.num_workers)
    ]

    trainer_process = Process(
        name="trainer",
        target=_trainer_worker,
        args=(
            params_store,
            replay_ring.export_handles(),
            replay_ring_write_pos,
            metrics_queue,
            warmup_complete_signal,
            config,
            run.name,
            global_gradient_steps,
        ),
    )

    for process in runner_processes:
        process.start()
    trainer_process.start()

    utd_calculator = None

    try:
        last_track_time = 0.0
        while True:
            # Handle warmup completion
            if not warmup_complete_signal.is_set():
                if replay_ring_write_pos.value >= config.replay_buffer_warmup_size:
                    warmup_complete_signal.set()
                    utd_calculator = UTDCalculator(
                        current_produced_transitions=replay_ring_write_pos.value,
                        current_gradient_steps=global_gradient_steps.value,
                    )

            now = time.time()
            if now - last_track_time >= 5.0:
                run.track(
                    min(replay_ring_write_pos.value, replay_ring.capacity),
                    name="replay_buffer_size",
                    step=replay_ring_write_pos.value,
                )

                if utd_calculator is not None:
                    utd = utd_calculator.compute(
                        current_produced_transitions=replay_ring_write_pos.value,
                        current_gradient_steps=global_gradient_steps.value,
                    )
                    run.track(
                        utd["utd"],
                        name="utd",
                        step=global_gradient_steps.value,
                    )
                    run.track(
                        utd["gradient_steps_per_second"],
                        name="steps_per_second",
                        step=global_gradient_steps.value,
                        context={"type": "gradient"},
                    )
                    run.track(
                        utd["transitions_per_second"],
                        name="steps_per_second",
                        step=global_gradient_steps.value,
                        context={"type": "transitions"},
                    )

                last_track_time = now

            # Drain metrics without letting the main loop block indefinitely.
            drained = 0
            while drained < 1000:
                try:
                    msg = metrics_queue.get(timeout=0.5 if drained == 0 else 0.0)
                except Empty:
                    break

                msg_type = msg.get("type")
                if msg_type == "track":
                    run.track(
                        msg["value"],
                        name=msg["name"],
                        step=msg["step"],
                        context=msg.get("context"),
                    )
                elif msg_type == "distribution":
                    run.track(
                        Distribution(msg["values"], bin_count=msg["bin_count"]),
                        name=msg["name"],
                        step=msg["step"],
                        context=msg.get("context"),
                    )
                drained += 1
    finally:
        for process in runner_processes:
            process.join()
        trainer_process.join()
        replay_ring.close_and_unlink()
