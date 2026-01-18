# RingRL

<p align="center">
  <img src="img/ringrl.png" alt="RingRL Logo" width="300" height="300"/>
</p>


A high-throughput multiprocessing reinforcement learning runtime built around shared-memory ring buffers and explicit actor–learner coordination.

## ⁉️ Why?

In off-policy reinforcement learning, experience generation and learning are fundamentally decoupled processes: new transitions can be collected independently of when and how they are consumed by the learner. In practice, however, naive Python implementations tightly interleave environment rollouts and gradient updates in a single process or thread. This often leads to severely underutilized accelerators (GPUs/MPS) and makes it difficult to scale experience generation beyond a small number of environments.

A common first step is to introduce multiprocessing. While this may improve throughput, it also quickly espose new bottlenecks - especially when observations are large and frequent (as in the Atari environments with stacked image frames). Using standard queues (`multiprocessing.Queue` or similar abstractions) requires serializing and deserializing hundreds of thousands of transitions, which incurs significant overhead and rapidly dominates wall-clock time. In other words, **data movement and coordination** is often the real performance limit.

RingRL was built to explore what happens when these bottlenecks are treated as first-class systems problems: instead of passing transitions and parameters through queues, it relies on shared-memory data structures with simple lock-free protocols. This makes it possible to scale the number of actors while keeping the learner constantly busy. 

> **Note**
>
> RingRL is primarily a didactic systems project. Its purpose is to study and demonstrate how far a Python-based actor–learner architecture can be pushed when memory layout and inter-process communication are designed explicitly.
>
> It is not intended to be a turn-key reinforcement learning framework.

## 💡 Core ideas

RingRL is built around a small set of explicit design principles aimed at maximizing throughput and clarity in a multiprocessing reinforcement learning setting.

### Actor-learner separation

Experience generation and learning are treated as distinct concerns. Actors continuously run independent environment interactions, while a dedicated learner process is fully dedicated to perform gradient updates.

This separation maximizes GPU/MPS usage by avoiding artificial synchronization points (e.g. "train after an episode") and makes it possible to reason explicitly about the balance between data generation and learning using metrics such as Update-to-Data (UTD).

### Shared-memory experience replay

Instead of passing transitions through traditional queues, experience is written directly into a shared-memory **ring buffer**. Actors appens transitions, and the learner samples batches directly from the same memory region. The design eliminates entirely serialization and copy overhead.

### Broadcast parameters update

To avoid concurrency on the accelerators, actors run independently with slightly stale parameters, which is acceptable in off-policy RL and enables asynchronous, high-throughput execution. The learner periodically broadcasts new, updated parameters into a shared memory buffer that actors access independently.

### Lock-free commit protocols for shared data

To minimize coordination overhead between processes, RingRL relies on simple lock-free commit protocols for shared data structures. 

For the **experience replay**, transitions are written into a shared-memory ring buffer using per-slot sequence numbers. Each slot follows a small state machine:
* A writer marks the slot as “being written”
* Commits the payload
* Marks the slot as complete. 

Readers can detect incomplete writes and retry without blocking, and no process hold a global lock, thus enabling low contention even at very high throughputs.

A similar idea is used for **parameter synchronization**. Model parameters are published to shared memory together with a monotonically increasing version counter (with odd values signaling "writing" and even ones signaling "commit"). Actors periodically check the version and only apply updates when a new version is observed, treating weight synchronization as a broadcast operation.

## 🛠️ Run locally

To run the training locally you'll need to start two separate processes using [uv](https://github.com/astral-sh/uv). If you haven't already, first of all install all the dependencies:
```
uv sync
```

Then, you need to start two separate processes:

**Training**
```bash
uv run python main.py
```

**AIMStack** (to track the metrics)
```bash
uv run aim up
```

## 🗒️ References
* Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013 [[paper](https://arxiv.org/abs/1312.5602)]
* van Hasselt tt al., *Deep Reinforcement Learning with Double Q-Learning*, 2015 [[paper](https://arxiv.org/abs/1509.06461)]