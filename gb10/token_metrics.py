"""Training token accounting for the two-node GB10 profile (CPU-testable).

Stage batch_size in TrainingSchedule is already GLOBAL across both nodes.
Do NOT multiply that number by world_size, nor use the maximum stage's
tokens per GPU when describing an early 8-step smoke result.
"""


def global_token_budget(schedule, run_steps):
    """Exact scheduled global tokens, including stage boundaries and taper."""
    if not isinstance(run_steps, int) or run_steps < 0:
        raise ValueError("run_steps must be a nonnegative integer")
    return sum(int(schedule.lookup(step)[0].batch_size) for step in range(run_steps))


def local_tokens_for_step(schedule, step, world_size):
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be >=1")
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    global_tokens = int(schedule.lookup(step)[0].batch_size)
    if global_tokens % world_size:
        raise ValueError("global batch must be divisible by world_size")
    return global_tokens // world_size
