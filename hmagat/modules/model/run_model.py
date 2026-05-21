import time

from pogema import pogema_v0, AnimationMonitor

import torch

from hmagat.runtime_data_generation import get_runtime_data_generator
from hmagat.collision_shielding import get_collision_shielded_model


@torch.no_grad()
def run_model_on_grid(
    model,
    device,
    grid_config,
    args,
    hypergraph_model,
    dataset_kwargs,
    use_target_vec,
    max_episodes=None,
    aux_func=None,
    animation_monitor=False,
    debug_label=None,
    debug_slow_step_sec=None,
):
    env = pogema_v0(grid_config=grid_config)
    if animation_monitor:
        env = AnimationMonitor(env)
    observations, infos = env.reset()

    model.in_simulation(True)

    rt_data_generator = get_runtime_data_generator(
        grid_config=grid_config,
        args=args,
        hypergraph_model=hypergraph_model,
        dataset_kwargs=dataset_kwargs,
        use_target_vec=use_target_vec,
    )

    if aux_func is not None:
        aux_func(
            env=env, observations=observations, actions=None, rtdg=rt_data_generator
        )

    model = get_collision_shielded_model(
        model, env, args, rt_data_generator=rt_data_generator
    )

    step_idx = 0
    while True:
        actions_start = time.monotonic()
        actions = model.get_actions(observations)
        actions_sec = time.monotonic() - actions_start

        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = env.step(actions)
        env_step_sec = time.monotonic() - env_step_start

        completed_step = step_idx + 1

        if aux_func is not None:
            aux_func(
                env=env,
                observations=observations,
                actions=actions,
                rtdg=rt_data_generator,
            )

        if (
            (debug_label is not None)
            and (debug_slow_step_sec is not None)
            and (
                actions_sec >= debug_slow_step_sec
                or env_step_sec >= debug_slow_step_sec
            )
        ):
            print(
                (
                    f"{debug_label} step={completed_step} done: "
                    f"get_actions_sec={actions_sec:.3f}, env_step_sec={env_step_sec:.3f}, "
                    f"terminated={all(terminated)}, truncated={all(truncated)}"
                ),
                flush=True,
            )

        if all(terminated) or all(truncated):
            break

        if max_episodes is not None:
            max_episodes -= 1
            if max_episodes <= 0:
                break
        step_idx += 1
    model.in_simulation(False)
    return all(terminated), env, observations
