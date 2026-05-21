import argparse
import csv
import pickle
import pathlib
import numpy as np
import wandb
from collections import OrderedDict
import time
import subprocess

import torch

from hmagat.run_expert import add_expert_dataset_args

from hmagat.training_args import add_training_args
from hmagat.convert_to_imitation_dataset import add_imitation_dataset_args
from hmagat.generate_hypergraphs import add_hypergraph_generation_args

# from agents import run_model_on_grid, get_model
from hmagat.modules.model.run_model import run_model_on_grid
from grid_config_generator import grid_config_generator_factory

from hmagat.generate_additional_data import add_additional_data_args

from hmagat.temperature_training import (
    add_temperature_sampling_args,
    get_temperature_sampling_model,
)
from hmagat.modules.agents import get_model
from hmagat.checkpointing import (
    load_model_state_dict_from_checkpoint_path,
    load_partial_checkpoint_into_model,
    resolve_evaluation_checkpoint_path,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _resolve_step_metrics_model_label(args):
    if args.step_metrics_model_label is not None:
        return args.step_metrics_model_label
    if args.run_name is not None:
        return args.run_name
    return args.imitation_learning_model


def _build_step_metrics_rows(
    *,
    model_label,
    graph_idx,
    instance_seed,
    sampling_seed,
    solved_agents_per_step,
    total_agents,
    max_episode_steps,
    rollout_success,
    makespan,
    pad_to_max_episode_steps,
):
    solved_agents_per_step = list(int(v) for v in solved_agents_per_step)
    if not solved_agents_per_step:
        solved_agents_per_step = [0]

    if pad_to_max_episode_steps and max_episode_steps is not None:
        target_len = max(1, int(max_episode_steps) + 1)
        if len(solved_agents_per_step) < target_len:
            solved_agents_per_step.extend(
                [solved_agents_per_step[-1]] * (target_len - len(solved_agents_per_step))
            )
        elif len(solved_agents_per_step) > target_len:
            solved_agents_per_step = solved_agents_per_step[:target_len]

    rows = []
    for step_idx, solved_agents in enumerate(solved_agents_per_step):
        success_fraction = 0.0 if total_agents <= 0 else solved_agents / total_agents
        rows.append(
            {
                "model": model_label,
                "graph_idx": graph_idx,
                "instance_seed": int(instance_seed),
                "sampling_seed": int(sampling_seed),
                "step": step_idx,
                "solved_agents": int(solved_agents),
                "total_agents": int(total_agents),
                "success_fraction": float(success_fraction),
                "all_agents_on_goal": int(solved_agents == total_agents),
                "rollout_success": int(bool(rollout_success)),
                "makespan": int(makespan),
                "max_episode_steps": int(max_episode_steps),
            }
        )
    return rows


def _append_step_metrics_csv(csv_path, rows):
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "graph_idx",
        "instance_seed",
        "sampling_seed",
        "step",
        "solved_agents",
        "total_agents",
        "success_fraction",
        "all_agents_on_goal",
        "rollout_success",
        "makespan",
        "max_episode_steps",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


class TimingModelWrapper:
    def __init__(self, base_obj):
        self.base_obj = base_obj
        self.timings = []

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        results = self.base_obj(*args, **kwargs)
        end_time = time.time()
        self.timings.append(end_time - start_time)
        return results

    def __getattr__(self, name):
        return getattr(self.base_obj, name)


def main():
    parser = argparse.ArgumentParser(description="Test imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)
    parser = add_temperature_sampling_args(parser)

    parser.add_argument("--test_map_type", type=str, default="RandomGrid")
    parser.add_argument("--test_map_h", type=int, default=20)
    parser.add_argument("--test_map_w", type=int, default=20)
    parser.add_argument("--test_robot_density", type=float, default=0.025)
    parser.add_argument("--test_obstacle_density", type=float, default=0.1)
    parser.add_argument("--test_max_episode_steps", type=int, default=128)
    parser.add_argument("--test_obs_radius", type=int, default=3)
    parser.add_argument("--test_collision_system", type=str, default="soft")
    parser.add_argument("--test_on_target", type=str, default="nothing")

    parser.add_argument("--test_num_samples", type=int, default=2000)
    parser.add_argument("--test_dataset_seed", type=int, default=42)
    parser.add_argument("--test_fixed_instance_seed", type=int, default=None)
    parser.add_argument("--test_sampling_seed", type=int, default=None)
    parser.add_argument("--test_dataset_dir", type=str, default="dataset")

    parser.add_argument("--test_comm_radius", type=int, default=7)
    parser.add_argument("--model_epoch_num", type=int, default=None)

    parser.add_argument("--test_name", type=str, default="in_distribution")
    parser.add_argument(
        "--test_wrt_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--test_min_dist", type=int, default=None)
    parser.add_argument("--test_max_dist", type=int, default=None)

    parser.add_argument("--test_map_types", type=str, default="random=0.1+maze=0.9")
    parser.add_argument("--test_map_w_min", type=int, default=16)
    parser.add_argument("--test_map_w_max", type=int, default=20)
    parser.add_argument("--test_num_agents", type=str, default="16+24+32")
    parser.add_argument("--test_obstacle_density_min", type=float, default=0.2)
    parser.add_argument("--test_obstacle_density_max", type=float, default=1.0)
    parser.add_argument("--test_go_straight_min", type=float, default=0.75)
    parser.add_argument("--test_go_straight_max", type=float, default=0.85)

    parser.add_argument("--test_wall_width_min", type=int, default=3)
    parser.add_argument("--test_wall_width_max", type=int, default=5)
    parser.add_argument("--test_wall_height_min", type=int, default=2)
    parser.add_argument("--test_wall_height_max", type=int, default=2)
    parser.add_argument("--test_side_pad", type=int, default=2)
    parser.add_argument("--test_horizontal_gap", type=int, default=1)
    parser.add_argument("--test_vertical_gap", type=int, default=3)
    parser.add_argument("--test_vertical_gap_min", type=int, default=None)
    parser.add_argument("--test_vertical_gap_max", type=int, default=None)
    parser.add_argument("--test_num_wall_rows_min", type=int, default=None)
    parser.add_argument("--test_num_wall_rows_max", type=int, default=None)
    parser.add_argument("--test_num_wall_cols_min", type=int, default=None)
    parser.add_argument("--test_num_wall_cols_max", type=int, default=None)
    parser.add_argument("--test_wfi_instance", action="store_true", default=False)
    parser.add_argument("--test_block_extra_space", action="store_true", default=True)

    parser.add_argument("--test_room_width_min", type=int, default=5)
    parser.add_argument("--test_room_width_max", type=int, default=9)
    parser.add_argument("--test_room_height_min", type=int, default=5)
    parser.add_argument("--test_room_height_max", type=int, default=9)
    parser.add_argument("--test_num_rows_min", type=int, default=3)
    parser.add_argument("--test_num_rows_max", type=int, default=5)
    parser.add_argument("--test_num_cols_min", type=int, default=3)
    parser.add_argument("--test_num_cols_max", type=int, default=5)
    parser.add_argument("--test_room_grid_uniform", action="store_true", default=True)
    parser.add_argument(
        "--test_room_only_centre_obstacles", action="store_true", default=False
    )

    parser.add_argument("--test_map_dir", type=str, default=None)
    parser.add_argument("--test_num_maps", type=int, default=1)

    parser.add_argument(
        "--test_ensure_grid_config_is_generatable",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--test_regulate_obstacle_density_max",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--temperature_sampling_model_epoch_num", type=int, default=None
    )

    parser.add_argument("--skip_n", type=int, default=None)
    parser.add_argument("--subsample_n", type=int, default=None)

    parser.add_argument("--svg_save_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="hyper-mapf-test")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tag", type=str, default=None)

    parser.add_argument("--record_timings", action="store_true", default=False)
    parser.add_argument("--step_metrics_csv_path", type=str, default=None)
    parser.add_argument("--step_metrics_model_label", type=str, default=None)
    parser.add_argument(
        "--step_metrics_pad_to_max_episode_steps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    print(args)

    assert args.save_termination_state

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(args.test_dataset_seed)
    seeds = rng.integers(10**10, size=args.test_num_samples)

    _grid_config_generator = grid_config_generator_factory(args, testing=True)

    model, hypergraph_model, dataset_kwargs = get_model(args, device)

    if args.temperature_sampling_model_epoch_num is None:
        if args.rl_based_temperature_sampling:
            # Only here to count the number of parameters
            model = get_temperature_sampling_model(model, args, device, state_dict=None)

        num_parameters = count_parameters(model)
        print(f"Num Parameters: {num_parameters}")

    checkpoint_path = resolve_evaluation_checkpoint_path(
        args.checkpoints_dir,
        model_epoch_num=args.model_epoch_num,
        map_location=device,
    )

    if args.load_partial_parameters_path is not None:
        print("Partially Loading Weights.............")
        partial_path = pathlib.Path(args.load_partial_parameters_path)
        load_partial_checkpoint_into_model(
            model,
            partial_path,
            map_location=device,
            print_prefix="[partial-load] ",
        )
    else:
        state_dict = load_model_state_dict_from_checkpoint_path(
            checkpoint_path,
            map_location=device,
            print_prefix="[eval-load] ",
        )
        model.load_state_dict(state_dict)
    model = model.eval()

    if args.rl_based_temperature_sampling:
        assert args.temperature_sampling_model_epoch_num is not None
        checkpoint_path = pathlib.Path(
            args.temperature_checkpoints_dir,
            f"epoch_{args.temperature_sampling_model_epoch_num}.pt",
        )
        state_dict = torch.load(checkpoint_path, map_location=device)

        model = get_temperature_sampling_model(model, args, device, state_dict)

        num_parameters = count_parameters(model)
        print(f"Num Parameters: {num_parameters}")

    run_name = f"{args.test_name}_{args.run_name}"
    if args.rl_based_temperature_sampling:
        run_name += (
            f"_{args.temperature_run_name}_e{args.temperature_sampling_model_epoch_num}"
        )
    use_wandb = args.wandb_entity is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args) | {"num_params": num_parameters},
            entity=args.wandb_entity,
            tags=[args.wandb_tag] if args.wandb_tag is not None else None,
        )

    if args.record_timings:
        model = TimingModelWrapper(model)

    if args.svg_save_dir is None:

        def aux_func(env, observations, actions, **kwargs):
            if actions is None:
                aux_func.original_pos = np.array(
                    [obs["global_xy"] for obs in observations]
                )
                aux_func.makespan = 0
                aux_func.costs = np.ones(env.get_num_agents())
                aux_func.step_solved_agents = [int(np.sum(env.was_on_goal))]
            else:
                new_pos = np.array([obs["global_xy"] for obs in observations])
                at_goals = np.array(env.was_on_goal)
                aux_func.makespan += 1
                aux_func.original_pos = new_pos
                aux_func.costs[~at_goals] = aux_func.makespan + 1
                aux_func.step_solved_agents.append(int(np.sum(env.was_on_goal)))

    else:
        file_path = pathlib.Path(args.svg_save_dir)
        file_path.mkdir(parents=True, exist_ok=True)

        def _extract_edge_data(gdata):
            if getattr(gdata, "edge_index", None) is not None:
                return {
                    "type": "graph",
                    "edge_index": gdata.edge_index.detach().cpu().numpy(),
                }
            if (
                getattr(gdata, "edge_index_src", None) is not None
                and getattr(gdata, "edge_index_dst", None) is not None
            ):
                edge_data = {
                    "type": "hypergraph",
                    "edge_index_src": gdata.edge_index_src.detach().cpu().numpy(),
                    "edge_index_dst": gdata.edge_index_dst.detach().cpu().numpy(),
                }
                if getattr(gdata, "hton_edge_index_src", None) is not None:
                    edge_data["hton_edge_index_src"] = (
                        gdata.hton_edge_index_src.detach().cpu().numpy()
                    )
                if getattr(gdata, "hton_edge_index_dst", None) is not None:
                    edge_data["hton_edge_index_dst"] = (
                        gdata.hton_edge_index_dst.detach().cpu().numpy()
                    )
                return edge_data
            return {"type": "unknown"}

        def aux_func(env, observations, actions, rtdg, **kwargs):
            if actions is None:
                aux_func.original_pos = np.array(
                    [obs["global_xy"] for obs in observations]
                )
                aux_func.makespan = 0
                aux_func.costs = np.ones(env.get_num_agents())
                aux_func.edge_index = []
                aux_func.step_solved_agents = [int(np.sum(env.was_on_goal))]
            else:
                new_pos = np.array([obs["global_xy"] for obs in observations])
                at_goals = np.array(env.was_on_goal)
                aux_func.makespan += 1
                aux_func.original_pos = new_pos
                aux_func.costs[~at_goals] = aux_func.makespan + 1
                aux_func.step_solved_agents.append(int(np.sum(env.was_on_goal)))
            gdata = rtdg(observations, env)
            aux_func.edge_index.append(_extract_edge_data(gdata))

    num_completed = 0
    num_tested = 0

    all_makespan = []
    all_partial_success_rate = []
    all_sum_of_costs = []

    if args.skip_n is not None:
        seeds = seeds[args.skip_n :]
    if args.subsample_n is not None:
        seeds = seeds[: args.subsample_n]

    use_target_vec = args.use_target_vec
    if use_target_vec is None and args.rl_based_temperature_sampling:
        # Setting use_target_vec so that the actor can use it
        use_target_vec = "target-vec"

    for i, seed in enumerate(seeds):
        instance_seed = int(seed)
        if args.test_fixed_instance_seed is not None:
            instance_seed = int(args.test_fixed_instance_seed)

        sampling_seed = instance_seed
        if args.test_sampling_seed is not None:
            sampling_seed = int(args.test_sampling_seed)

        grid_config = _grid_config_generator(instance_seed)
        object.__setattr__(grid_config, "sampling_seed", sampling_seed)
        success, env, _ = run_model_on_grid(
            model,
            device,
            grid_config,
            args,
            hypergraph_model,
            dataset_kwargs=dataset_kwargs,
            use_target_vec=use_target_vec,
            aux_func=aux_func,
            animation_monitor=args.svg_save_dir is not None,
        )
        makespan = aux_func.makespan
        costs = aux_func.costs

        num_tested += 1
        if success:
            num_completed += 1
        success_rate = num_completed / num_tested
        partial_success_rate = np.mean(env.was_on_goal)
        sum_of_costs = np.sum(costs)

        all_makespan.append(makespan)
        all_partial_success_rate.append(partial_success_rate)
        all_sum_of_costs.append(sum_of_costs)

        results = {
            "success_rate": success_rate,
            "average_makespan": np.mean(all_makespan),
            "average_partial_success_rate": np.mean(all_partial_success_rate),
            "average_sum_of_costs": np.mean(all_sum_of_costs),
            "seed": instance_seed,
            "sampling_seed": sampling_seed,
            "success": success,
            "makespan": makespan,
            "partial_success_rate": partial_success_rate,
            "sum_of_costs": sum_of_costs,
        }
        if args.record_timings:
            timings = np.mean(model.timings)
            model.timings = []
            results = results | {"mean_timings": timings}

        if use_wandb:
            wandb.log(results)

        if args.svg_save_dir is not None:
            file_path = pathlib.Path(f"{args.svg_save_dir}", f"anim_{i}.svg")
            env.save_animation(file_path)

            file_path = pathlib.Path(f"{args.svg_save_dir}", f"edge_index_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(aux_func.edge_index, f)

        if args.step_metrics_csv_path is not None:
            step_metric_rows = _build_step_metrics_rows(
                model_label=_resolve_step_metrics_model_label(args),
                graph_idx=i,
                instance_seed=instance_seed,
                sampling_seed=sampling_seed,
                solved_agents_per_step=aux_func.step_solved_agents,
                total_agents=env.get_num_agents(),
                max_episode_steps=args.test_max_episode_steps,
                rollout_success=success,
                makespan=makespan,
                pad_to_max_episode_steps=args.step_metrics_pad_to_max_episode_steps,
            )
            _append_step_metrics_csv(args.step_metrics_csv_path, step_metric_rows)

        print(
            f"Testing Graph {i + 1}/{len(seeds)}, "
            f"Current Success Rate: {success_rate}"
        )

        if args.record_timings:
            print(f"Timings for Graph {i + 1}: {timings}")
    print("Final results:")
    print(f"Success Rate: {success_rate}")
    print(f"Average Makespan: {np.mean(all_makespan)}")
    print(f"Average Partial Success Rate: {np.mean(all_partial_success_rate)}")
    print(f"Average Sum of Costs: {np.mean(all_sum_of_costs)}")
    if args.record_timings:
        print(f"Average Timings: {np.mean(model.timings)}")


if __name__ == "__main__":
    main()
