import argparse
import pickle
import pathlib
from loguru import logger

from pogema import pogema_v0, GridConfig

from hmagat.progress_logging import ProgressLogger

from grid_config_generator import add_grid_config_args, grid_config_generator_factory

DATASET_FILE_NAME_DEFAULT = {
    "expert_algorithm": "LaCAM",
    "map_type": "RandomGrid",
    "map_h": 20,
    "map_w": 20,
    "robot_density": 0.025,
    "obstacle_density": 0.1,
    "max_episode_steps": 128,
    "obs_radius": 4,
    "num_samples": 30000,
    "dataset_seed": 42,
    "save_termination_state": True,
    "collision_system": "soft",
    "on_target": "nothing",
    "min_dist": None,
    "max_dist": None,
    "map_types": "random=0.1+maze=0.9",
    "map_w_min": 16,
    "map_w_max": 20,
    "num_agents": "16+24+32",
    "obstacle_density_min": 0.2,
    "obstacle_density_max": 1.0,
    "go_straight_min": 0.75,
    "go_straight_max": 0.85,
    "wall_width_min": 4,
    "wall_width_max": 7,
    "wall_height_min": 2,
    "wall_height_max": 2,
    "side_pad": 2,
    "horizontal_gap": 1,
    "vertical_gap": 3,
    "vertical_gap_min": None,
    "vertical_gap_max": None,
    "num_wall_rows_min": None,
    "num_wall_rows_max": None,
    "num_wall_cols_min": None,
    "num_wall_cols_max": None,
    "wfi_instance": False,
    "block_extra_space": True,
    "room_width_min": 5,
    "room_width_max": 9,
    "room_height_min": 5,
    "room_height_max": 9,
    "num_rows_min": 3,
    "num_rows_max": 5,
    "num_cols_min": 3,
    "num_cols_max": 5,
    "room_grid_uniform": True,
    "regulate_obstacle_density_max": True,
    "maps_name": None,
}
DATASET_FILE_NAME_ALIASES = {
    "expert_algorithm": "ea",
    "map_type": "mtype",
    "map_h": "h",
    "map_w": "w",
    "robot_density": "rd",
    "obstacle_density": "od",
    "max_episode_steps": "maxstep",
    "obs_radius": "obs_r",
    "num_samples": "num_samp",
    "dataset_seed": "dseed",
    "collision_system": "cs",
    "map_types": "mtypes",
    "map_w_min": "wmin",
    "map_w_max": "wmax",
    "obstacle_density_min": "od_min",
    "obstacle_density_max": "od_max",
    "go_straight_min": "go_str_min",
    "go_straight_max": "go_str_max",
    "wall_width_min": "ww_min",
    "wall_width_max": "ww_max",
    "wall_height_min": "wh_min",
    "wall_height_max": "wh_max",
    "side_pad": "side",
    "horizontal_gap": "hg",
    "vertical_gap": "vg",
    "vertical_gap_min": "vgmin",
    "vertical_gap_max": "vgmax",
    "num_wall_rows_min": "nwalls_rmin",
    "num_wall_rows_max": "nwalls_rmax",
    "num_wall_cols_min": "nwalls_cmin",
    "num_wall_cols_max": "nwalls_cmax",
    "wfi_instance": "wfi",
    "block_extra_space": "block_es",
    "room_width_min": "rw_min",
    "room_width_max": "rw_max",
    "room_height_min": "rh_min",
    "room_height_max": "rh_max",
    "num_rows_min": "nr_min",
    "num_rows_max": "nr_max",
    "num_cols_min": "nc_min",
    "num_cols_max": "nc_max",
    "room_grid_uniform": "rg_uniform",
    "regulate_obstacle_density_max": "rod_max",
    "mname": "maps_name",
}

DATASET_FILE_NAME_KEYS = list(DATASET_FILE_NAME_DEFAULT.keys())


def add_expert_dataset_args(parser):
    parser.add_argument("--expert_algorithm", type=str, default="LaCAM")

    parser = add_grid_config_args(parser)

    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--maps_name", type=str, default=None)

    parser.add_argument(
        "--save_termination_state", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--pibt_expert_relevance_training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--override_name", type=str, default=None)
    parser.add_argument("--sample_start", type=int, default=None)
    parser.add_argument("--sample_end", type=int, default=None)
    parser.add_argument("--shard_output_name", type=str, default=None)

    return parser


def get_expert_dataset_file_name(args):
    if args.override_name is not None:
        return f"{args.override_name}.pkl"
    file_name = ""
    dict_args = vars(args)
    for key in sorted(DATASET_FILE_NAME_KEYS):
        if dict_args[key] != DATASET_FILE_NAME_DEFAULT[key]:
            if key in DATASET_FILE_NAME_ALIASES:
                file_name += f"_{DATASET_FILE_NAME_ALIASES[key]}_{dict_args[key]}"
            else:
                file_name += f"_{key}_{dict_args[key]}"
    if not args.ensure_grid_config_is_generatable:
        file_name += "_not_egcg"
    if args.room_only_centre_obstacles:
        file_name += "_cenobs"
    if len(file_name) > 0:
        file_name = file_name[1:] + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


class ExpertWrapper:
    def __init__(self, base_obj, withMaxSteps=False):
        self.base_obj = base_obj
        self.withMaxSteps = withMaxSteps

    def reset_states(self, env):
        self.base_obj.reset_states()
        if self.withMaxSteps:
            self.base_obj.lacam_lib.max_timesteps = env.grid.config.max_episode_steps

    def __getattr__(self, name):
        return getattr(self.base_obj, name)


def wrapped_class(cls, withMaxSteps=False):
    def _get_wrapped_class(config):
        return ExpertWrapper(cls(config), withMaxSteps=withMaxSteps)

    return _get_wrapped_class


def get_expert_algorithm_and_config(args):
    if args.expert_algorithm == "LaCAM":
        from lacam.inference import LacamInference, LacamInferenceConfig

        inference_config = LacamInferenceConfig()
        expert_algorithm = wrapped_class(LacamInference)
    elif args.expert_algorithm[: len("LaCAM-withMaxSteps")] == "LaCAM-withMaxSteps":
        from lacam.inference import LacamInference, LacamInferenceConfig

        lacam_args = args.expert_algorithm.split("-")
        if len(lacam_args) == 2:
            timeouts = [1.0, 5.0, 10.0, 60.0]
        elif len(lacam_args) > 2:
            timeouts = [float(t) for t in lacam_args[2:]]

        inference_config = LacamInferenceConfig(
            max_timesteps=args.max_episode_steps, timeouts=timeouts
        )
        expert_algorithm = wrapped_class(LacamInference, withMaxSteps=True)
    elif args.expert_algorithm[: len("LaCAM")] == "LaCAM":
        from lacam.inference import LacamInference, LacamInferenceConfig

        timeouts = args.expert_algorithm.split("-")[1:]
        timeouts = [float(t) for t in timeouts]

        inference_config = LacamInferenceConfig(
            max_timesteps=args.max_episode_steps,
            time_limit=max(timeouts),
            timeouts=timeouts,
        )
        expert_algorithm = wrapped_class(LacamInference)
    elif args.expert_algorithm == "DCC":
        from dcc.inference import DCCInference, DCCInferenceConfig

        inference_config = DCCInferenceConfig()
        expert_algorithm = wrapped_class(DCCInference)
    elif args.expert_algorithm == "PIBT":
        from pibt.inference import PIBTInference, PIBTInferenceConfig

        inference_config = PIBTInferenceConfig()
        expert_algorithm = PIBTInference
    elif args.expert_algorithm[: len("MAPF-GPT")] == "MAPF-GPT":
        from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

        offset = len("MAPF-GPT") + 1
        pibt_collision_shielding = None
        do_sample = True
        sampling_temperature = 1.0
        model_weight = "6M"
        if args.expert_algorithm[: len("MAPF-GPT-PIBT")] == "MAPF-GPT-PIBT":
            pibt_collision_shielding = "pibt"
            offset = len("MAPF-GPT-PIBT") + 1

        if args.expert_algorithm[offset : offset + len("Det")] == "Det":
            do_sample = False
            offset += len("Det") + 1

        if len(args.expert_algorithm) + 1 > offset:
            gpt_args = args.expert_algorithm[offset:].split("-")
            if len(gpt_args) == 2:
                model_weight = gpt_args[0]
                sampling_temperature = float(gpt_args[1])
            elif len(gpt_args) == 1:
                try:
                    sampling_temperature = float(gpt_args[0])
                except ValueError as exc:
                    logger.warning(
                        "Expert algorithm argument fallback: could not parse "
                        f"{gpt_args[0]!r} as sampling temperature ({exc}); "
                        "using it as model weight."
                    )
                    model_weight = gpt_args[0]
            elif len(gpt_args) > 2:
                raise ValueError(
                    f"Unsupported expert algorithm {args.expert_algorithm}."
                )

        if model_weight == "2MDDG":
            model_weight = "2M-DDG"

        inference_config = MAPFGPTInferenceConfig(
            path_to_weights=f"weights/model-{model_weight}.pt",
            pibt_collision_shielding=pibt_collision_shielding,
            do_sample=do_sample,
            sampling_temperature=sampling_temperature,
        )
        expert_algorithm = MAPFGPTInference
    elif args.expert_algorithm == "SSIL":
        from ssil.inference import SSILInference, SSILInferenceConfig

        inference_config = SSILInferenceConfig()
        expert_algorithm = SSILInference
    elif args.expert_algorithm[: len("SCRIMP")] == "SCRIMP":
        from scrimp.inference import SCRIMPInference, SCRIMPInferenceConfig

        inference_config = SCRIMPInferenceConfig()
        expert_algorithm = SCRIMPInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")
    return expert_algorithm, inference_config


def run_expert_algorithm(
    expert,
    env=None,
    observations=None,
    grid_config=None,
    save_termination_state=True,
    additional_data_func=None,
):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    all_actions = []
    all_observations = []
    all_terminated = []
    additional_data = []

    expert.reset_states(env)

    while True:
        actions = expert.act(observations)

        all_observations.append(observations)
        all_actions.append(actions)

        if additional_data_func is not None:
            additional_data.append(
                additional_data_func(
                    env=env, observations=observations, actions=actions
                )
            )

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if save_termination_state:
            all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    if additional_data_func is not None:
        return all_actions, all_observations, all_terminated, additional_data
    return all_actions, all_observations, all_terminated


def main():
    parser = argparse.ArgumentParser(description="Run Expert")
    parser = add_expert_dataset_args(parser)

    args = parser.parse_args()
    logger.info(args)

    from hmagat.expert_shards import (
        build_global_seeds,
        default_shard_output_name,
        is_shard_mode,
        validate_sample_range,
        write_expert_shard,
    )

    if args.map_dir is not None:
        assert args.maps_name is not None

    seeds = build_global_seeds(args.dataset_seed, args.num_samples)
    sample_start, sample_end = validate_sample_range(
        args.sample_start,
        args.sample_end,
        args.num_samples,
    )
    seeds = seeds[sample_start:sample_end]

    _grid_config_generator = grid_config_generator_factory(args)

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    dataset = []
    seed_mask = []
    num_success = 0
    shard_mode = is_shard_mode(args)
    progress_label = "Expert generation"
    if shard_mode:
        progress_label += f" shard [{sample_start}, {sample_end})"
    progress = ProgressLogger(
        progress_label,
        len(seeds),
        every_n=1,
        every_seconds=30.0,
        requested_total=args.num_samples if shard_mode else None,
    )
    for i, seed in enumerate(seeds):
        grid_config = _grid_config_generator(seed)
        expert = expert_algorithm(inference_config)

        all_actions, all_observations, all_terminated = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=args.save_termination_state,
        )

        if all(all_terminated[-1]):
            seed_mask.append(True)
            num_success += 1
            if args.save_termination_state:
                dataset.append((all_observations, all_actions, all_terminated))
            else:
                dataset.append((all_observations, all_actions))
        else:
            seed_mask.append(False)

        progress.update(
            i + 1,
            extra=(
                f"global_sample={sample_start + i}, "
                f"success_rate={num_success / (i + 1):.6f}, "
                f"last_success={seed_mask[-1]}, saved_samples={len(dataset)}"
            ),
        )

    logger.info(
        f"{len(dataset)}/{len(seeds)} samples were successfully added to the dataset"
    )

    if shard_mode:
        shard_output_name = args.shard_output_name
        if shard_output_name is None:
            shard_output_name = default_shard_output_name(args, sample_start, sample_end)
        write_expert_shard(
            args,
            dataset=dataset,
            seed_mask=seed_mask,
            sample_start=sample_start,
            sample_end=sample_end,
            shard_output_name=shard_output_name,
        )
    else:
        file_name = get_expert_dataset_file_name(args)
        path = pathlib.Path(
            f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((dataset, seed_mask), f)


if __name__ == "__main__":
    main()
