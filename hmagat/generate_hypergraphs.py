import argparse
import pickle
import pathlib
import numpy as np
import time
from loguru import logger

from pogema import pogema_v0

from hmagat.run_expert import get_expert_dataset_file_name, add_expert_dataset_args
from hmagat.dataset_loading import load_dataset
from hmagat.downstream_shards import (
    add_sharded_downstream_args,
    iter_raw_expert_shards,
    successful_shard_seeds,
    write_stage_manifest,
    write_stage_shard,
)
from hmagat.progress_logging import ProgressLogger

from grid_config_generator import grid_config_generator_factory

from hmagat.hypergraph_gen_strategies.base import (
    HyperedgeGenerator,
    ShortestDistanceSampleCliquesHyperedgeGenerator,
    kMeansHyperedgeGenerator,
    LloydsHyperedgeGenerator,
    InfrequentHyperedgeGenerator,
)

HYPERGRAPH_FILE_NAME_DEFAULTS = {
    "hypergraph_comm_radius": 2,
    "hypergraph_max_group_size": None,
    "hyperedge_generation_method": "kmeans",
    "add_hypergraph_self_loop": False,
    "comm_self": True,
    "hypergraph_max_neighbours": None,
    "hypergraph_max_dist_threshold": 3,
    "hypergraph_max_dist_frac": 0.8,
    "hypergraph_max_clique_size": 4,
    "hypergraph_initial_colperc": 0.1,
    "hypergraph_final_colperc": 0.1,
    "hypergraph_wait_one": False,
    "hypergraph_time_period": 1,
    "hypergraph_num_updates": 100,
}
HYPERGRAPH_FILE_NAME_ALIASES = {
    "hyperedge_generation_method": "gen",
    "add_hypergraph_self_loop": "slf",
    "comm_self": "cs",
    "hypergraph_max_neighbours": "hmn",
    "hypergraph_max_dist_threshold": "hmdt",
    "hypergraph_max_dist_frac": "hmdf",
    "hypergraph_max_clique_size": "hmcs",
    "hypergraph_initial_colperc": "hicp",
    "hypergraph_final_colperc": "hfcp",
    "hypergraph_wait_one": "hwo",
    "hypergraph_time_period": "htp",
    "hypergraph_num_updates": "hnu",
    "hypergraph_comm_radius": "hgd",
    "hypergraph_max_group_size": "hgms",
}
HYPERGRAPH_FILE_NAME_KEYS = list(HYPERGRAPH_FILE_NAME_DEFAULTS.keys())


def add_hypergraph_generation_args(parser):
    parser.add_argument("--hypergraph_comm_radius", type=int, default=2)
    parser.add_argument(
        "--take_all_seeds", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--hypergraph_max_group_size", type=int, default=None)
    parser.add_argument("--hyperedge_generation_method", type=str, default="kmeans")
    parser.add_argument(
        "--add_hypergraph_self_loop",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--comm_self", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--hypergraph_max_neighbours", type=int, default=None)

    parser.add_argument("--hypergraph_max_dist_threshold", type=int, default=3)
    parser.add_argument("--hypergraph_max_dist_frac", type=float, default=0.8)
    parser.add_argument("--hypergraph_max_clique_size", type=int, default=4)
    parser.add_argument("--hypergraph_initial_colperc", type=float, default=0.1)
    parser.add_argument("--hypergraph_final_colperc", type=float, default=0.1)
    parser.add_argument(
        "--hypergraph_wait_one", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--hypergraph_time_period", type=int, default=1)
    parser.add_argument("--hypergraph_num_updates", type=int, default=100)

    return parser


def get_hypergraph_file_name(args):
    dict_args = vars(args)

    file_name = get_expert_dataset_file_name(args)
    file_name = file_name[:-4]
    if file_name == "default":
        file_name = ""

    for key in sorted(HYPERGRAPH_FILE_NAME_KEYS):
        if dict_args[key] != HYPERGRAPH_FILE_NAME_DEFAULTS[key]:
            if key in HYPERGRAPH_FILE_NAME_ALIASES:
                file_name += f"_{HYPERGRAPH_FILE_NAME_ALIASES[key]}_{dict_args[key]}"
            else:
                file_name += f"_{key}_{dict_args[key]}"

    if len(file_name) > 0:
        if file_name[0] == "_":
            file_name = file_name[1:]
        file_name = file_name + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


def get_hypergraph_indices_generator(
    hypergraph_comm_radius,
    max_group_size,
    hyperedge_generation_method="proximity",
    comm_self=True,
    hypergraph_max_neighbours=None,
    max_dist_threshold=3,
    max_dist_frac=0.8,
    max_clique_size=4,
    initial_colour_percentage=0.1,
    final_colour_percentage=0.1,
    only_wait_for_atleast_one_colour=False,
    add_hypergraph_self_loop=False,
    hypergraph_num_updates=100,
    hypergraph_time_period=1,
) -> HyperedgeGenerator:
    if hyperedge_generation_method == "kmeans":
        generator = kMeansHyperedgeGenerator(
            comm_radius=hypergraph_comm_radius,
            comm_self=comm_self,
            hypergraph_max_neighbours=hypergraph_max_neighbours,
            add_hypergraph_self_loop=add_hypergraph_self_loop,
            initial_colour_percentage=initial_colour_percentage,
            final_colour_percentage=final_colour_percentage,
            num_updates=hypergraph_num_updates,
            only_wait_for_atleast_one_colour=only_wait_for_atleast_one_colour,
        )
    elif hyperedge_generation_method == "lloyds":
        generator = LloydsHyperedgeGenerator(
            comm_radius=hypergraph_comm_radius,
            comm_self=comm_self,
            hypergraph_max_neighbours=hypergraph_max_neighbours,
            add_hypergraph_self_loop=add_hypergraph_self_loop,
            initial_colour_percentage=initial_colour_percentage,
            final_colour_percentage=final_colour_percentage,
            num_updates=hypergraph_num_updates,
            only_wait_for_atleast_one_colour=only_wait_for_atleast_one_colour,
        )
    elif hyperedge_generation_method == "shortest-distance":
        generator = ShortestDistanceSampleCliquesHyperedgeGenerator(
            comm_radius=hypergraph_comm_radius,
            comm_self=comm_self,
            hypergraph_max_neighbours=hypergraph_max_neighbours,
            max_dist_threshold=max_dist_threshold,
            max_dist_frac=max_dist_frac,
            max_clique_size=max_clique_size,
            add_hypergraph_self_loop=add_hypergraph_self_loop,
            recommended_hyperedges=max_group_size,
        )
    else:
        raise ValueError(
            f"Unknown hyperedge generation method: {hyperedge_generation_method}."
        )

    if hypergraph_time_period > 1:
        generator = InfrequentHyperedgeGenerator(
            hyperedge_generator=generator, time_period=hypergraph_time_period
        )

    return generator


def generate_hypergraphs_for_dataset(args, dataset, seeds, print_prefix=""):
    _grid_config_generator = grid_config_generator_factory(args)

    hyperedge_generator = get_hypergraph_indices_generator(
        hypergraph_comm_radius=args.hypergraph_comm_radius,
        max_group_size=args.hypergraph_max_group_size,
        hyperedge_generation_method=args.hyperedge_generation_method,
        comm_self=args.comm_self,
        hypergraph_max_neighbours=args.hypergraph_max_neighbours,
        max_dist_threshold=args.hypergraph_max_dist_threshold,
        max_dist_frac=args.hypergraph_max_dist_frac,
        max_clique_size=args.hypergraph_max_clique_size,
        initial_colour_percentage=args.hypergraph_initial_colperc,
        final_colour_percentage=args.hypergraph_final_colperc,
        only_wait_for_atleast_one_colour=args.hypergraph_wait_one,
        add_hypergraph_self_loop=args.add_hypergraph_self_loop,
        hypergraph_num_updates=args.hypergraph_num_updates,
        hypergraph_time_period=args.hypergraph_time_period,
    )

    all_ntoh = []
    all_hton_index = []
    total_samples = len(dataset)
    progress = ProgressLogger(
        f"{print_prefix}Hypergraph generation".strip(),
        total_samples,
        every_n=max(1, total_samples // 100),
        every_seconds=30.0,
        requested_total=args.num_samples,
    )
    for sample_num, (seed, data) in enumerate(zip(seeds, dataset)):
        grid_config = _grid_config_generator(seed)

        env = pogema_v0(grid_config)
        _, _ = env.reset()
        _, all_actions, _ = data

        hyperedge_generator.reset_state(env)

        for actions in all_actions:
            ntoh_index, hton_index = hyperedge_generator(env)

            all_ntoh.append(ntoh_index)
            all_hton_index.append(hton_index)
            env.step(actions)
        progress.update(
            sample_num + 1,
            extra=f"snapshots={len(all_ntoh)}",
        )
    return all_ntoh, all_hton_index


def generate_hypergraph_shards(args):
    entries = []
    for shard_idx, record in enumerate(iter_raw_expert_shards(args)):
        payload = record["payload"]
        hypergraphs = generate_hypergraphs_for_dataset(
            args,
            payload["dataset"],
            successful_shard_seeds(args, payload),
            print_prefix=f"Hypergraph shard {shard_idx}: ",
        )
        snapshot_count = len(hypergraphs[0])
        entry = write_stage_shard(
            args,
            stage_dir_name="hypergraphs",
            stage="hypergraphs",
            shard_idx=shard_idx,
            sample_start=payload["sample_start"],
            sample_end=payload["sample_end"],
            payload=hypergraphs,
            saved_samples=len(payload["dataset"]),
            snapshot_count=snapshot_count,
        )
        entries.append(entry)
    return write_stage_manifest(args, "hypergraphs", "hypergraphs", entries)


def main():
    parser = argparse.ArgumentParser(description="Generate Hypergraphs")
    parser = add_expert_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_sharded_downstream_args(parser)

    args = parser.parse_args()
    logger.info(args)

    if args.use_shards:
        generate_hypergraph_shards(args)
        return

    dataset = load_dataset(
        [get_expert_dataset_file_name], "raw_expert_predictions", args
    )

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset
        seeds = seeds[seed_mask]
    elif not args.take_all_seeds:
        raise ValueError("Dataset is expected to have a seed_mask.")

    all_hypergraphs = generate_hypergraphs_for_dataset(args, dataset, seeds)

    file_name = get_hypergraph_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "hypergraphs", f"{file_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_hypergraphs, f)


if __name__ == "__main__":
    main()
