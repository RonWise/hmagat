import argparse
import pickle
import pathlib
import numpy as np
import torch
from loguru import logger

from hmagat.run_expert import add_expert_dataset_args, get_expert_dataset_file_name
from hmagat.dataset_loading import load_dataset

from hmagat.convert_to_imitation_dataset import add_imitation_dataset_args
from hmagat.downstream_shards import (
    add_sharded_downstream_args,
    iter_raw_expert_shards,
    write_stage_manifest,
    write_stage_shard,
)
from hmagat.progress_logging import ProgressLogger


def get_pos_file_name(args):
    file_name = get_expert_dataset_file_name(args)
    file_name = file_name[:-4]

    if args.num_neighbour_cutoff is not None:
        file_name += f"_{args.num_neighbour_cutoff}_{args.neighbour_cutoff_method}_neighbour_cutoff"
    if args.use_edge_attr:
        file_name += "_pos"
    if len(file_name) > 0:
        file_name = file_name + ".pkl"
    else:
        file_name = "default.pkl"
    return file_name


def generate_graph_dataset(
    dataset,
    comm_radius,
    obs_radius,
    num_samples,
    save_termination_state,
    use_edge_attr=False,
    print_prefix="",
    id_offset=0,
    num_neighbour_cutoff=None,
    neighbour_cutoff_method=None,
    stack_with_np=True,
):
    dataset_agent_pos = []
    assert use_edge_attr

    actual_num_samples = len(dataset)
    progress = None
    if print_prefix is not None:
        progress = ProgressLogger(
            f"{print_prefix}Position generation".strip(),
            actual_num_samples,
            every_n=max(1, actual_num_samples // 100),
            every_seconds=30.0,
            requested_total=num_samples,
        )

    for id, (sample_observations, actions, terminated) in enumerate(dataset):
        for observations in sample_observations:
            global_xys = np.array([obs["global_xy"] for obs in observations])

            if use_edge_attr:
                dataset_agent_pos.append(global_xys)
        if progress is not None:
            progress.update(
                id + 1,
                extra=f"snapshots={len(dataset_agent_pos)}",
            )
    if not stack_with_np:
        return [torch.from_numpy(np.array(data)) for data in dataset_agent_pos]

    dataset_agent_pos = np.stack(dataset_agent_pos)
    return torch.from_numpy(dataset_agent_pos)


def generate_pos_shards(args):
    entries = []
    for shard_idx, record in enumerate(iter_raw_expert_shards(args)):
        payload = record["payload"]
        positions = generate_graph_dataset(
            payload["dataset"],
            args.comm_radius,
            args.obs_radius,
            args.num_samples,
            args.save_termination_state,
            args.use_edge_attr,
            print_prefix=f"Position shard {shard_idx}: ",
            num_neighbour_cutoff=args.num_neighbour_cutoff,
            neighbour_cutoff_method=args.neighbour_cutoff_method,
            stack_with_np=not args.use_lists,
        )
        entry = write_stage_shard(
            args,
            stage_dir_name="positions",
            stage="positions",
            shard_idx=shard_idx,
            sample_start=payload["sample_start"],
            sample_end=payload["sample_end"],
            payload=positions,
            saved_samples=len(payload["dataset"]),
            snapshot_count=len(positions),
        )
        entries.append(entry)
    return write_stage_manifest(args, "positions", "positions", entries)


def main():
    parser = argparse.ArgumentParser(
        description="Convert to Imitation Learning Dataset"
    )
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_sharded_downstream_args(parser)

    parser.add_argument("--use_edge_attr", action="store_true", default=False)

    args = parser.parse_args()
    logger.info(args)

    if args.use_shards:
        generate_pos_shards(args)
        return

    dataset = load_dataset(
        [get_expert_dataset_file_name], "raw_expert_predictions", args
    )
    if isinstance(dataset, tuple):
        dataset, seed_mask = dataset

    graph_dataset = generate_graph_dataset(
        dataset,
        args.comm_radius,
        args.obs_radius,
        args.num_samples,
        args.save_termination_state,
        args.use_edge_attr,
        num_neighbour_cutoff=args.num_neighbour_cutoff,
        neighbour_cutoff_method=args.neighbour_cutoff_method,
        stack_with_np=not args.use_lists,
    )

    file_name = get_pos_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "positions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((graph_dataset), f)


if __name__ == "__main__":
    main()
