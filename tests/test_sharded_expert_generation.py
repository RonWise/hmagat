import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from loguru import logger

from hmagat import audit_sequence_dataset
from hmagat.compare_processed_dataset_shards import compare_processed_dataset_shards
from hmagat.convert_to_imitation_dataset import generate_processed_dataset_shards
from hmagat.downstream_shards import (
    iter_raw_expert_shards,
    load_stage_manifest,
    validate_stage_manifests,
    write_stage_shard,
    write_stage_manifest,
)
from hmagat.expert_shards import (
    build_global_seeds,
    default_shard_output_name,
    merge_expert_shards,
    split_sample_ranges,
    write_expert_shard,
)
from hmagat.generate_expert_sharded import (
    _build_worker_command,
    _is_tracked_resource_process,
)


def _args(tmp_dir, name="merged", num_samples=6, dataset_seed=42):
    return SimpleNamespace(
        dataset_dir=tmp_dir,
        override_name=name,
        num_samples=num_samples,
        dataset_seed=dataset_seed,
    )


def _processed_args(tmp_dir, name="merged", num_samples=6, dataset_seed=42):
    args = _args(tmp_dir, name=name, num_samples=num_samples, dataset_seed=dataset_seed)
    args.comm_radius = 7
    args.obs_radius = 1
    args.save_termination_state = True
    args.expert_algorithm = "LaCAM"
    args.obstacle_density_max = 0.7
    args.ensure_grid_config_is_generatable = True
    args.use_edge_attr = False
    args.num_neighbour_cutoff = None
    args.neighbour_cutoff_method = "closest"
    args.distance_metric = "euclidean"
    args.random_edge_probs = None
    args.use_lists = True
    return args


def _positions_args(tmp_dir, name="merged", num_samples=6, dataset_seed=42):
    args = _processed_args(
        tmp_dir,
        name=name,
        num_samples=num_samples,
        dataset_seed=dataset_seed,
    )
    args.use_edge_attr = True
    return args


def _observation(agent_idx):
    obstacles = np.zeros((3, 3), dtype=np.float32)
    agents = np.zeros((3, 3), dtype=np.float32)
    return {
        "obstacles": obstacles,
        "agents": agents,
        "global_xy": np.array([agent_idx, 0]),
        "global_target_xy": np.array([agent_idx, 1]),
    }


def _sample(num_steps=2, num_agents=2):
    observations = [
        [_observation(agent_idx) for agent_idx in range(num_agents)]
        for _ in range(num_steps)
    ]
    actions = [np.zeros(num_agents, dtype=np.int64) for _ in range(num_steps)]
    terminated = [np.zeros(num_agents, dtype=bool) for _ in range(num_steps)]
    return observations, actions, terminated


def _merge_dense_datasets(dense_datasets):
    merged = []
    for field_idx in range(len(dense_datasets[0])):
        field = []
        for dense_dataset in dense_datasets:
            field.extend(dense_dataset[field_idx])
        merged.append(field)
    return merged


def _write_shard(
    tmp_dir,
    *,
    name,
    sample_start,
    sample_end,
    seed_mask,
    dataset=None,
    num_samples=6,
    dataset_seed=42,
):
    args = _args(tmp_dir, name="merged", num_samples=num_samples, dataset_seed=dataset_seed)
    if dataset is None:
        dataset = [f"sample_{idx}" for idx, ok in enumerate(seed_mask) if ok]
    return write_expert_shard(
        args,
        dataset=dataset,
        seed_mask=seed_mask,
        sample_start=sample_start,
        sample_end=sample_end,
        shard_output_name=name,
    )


class ShardedExpertGenerationTest(unittest.TestCase):
    def test_split_sample_ranges_covers_all_samples_without_overlap(self):
        self.assertEqual(
            split_sample_ranges(num_samples=10, num_shards=3),
            [(0, 4), (4, 7), (7, 10)],
        )

    def test_split_sample_ranges_rejects_more_shards_than_samples(self):
        records = []
        sink_id = logger.add(
            lambda message: records.append(message.record),
            level="WARNING",
            format="{message}",
        )
        try:
            with self.assertRaises(ValueError):
                split_sample_ranges(num_samples=2, num_shards=3)
        finally:
            logger.remove(sink_id)
        warnings = [record["message"] for record in records]

        self.assertTrue(any("num_shards" in message for message in warnings))

    def test_build_global_seeds_is_deterministic_and_sliceable(self):
        seeds = build_global_seeds(dataset_seed=42, num_samples=10)
        same_seeds = build_global_seeds(dataset_seed=42, num_samples=10)

        self.assertEqual(seeds.tolist(), same_seeds.tolist())
        self.assertEqual(seeds[3:7].tolist(), same_seeds[3:7].tolist())

    def test_merge_expert_shards_writes_legacy_dataset_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="shard_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, False],
                dataset=["sample_0"],
            )
            _write_shard(
                tmp_dir,
                name="shard_01.pkl",
                sample_start=2,
                sample_end=6,
                seed_mask=[True, True, False, True],
                dataset=["sample_2", "sample_3", "sample_5"],
            )

            output_path = merge_expert_shards(_args(tmp_dir), expected_num_shards=2)

            with open(output_path, "rb") as f:
                dataset, seed_mask = pickle.load(f)

            self.assertEqual(dataset, ["sample_0", "sample_2", "sample_3", "sample_5"])
            self.assertEqual(seed_mask, [True, False, True, True, False, True])
            self.assertEqual(
                output_path,
                os.path.join(tmp_dir, "raw_expert_predictions", "merged.pkl"),
            )

    def test_merge_rejects_gap_and_logs_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="shard_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, True],
            )
            _write_shard(
                tmp_dir,
                name="shard_01.pkl",
                sample_start=3,
                sample_end=6,
                seed_mask=[True, True, True],
            )

            records = []
            sink_id = logger.add(
                lambda message: records.append(message.record),
                level="WARNING",
                format="{message}",
            )
            try:
                with self.assertRaises(ValueError):
                    merge_expert_shards(_args(tmp_dir), expected_num_shards=2)
            finally:
                logger.remove(sink_id)
            warnings = [record["message"] for record in records]

            self.assertTrue(any("gap" in message.lower() for message in warnings))

    def test_merge_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="shard_00.pkl",
                sample_start=0,
                sample_end=3,
                seed_mask=[True, True, True],
            )
            _write_shard(
                tmp_dir,
                name="shard_01.pkl",
                sample_start=2,
                sample_end=6,
                seed_mask=[True, True, True, True],
            )

            with self.assertRaises(ValueError):
                merge_expert_shards(_args(tmp_dir), expected_num_shards=2)

    def test_merge_rejects_mismatched_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="shard_00.pkl",
                sample_start=0,
                sample_end=3,
                seed_mask=[True, True, True],
                dataset_seed=42,
            )
            _write_shard(
                tmp_dir,
                name="shard_01.pkl",
                sample_start=3,
                sample_end=6,
                seed_mask=[True, True, True],
                dataset_seed=7,
            )

            with self.assertRaises(ValueError):
                merge_expert_shards(_args(tmp_dir), expected_num_shards=2)

    def test_worker_command_preserves_boolean_optional_false_values(self):
        args = SimpleNamespace(
            dataset_dir="/tmp/dataset",
            override_name="paper",
            num_samples=8,
            dataset_seed=42,
            logs_dir="/tmp/dataset/logs",
            num_workers=2,
            poll_seconds=1.0,
            resource_monitor=True,
            resource_monitor_interval=30.0,
            sample_start=None,
            sample_end=None,
            shard_output_name=None,
            save_termination_state=True,
            block_extra_space=False,
            ensure_grid_config_is_generatable=True,
            regulate_obstacle_density_max=False,
            wfi_instance=False,
        )

        command = _build_worker_command(args, shard_idx=0, sample_start=0, sample_end=4)

        self.assertIn("--save_termination_state", command)
        self.assertIn("paper_raw_shard_000_00000_00004.pkl", command)
        self.assertIn("--no-block_extra_space", command)
        self.assertIn("--no-regulate_obstacle_density_max", command)
        self.assertNotIn("--wfi_instance", command)
        self.assertNotIn("--resource_monitor", command)
        self.assertNotIn("--resource_monitor_interval", command)

    def test_default_raw_shard_name_includes_stage(self):
        args = _args("/tmp/dataset", name="paper", num_samples=128)

        self.assertEqual(
            default_shard_output_name(args, sample_start=0, sample_end=64),
            "paper_raw_shard_00000_00064.pkl",
        )

    def test_iter_raw_expert_shards_yields_coverage_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="raw_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, False],
                dataset=["sample_0"],
                num_samples=4,
            )
            _write_shard(
                tmp_dir,
                name="raw_01.pkl",
                sample_start=2,
                sample_end=4,
                seed_mask=[True, True],
                dataset=["sample_2", "sample_3"],
                num_samples=4,
            )

            records = iter_raw_expert_shards(_args(tmp_dir, num_samples=4))

            first = next(records)
            self.assertEqual(first["payload"]["sample_start"], 0)
            second = next(records)
            self.assertEqual(second["payload"]["sample_start"], 2)
            with self.assertRaises(StopIteration):
                next(records)

    def test_resource_process_filter_matches_only_real_python_entrypoints(self):
        dataset_dir = "/workspace/datasets/example"

        self.assertTrue(
            _is_tracked_resource_process(
                "/opt/conda/bin/python -m hmagat.run_expert "
                "--dataset_dir /workspace/datasets/example",
                dataset_dir,
            )
        )
        self.assertTrue(
            _is_tracked_resource_process(
                "python -m hmagat.generate_expert_sharded "
                "--dataset_dir /workspace/datasets/example",
                dataset_dir,
            )
        )
        self.assertFalse(
            _is_tracked_resource_process(
                "bash -lc python - <<'PY' hmagat.run_expert "
                "/workspace/datasets/example",
                dataset_dir,
            )
        )
        self.assertFalse(
            _is_tracked_resource_process(
                "python -m hmagat.run_expert --dataset_dir /workspace/other",
                dataset_dir,
            )
        )

    def test_sharded_processed_generation_writes_stage_named_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="raw_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, False],
                dataset=[_sample(num_steps=2)],
                num_samples=4,
            )
            _write_shard(
                tmp_dir,
                name="raw_01.pkl",
                sample_start=2,
                sample_end=4,
                seed_mask=[True, True],
                dataset=[_sample(num_steps=1), _sample(num_steps=3)],
                num_samples=4,
            )

            manifest = generate_processed_dataset_shards(
                _processed_args(tmp_dir, name="paper", num_samples=4)
            )

            self.assertEqual(manifest["stage"], "processed")
            self.assertEqual(manifest["total_saved_samples"], 3)
            self.assertEqual(manifest["total_snapshots"], 6)
            self.assertEqual(len(manifest["entries"]), 2)
            self.assertTrue(
                all("_processed_shard_" in entry["file_name"] for entry in manifest["entries"])
            )

            manifest_path = os.path.join(
                tmp_dir,
                "processed_dataset",
                "shards",
                "manifest.json",
            )
            with open(manifest_path) as f:
                manifest_on_disk = json.load(f)
            self.assertEqual(manifest_on_disk["total_snapshots"], 6)

            with open(manifest["entries"][1]["path"], "rb") as f:
                dense_dataset = pickle.load(f)
            self.assertEqual(
                [item.item() for item in dense_dataset[4]],
                [1, 2, 2, 2],
            )
            self.assertIsInstance(dense_dataset[0][0], torch.Tensor)
            self.assertEqual(
                audit_sequence_dataset._as_tensor(np.array([1, 2])).tolist(),
                [1, 2],
            )

            argv = [
                "audit_sequence_dataset",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "paper",
                "--num_samples",
                "4",
                "--obs_radius",
                "1",
                "--save_termination_state",
                "--expert_algorithm",
                "LaCAM",
                "--obstacle_density_max",
                "0.7",
                "--ensure_grid_config_is_generatable",
                "--use_lists",
                "--use_shards",
                "--validation_fraction",
                "0.25",
                "--test_fraction",
                "0.25",
            ]
            with mock.patch("sys.argv", argv):
                audit_sequence_dataset.main()

            stamp_path = os.path.join(
                tmp_dir,
                "processed_dataset",
                "shards",
                "sequence_audit.json",
            )
            self.assertTrue(os.path.exists(stamp_path))
            with open(stamp_path) as f:
                stamp = json.load(f)
            self.assertEqual(stamp["status"], "passed")
            self.assertTrue(stamp["checked_train"])
            self.assertFalse(stamp["checked_validation"])

    def test_sharded_audit_accepts_separate_positions_manifest_edge_attr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="raw_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, False],
                dataset=[_sample(num_steps=2)],
                num_samples=4,
            )
            _write_shard(
                tmp_dir,
                name="raw_01.pkl",
                sample_start=2,
                sample_end=4,
                seed_mask=[True, True],
                dataset=[_sample(num_steps=1), _sample(num_steps=3)],
                num_samples=4,
            )

            processed_manifest = generate_processed_dataset_shards(
                _processed_args(tmp_dir, name="paper", num_samples=4)
            )
            position_entries = []
            positions_args = _positions_args(tmp_dir, name="paper", num_samples=4)
            for entry in processed_manifest["entries"]:
                with open(entry["path"], "rb") as f:
                    dense_dataset = pickle.load(f)
                positions = list(dense_dataset[0])
                position_entries.append(
                    write_stage_shard(
                        positions_args,
                        stage_dir_name="positions",
                        stage="positions",
                        shard_idx=entry["shard_idx"],
                        sample_start=entry["sample_start"],
                        sample_end=entry["sample_end"],
                        payload=positions,
                        saved_samples=entry["saved_samples"],
                        snapshot_count=entry["snapshot_count"],
                    )
                )
            write_stage_manifest(positions_args, "positions", "positions", position_entries)

            argv = [
                "audit_sequence_dataset",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "paper",
                "--num_samples",
                "4",
                "--obs_radius",
                "1",
                "--save_termination_state",
                "--expert_algorithm",
                "LaCAM",
                "--obstacle_density_max",
                "0.7",
                "--ensure_grid_config_is_generatable",
                "--use_edge_attr",
                "--use_lists",
                "--load_positions_separately",
                "--use_shards",
                "--validation_fraction",
                "0.25",
                "--test_fraction",
                "0.25",
            ]
            with mock.patch("sys.argv", argv):
                audit_sequence_dataset.main()

    def test_sharded_audit_rejects_materialized_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "audit_sequence_dataset",
                "--dataset_dir",
                tmp_dir,
                "--override_name",
                "paper",
                "--num_samples",
                "4",
                "--obs_radius",
                "1",
                "--save_termination_state",
                "--expert_algorithm",
                "LaCAM",
                "--obstacle_density_max",
                "0.7",
                "--ensure_grid_config_is_generatable",
                "--use_shards",
                "--materialize_audit_dataset",
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaisesRegex(
                    ValueError,
                    "Materialized sequence audit is not implemented for sharded datasets",
                ):
                    audit_sequence_dataset.main()

    def test_compare_processed_dataset_shards_accepts_matching_legacy_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="raw_00.pkl",
                sample_start=0,
                sample_end=2,
                seed_mask=[True, False],
                dataset=[_sample(num_steps=2)],
                num_samples=4,
            )
            _write_shard(
                tmp_dir,
                name="raw_01.pkl",
                sample_start=2,
                sample_end=4,
                seed_mask=[True, True],
                dataset=[_sample(num_steps=1), _sample(num_steps=3)],
                num_samples=4,
            )

            args = _processed_args(tmp_dir, name="paper", num_samples=4)
            manifest = generate_processed_dataset_shards(args)
            dense_datasets = []
            for entry in manifest["entries"]:
                with open(entry["path"], "rb") as f:
                    dense_datasets.append(pickle.load(f))
            legacy_dataset = _merge_dense_datasets(dense_datasets)
            legacy_path = os.path.join(tmp_dir, "processed_dataset", "paper.pkl")
            with open(legacy_path, "wb") as f:
                pickle.dump(legacy_dataset, f)

            summary = compare_processed_dataset_shards(args)

            self.assertEqual(summary["snapshots"], 6)
            self.assertEqual(summary["shards"], 2)

    def test_compare_processed_dataset_shards_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_shard(
                tmp_dir,
                name="raw_00.pkl",
                sample_start=0,
                sample_end=1,
                seed_mask=[True],
                dataset=[_sample(num_steps=1)],
                num_samples=2,
            )
            _write_shard(
                tmp_dir,
                name="raw_01.pkl",
                sample_start=1,
                sample_end=2,
                seed_mask=[True],
                dataset=[_sample(num_steps=1)],
                num_samples=2,
            )

            args = _processed_args(tmp_dir, name="paper", num_samples=2)
            manifest = generate_processed_dataset_shards(args)
            with open(manifest["entries"][0]["path"], "rb") as f:
                first_dense_dataset = pickle.load(f)
            legacy_dataset = _merge_dense_datasets([first_dense_dataset])
            legacy_dataset[4][0] = torch.as_tensor(99)
            legacy_path = os.path.join(tmp_dir, "processed_dataset", "paper.pkl")
            with open(legacy_path, "wb") as f:
                pickle.dump(legacy_dataset, f)

            with self.assertRaises(ValueError):
                compare_processed_dataset_shards(args)

    def test_stage_manifest_rejects_override_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _processed_args(tmp_dir, name="paper", num_samples=4)
            entries = [
                {
                    "stage": "processed",
                    "path": os.path.join(tmp_dir, "dummy.pkl"),
                    "file_name": "dummy.pkl",
                    "shard_idx": 0,
                    "sample_start": 0,
                    "sample_end": 4,
                    "saved_samples": 4,
                    "snapshot_count": 8,
                    "graph_map_id_start": 0,
                    "graph_map_id_end": 4,
                    "num_samples": 4,
                    "dataset_seed": 42,
                    "override_name": "paper",
                }
            ]
            write_stage_manifest(args, "processed_dataset", "processed", entries)

            mismatched_args = _processed_args(tmp_dir, name="other", num_samples=4)
            with self.assertRaises(ValueError):
                load_stage_manifest(
                    mismatched_args,
                    "processed_dataset",
                    expected_stage="processed",
                )

    def test_stage_manifest_rejects_generation_arg_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _processed_args(tmp_dir, name="paper", num_samples=4)
            entries = [
                {
                    "stage": "processed",
                    "path": os.path.join(tmp_dir, "dummy.pkl"),
                    "file_name": "dummy.pkl",
                    "shard_idx": 0,
                    "sample_start": 0,
                    "sample_end": 4,
                    "saved_samples": 4,
                    "snapshot_count": 8,
                    "graph_map_id_start": 0,
                    "graph_map_id_end": 4,
                    "num_samples": 4,
                    "dataset_seed": 42,
                    "override_name": "paper",
                }
            ]
            write_stage_manifest(args, "processed_dataset", "processed", entries)

            mismatched_args = _processed_args(tmp_dir, name="paper", num_samples=4)
            mismatched_args.use_lists = False
            with self.assertRaises(ValueError):
                load_stage_manifest(
                    mismatched_args,
                    "processed_dataset",
                    expected_stage="processed",
                )

    def test_stage_manifest_rejects_missing_generation_arg_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _processed_args(tmp_dir, name="paper", num_samples=4)
            entries = [
                {
                    "stage": "processed",
                    "path": os.path.join(tmp_dir, "dummy.pkl"),
                    "file_name": "dummy.pkl",
                    "shard_idx": 0,
                    "sample_start": 0,
                    "sample_end": 4,
                    "saved_samples": 4,
                    "snapshot_count": 8,
                    "graph_map_id_start": 0,
                    "graph_map_id_end": 4,
                    "num_samples": 4,
                    "dataset_seed": 42,
                    "override_name": "paper",
                }
            ]
            write_stage_manifest(args, "processed_dataset", "processed", entries)
            manifest_path = os.path.join(
                tmp_dir, "processed_dataset", "shards", "manifest.json"
            )
            with open(manifest_path) as f:
                manifest = json.load(f)
            del manifest["generation_args"]["use_lists"]
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            records = []
            sink_id = logger.add(
                lambda message: records.append(message.record),
                level="WARNING",
                format="{message}",
            )
            try:
                with self.assertRaises(ValueError):
                    load_stage_manifest(
                        args,
                        "processed_dataset",
                        expected_stage="processed",
                    )
            finally:
                logger.remove(sink_id)
            warnings = [record["message"] for record in records]

            self.assertTrue(any("missing generation args" in msg for msg in warnings))

    def test_validate_stage_manifests_rejects_snapshot_mismatch(self):
        base = {
            "total_saved_samples": 4,
            "total_snapshots": 8,
            "entries": [
                {
                    "sample_start": 0,
                    "sample_end": 4,
                    "saved_samples": 4,
                    "snapshot_count": 8,
                }
            ],
        }
        mismatched = {
            "total_saved_samples": 4,
            "total_snapshots": 7,
            "entries": [
                {
                    "sample_start": 0,
                    "sample_end": 4,
                    "saved_samples": 4,
                    "snapshot_count": 7,
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_stage_manifests(
                {
                    "processed_dataset": base,
                    "hypergraphs": mismatched,
                }
            )


if __name__ == "__main__":
    unittest.main()
