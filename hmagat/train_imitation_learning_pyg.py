import argparse
import csv
import hashlib
import json
import pickle
import pathlib
import numpy as np
import wandb
import time
import random
import queue as queue_module
from contextlib import nullcontext
from dataclasses import dataclass

import multiprocessing as mp
from itertools import compress
from collections import OrderedDict

import torch
import torch.optim as optim
from loguru import logger

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

from hmagat.downstream_shards import add_sharded_downstream_args
from hmagat.training_args import add_training_args, validate_training_args_contract
from hmagat.convert_to_imitation_dataset import (
    add_imitation_dataset_args,
    generate_graph_dataset,
    get_imitation_dataset_file_name,
)
from hmagat.generate_hypergraphs import (
    add_hypergraph_generation_args,
    get_hypergraph_indices_generator,
    get_hypergraph_file_name,
)
from hmagat.generate_pos import get_pos_file_name
from hmagat.run_expert import (
    get_expert_algorithm_and_config,
    run_expert_algorithm,
    add_expert_dataset_args,
)
from hmagat.imitation_dataset_pyg import (
    MAPFGraphDataset,
    MAPFHypergraphDataset,
    MAPFSequenceDataset,
    SequenceBatchCollator,
    collate_mapf_sequences,
    validate_sequence_dataset_assumptions,
)
from hmagat.sequence_training import SequenceStepRuntimeTracker, compute_sequence_loss
from hmagat.sharded_training_dataset import (
    ShardedEpisodeBatchSampler,
    ShardedEpisodeSampler,
    ShardedSequenceDataset,
    build_sharded_snapshot_datasets,
    load_sequence_audit_stamp,
    sequence_audit_stamp_path,
    training_index_path,
)

from hmagat.modules.model.run_model import run_model_on_grid
from grid_config_generator import (
    grid_config_generator_factory,
    generate_grid_config_from_env,
)

from hmagat.loss import get_loss_function

from hmagat.generate_additional_data import (
    get_additional_data_file_name,
    add_additional_data_args,
    any_additional_data,
    generate_additional_data,
)

from hmagat.generate_expert_makespans import check_or_create_expert_makespans

from hmagat.lr_scheduler import get_lr_scheduler
from hmagat.checkpointing import (
    TRAINING_CHECKPOINT_TYPE,
    TRAINING_CHECKPOINT_VERSION,
    extract_model_state_dict_from_checkpoint,
)
from hmagat.dataset_loading import load_dataset
from hmagat.progress_logging import ProgressLogger


def _clone_metric_dict(metrics):
    cloned = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


class HyperedgeIndicesGenerator:
    def __init__(
        self,
        hypergraph_comm_radius,
        max_group_size,
        hyperedge_generation_method,
        comm_self,
        hypergraph_max_neighbours,
        max_dist_threshold,
        max_dist_frac,
        max_clique_size,
        initial_colour_percentage,
        final_colour_percentage,
        only_wait_for_atleast_one_colour,
        add_hypergraph_self_loop,
        hypergraph_num_updates,
        hypergraph_time_period,
    ):
        self.hyperedge_generator = get_hypergraph_indices_generator(
            hypergraph_comm_radius=hypergraph_comm_radius,
            max_group_size=max_group_size,
            hyperedge_generation_method=hyperedge_generation_method,
            comm_self=comm_self,
            hypergraph_max_neighbours=hypergraph_max_neighbours,
            max_dist_threshold=max_dist_threshold,
            max_dist_frac=max_dist_frac,
            max_clique_size=max_clique_size,
            initial_colour_percentage=initial_colour_percentage,
            final_colour_percentage=final_colour_percentage,
            only_wait_for_atleast_one_colour=only_wait_for_atleast_one_colour,
            add_hypergraph_self_loop=add_hypergraph_self_loop,
            hypergraph_num_updates=hypergraph_num_updates,
            hypergraph_time_period=hypergraph_time_period,
        )
        self.initialized = False

    def __call__(self, env, observations, actions):
        if not self.initialized:
            self.hyperedge_generator.reset_state(env)
            self.initialized = True
        return self.hyperedge_generator(env)


def aux_func(env, observations, actions, **kwargs):
    if actions is None:
        aux_func.original_pos = np.array([obs["global_xy"] for obs in observations])
        aux_func.makespan = 0
        aux_func.costs = np.ones(env.get_num_agents())
    else:
        new_pos = np.array([obs["global_xy"] for obs in observations])
        at_goals = np.array(env.was_on_goal)
        aux_func.makespan += 1
        aux_func.original_pos = new_pos
        aux_func.costs[~at_goals] = aux_func.makespan + 1


def aux_func_train(env, observations, actions, oe_period=None, **kwargs):
    if oe_period is not None:
        aux_func_train.oe_period = oe_period
        return
    if actions is None:
        aux_func_train.grid_configs = []
        aux_func_train.original_pos = np.array(
            [obs["global_xy"] for obs in observations]
        )
        aux_func_train.makespan = 0
        aux_func_train.costs = np.ones(env.get_num_agents())
    else:
        new_pos = np.array([obs["global_xy"] for obs in observations])
        at_goals = np.array(env.was_on_goal)
        aux_func_train.makespan += 1
        aux_func_train.original_pos = new_pos
        aux_func_train.costs[~at_goals] = aux_func_train.makespan + 1
        if aux_func_train.makespan % aux_func_train.oe_period == 0:
            aux_func_train.grid_configs.append(generate_grid_config_from_env(env))


def _effective_num_batches(total_batches, max_batches):
    if max_batches is None:
        return total_batches
    return min(total_batches, max_batches)


def _batch_limit_reached(processed_batches, max_batches):
    return max_batches is not None and processed_batches >= max_batches


def _validate_positive_batch_limit(value, arg_name):
    if value is None:
        return
    if value <= 0:
        logger.warning(f"{arg_name} must be positive when set; got {value}.")
        raise ValueError(f"{arg_name} must be positive when set")


def _validate_and_warn_batch_limits(args):
    _validate_positive_batch_limit(args.max_train_batches, "--max_train_batches")
    _validate_positive_batch_limit(
        args.max_validation_batches, "--max_validation_batches"
    )

    if args.max_train_batches is not None:
        logger.warning(
            "Limiting training to first "
            f"{args.max_train_batches} batches per epoch because "
            "--max_train_batches was set. This is a pilot/debug mode and "
            "must not be reported as full-dataset training."
        )
        if args.run_online_expert:
            logger.warning(
                "--max_train_batches does not limit online expert augmentation "
                "batches; disable --run_online_expert for pilot runs."
            )

    if args.max_validation_batches is not None:
        logger.warning(
            "Limiting validation accuracy to first "
            f"{args.max_validation_batches} batches because "
            "--max_validation_batches was set. This is a pilot/debug mode and "
            "must not be reported as full validation accuracy."
        )
        if args.skip_validation or args.skip_validation_accuracy:
            logger.warning(
                f"--max_validation_batches={args.max_validation_batches} has no "
                "effect because validation accuracy is disabled."
            )


def _build_dataloader_kwargs(args):
    if args.dataloader_num_workers < 0:
        logger.warning(
            "--dataloader_num_workers must be non-negative; got "
            f"{args.dataloader_num_workers}."
        )
        raise ValueError("--dataloader_num_workers must be non-negative")

    if args.dataloader_prefetch_factor is not None:
        if args.dataloader_num_workers <= 0:
            logger.warning(
                "--dataloader_prefetch_factor requires dataloader_num_workers > 0."
            )
            raise ValueError(
                "--dataloader_prefetch_factor requires dataloader_num_workers > 0"
            )
        if args.dataloader_prefetch_factor <= 0:
            logger.warning(
                "--dataloader_prefetch_factor must be positive when set; got "
                f"{args.dataloader_prefetch_factor}."
            )
            raise ValueError(
                "--dataloader_prefetch_factor must be positive when set"
            )

    if args.dataloader_persistent_workers and args.dataloader_num_workers <= 0:
        logger.warning(
            "--dataloader_persistent_workers requires dataloader_num_workers > 0."
        )
        raise ValueError(
            "--dataloader_persistent_workers requires dataloader_num_workers > 0"
        )

    kwargs = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = args.dataloader_persistent_workers
        if args.dataloader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    return kwargs


def _sharded_sequence_loader_risk_warnings(args):
    if not (args.use_shards and args.sequence_training):
        return []

    warnings = []
    if args.dataloader_num_workers > 0:
        warnings.append(
            "Sharded sequence training with dataloader_num_workers > 0 may "
            "duplicate shard bundle loads across worker processes and inflate RAM."
        )
    if args.dataloader_persistent_workers:
        warnings.append(
            "Sharded sequence training with dataloader_persistent_workers enabled "
            "may retain duplicated shard caches across workers."
        )
    if args.dataloader_prefetch_factor is not None:
        warnings.append(
            "Sharded sequence training with dataloader_prefetch_factor set may "
            "increase prefetched shard duplication and host RAM usage."
        )
    return warnings


def _warn_on_sharded_sequence_loader_profile(args):
    for message in _sharded_sequence_loader_risk_warnings(args):
        logger.warning(message)


def _validate_amp_args(args, device):
    if not args.amp:
        return

    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning(
            "Automatic mixed precision requires a CUDA device, but current "
            f"device is {device}."
        )
        raise ValueError("Automatic mixed precision requires a CUDA device")

    if args.amp_dtype not in {"float16", "bfloat16"}:
        logger.warning(f"Unsupported AMP dtype requested: {args.amp_dtype}.")
        raise ValueError(f"Unsupported AMP dtype: {args.amp_dtype}")

    if args.amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "Requested AMP dtype bfloat16 is not supported on the active CUDA "
            "device."
        )
        raise ValueError("Requested AMP dtype bfloat16 is not supported")


def _get_amp_dtype(args):
    if args.amp_dtype == "float16":
        return torch.float16
    if args.amp_dtype == "bfloat16":
        return torch.bfloat16
    logger.warning(f"Unsupported AMP dtype requested: {args.amp_dtype}.")
    raise ValueError(f"Unsupported AMP dtype: {args.amp_dtype}")


def _autocast_context(args, device):
    if not args.amp:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_get_amp_dtype(args))


def _create_grad_scaler(args):
    if not args.amp or args.amp_dtype != "float16":
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def _parse_grad_clip_norm_type(value):
    if isinstance(value, str):
        if value.lower() == "inf":
            return float("inf")
        return float(value)
    return float(value)


def _apply_gradient_clipping(
    args,
    model,
    *,
    optimizer,
    grad_scaler,
    global_step,
):
    if args.grad_clip_value is None:
        return None
    if args.grad_clip_value <= 0:
        logger.warning(
            "--grad_clip_value must be positive when enabled; got "
            f"{args.grad_clip_value}."
        )
        raise ValueError("--grad_clip_value must be positive when enabled")
    if args.grad_clip_warmup_steps is not None:
        if args.grad_clip_warmup_steps < 0:
            logger.warning(
                "--grad_clip_warmup_steps must be non-negative when set; got "
                f"{args.grad_clip_warmup_steps}."
            )
            raise ValueError("--grad_clip_warmup_steps must be non-negative when set")
        if global_step < args.grad_clip_warmup_steps:
            return None
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    if args.grad_clip_type == "norm":
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.grad_clip_value,
            norm_type=_parse_grad_clip_norm_type(args.grad_clip_norm),
        )
    if args.grad_clip_type == "value":
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            clip_value=args.grad_clip_value,
        )
        return float(args.grad_clip_value)
    logger.warning(
        "Unsupported gradient clipping type requested: "
        f"{args.grad_clip_type}."
    )
    raise ValueError(f"Unsupported gradient clipping type: {args.grad_clip_type}")


def _validate_checkpoint_source_args(args):
    if args.resume_checkpoint_path is None:
        return

    conflicting = []
    if args.pretrain_weights_path is not None:
        conflicting.append("--pretrain_weights_path")
    if args.load_partial_parameters_path is not None:
        conflicting.append("--load_partial_parameters_path")
    if conflicting:
        logger.warning(
            "--resume_checkpoint_path must not be combined with "
            + ", ".join(conflicting)
            + "."
        )
        raise ValueError(
            "--resume_checkpoint_path must not be combined with other weight-loading flags"
        )


def _load_resume_training_checkpoint(checkpoint_path, *, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_type") != TRAINING_CHECKPOINT_TYPE
    ):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} requires a training checkpoint "
            "payload with optimizer and scheduler state."
        )
        raise ValueError(
            "Resume checkpoint requires a training checkpoint payload"
        )

    required_keys = {
        "format_version",
        "checkpoint_type",
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "epoch",
        "args",
        "rng_state",
        "dataset_state",
        "training_state",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        logger.warning(
            f"Resume checkpoint {checkpoint_path} is missing required keys: "
            f"{missing_keys}."
        )
        raise ValueError("Resume checkpoint is missing required keys")
    if checkpoint["format_version"] != TRAINING_CHECKPOINT_VERSION:
        logger.warning(
            f"Resume checkpoint {checkpoint_path} format_version mismatch: "
            f"{checkpoint['format_version']} != {TRAINING_CHECKPOINT_VERSION}."
        )
        raise ValueError("Resume checkpoint format_version mismatch")
    if not isinstance(checkpoint["args"], dict):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} has a non-dict args payload."
        )
        raise ValueError("Resume checkpoint has a non-dict args payload")
    if not isinstance(checkpoint["training_state"], dict):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} has a non-dict training_state payload."
        )
        raise ValueError("Resume checkpoint has a non-dict training_state payload")
    if not isinstance(checkpoint["rng_state"], dict):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} has a non-dict rng_state payload."
        )
        raise ValueError("Resume checkpoint has a non-dict rng_state payload")
    required_rng_keys = {
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "cuda_rng_state_all",
    }
    missing_rng_keys = sorted(required_rng_keys - set(checkpoint["rng_state"]))
    if missing_rng_keys:
        logger.warning(
            f"Resume checkpoint {checkpoint_path} is missing rng_state keys: "
            f"{missing_rng_keys}."
        )
        raise ValueError("Resume checkpoint is missing rng_state keys")
    if not isinstance(checkpoint["dataset_state"], dict):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} has a non-dict dataset_state payload."
        )
        raise ValueError("Resume checkpoint has a non-dict dataset_state payload")
    required_dataset_state_keys = {
        "mode",
        "train_id_max",
        "validation_id_max",
        "train_dataset_len",
        "validation_dataset_len",
        "stages",
    }
    missing_dataset_state_keys = sorted(
        required_dataset_state_keys - set(checkpoint["dataset_state"])
    )
    if missing_dataset_state_keys:
        logger.warning(
            f"Resume checkpoint {checkpoint_path} is missing dataset_state keys: "
            f"{missing_dataset_state_keys}."
        )
        raise ValueError("Resume checkpoint is missing dataset_state keys")
    if not isinstance(checkpoint["dataset_state"]["stages"], dict):
        logger.warning(
            f"Resume checkpoint {checkpoint_path} has a non-dict dataset_state.stages payload."
        )
        raise ValueError(
            "Resume checkpoint has a non-dict dataset_state.stages payload"
        )
    required_training_state_keys = {
        "best_validation_success_rate",
        "best_validation_accuracy",
        "best_val_file_name",
        "cur_validation_id_max",
        "threshold_val_success_rate",
        "oe_improve_quality",
        "cs_warmup_freeze_baseline",
        "cs_warmup_old_decoder_input_size",
        "cs_warmup_decoder_prefix_snapshot",
    }
    missing_training_state_keys = sorted(
        required_training_state_keys - set(checkpoint["training_state"])
    )
    if missing_training_state_keys:
        logger.warning(
            f"Resume checkpoint {checkpoint_path} is missing training_state keys: "
            f"{missing_training_state_keys}."
        )
        raise ValueError("Resume checkpoint is missing training_state keys")
    extract_model_state_dict_from_checkpoint(
        checkpoint, checkpoint_path, print_prefix="[resume] "
    )
    return checkpoint


_RESUME_ALLOWED_ARG_DIFFS = {
    "resume_checkpoint_path",
    "checkpoints_dir",
    "run_name",
    "wandb_project",
    "wandb_entity",
    "tensorboard_dir",
    "tensorboard_flush_secs",
    "device",
    "save_intmd_checkpoints",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "amp",
    "amp_dtype",
}


def _validate_resume_checkpoint_args(args, resume_checkpoint):
    saved_args = resume_checkpoint["args"]
    saved_epoch = int(resume_checkpoint["epoch"])
    saved_training_state = resume_checkpoint["training_state"]
    saved_num_epochs = int(saved_args["num_epochs"])
    current_num_epochs = int(args.num_epochs)
    if saved_args.get("run_online_expert", False):
        logger.warning(
            "Resume from checkpoints created with --run_online_expert is not "
            "supported because online expert buffers are not checkpointed."
        )
        raise ValueError(
            "Resume from checkpoints created with --run_online_expert is not supported"
        )
    if bool(saved_training_state.get("cs_warmup_freeze_baseline", False)) != bool(
        args.cs_warmup_freeze_baseline
    ):
        logger.warning(
            "Resume checkpoint warmup mode mismatch: checkpoint="
            f"{saved_training_state.get('cs_warmup_freeze_baseline', False)} "
            f"current={args.cs_warmup_freeze_baseline}."
        )
        raise ValueError("Resume checkpoint warmup mode mismatch")

    mismatches = []
    for key, saved_value in saved_args.items():
        if key in _RESUME_ALLOWED_ARG_DIFFS:
            continue
        if key == "num_epochs":
            continue
        if not hasattr(args, key):
            mismatches.append((key, saved_value, "<missing>"))
            continue
        current_value = getattr(args, key)
        if current_value != saved_value:
            mismatches.append((key, saved_value, current_value))

    if mismatches:
        logger.warning("Resume checkpoint argument compatibility check failed.")
        for key, saved_value, current_value in mismatches:
            logger.warning(
                f"  {key}: checkpoint={saved_value} current={current_value}"
            )
        raise ValueError("Resume checkpoint arguments are incompatible")

    if current_num_epochs < saved_num_epochs:
        logger.warning(
            "Resume checkpoint num_epochs cannot be reduced: "
            f"checkpoint={saved_num_epochs} current={current_num_epochs}."
        )
        raise ValueError("Resume checkpoint num_epochs cannot be reduced")

    if current_num_epochs > saved_num_epochs:
        logger.warning(
            "Resume checkpoint extending total training horizon: "
            f"checkpoint_num_epochs={saved_num_epochs} "
            f"current_num_epochs={current_num_epochs}."
        )

    start_epoch = saved_epoch + 1
    if start_epoch >= args.num_epochs:
        logger.warning(
            f"Resume checkpoint already completed epoch {saved_epoch}, but "
            f"num_epochs={args.num_epochs} leaves no epochs to run."
        )
        raise ValueError("Resume checkpoint leaves no epochs to run")


@dataclass
class CSWarmupFreezeBaselineState:
    decoder_weight: torch.nn.Parameter
    old_decoder_input_size: int
    decoder_prefix_snapshot: torch.Tensor

    def restore_decoder_prefix(self):
        with torch.no_grad():
            self.decoder_weight[:, : self.old_decoder_input_size].copy_(
                self.decoder_prefix_snapshot
            )


def _validate_cs_warmup_freeze_baseline(
    args, partial_load_summary, resume_training_state=None
):
    if not args.cs_warmup_freeze_baseline:
        return

    if args.coordination_state_size <= 0:
        logger.warning(
            "--cs_warmup_freeze_baseline requires coordination_state_size > 0."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires coordination_state_size > 0"
        )

    if getattr(args, "resume_checkpoint_path", None) is not None:
        if resume_training_state is None:
            logger.warning(
                "--cs_warmup_freeze_baseline resume requires training_state from "
                "the resume checkpoint."
            )
            raise ValueError(
                "--cs_warmup_freeze_baseline resume requires checkpoint training_state"
            )
        if not resume_training_state.get("cs_warmup_freeze_baseline", False):
            logger.warning(
                "--cs_warmup_freeze_baseline resume requires a checkpoint created "
                "with the same warmup mode."
            )
            raise ValueError(
                "--cs_warmup_freeze_baseline resume requires a warmup checkpoint"
            )
        return

    if args.load_partial_parameters_path is None:
        logger.warning(
            "--cs_warmup_freeze_baseline requires --load_partial_parameters_path."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires --load_partial_parameters_path"
        )

    if partial_load_summary is None:
        logger.warning(
            "--cs_warmup_freeze_baseline requires a partial-load summary after "
            "load_partial_state_dict."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires a partial-load summary"
        )

    if "actionsMLP.0.weight" not in partial_load_summary.get("widened_linear", []):
        logger.warning(
            "--cs_warmup_freeze_baseline requires widened partial loading for "
            "actionsMLP.0.weight."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires widened partial loading for "
            "actionsMLP.0.weight"
        )


def _apply_cs_warmup_freeze_baseline(
    model, args, partial_load_summary, resume_training_state=None
):
    _validate_cs_warmup_freeze_baseline(
        args, partial_load_summary, resume_training_state=resume_training_state
    )
    if not args.cs_warmup_freeze_baseline:
        return None

    if model.coordination_state_cell is None:
        logger.warning(
            "--cs_warmup_freeze_baseline expected coordination_state_cell to be "
            "initialized, but the model does not have one."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires an initialized "
            "coordination_state_cell"
        )

    decoder_weight = model.actionsMLP[0].weight
    old_decoder_input_size = (
        model.actionsMLP[0].in_features - model.coordination_state_size
    )
    if old_decoder_input_size <= 0:
        logger.warning(
            "--cs_warmup_freeze_baseline computed a non-positive number of "
            "baseline decoder columns."
        )
        raise ValueError(
            "--cs_warmup_freeze_baseline requires baseline decoder columns"
        )

    if getattr(args, "resume_checkpoint_path", None) is not None:
        saved_old_decoder_input_size = resume_training_state.get(
            "cs_warmup_old_decoder_input_size"
        )
        if saved_old_decoder_input_size != old_decoder_input_size:
            logger.warning(
                "--cs_warmup_freeze_baseline resume decoder width mismatch: "
                f"{saved_old_decoder_input_size} != {old_decoder_input_size}."
            )
            raise ValueError(
                "--cs_warmup_freeze_baseline resume decoder width mismatch"
            )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.coordination_state_cell.parameters():
        param.requires_grad = True
    decoder_weight.requires_grad = True

    def _mask_old_decoder_columns(grad):
        if grad is None:
            return None
        grad = grad.clone()
        grad[:, :old_decoder_input_size] = 0
        return grad

    decoder_weight.register_hook(_mask_old_decoder_columns)
    if getattr(args, "resume_checkpoint_path", None) is not None:
        decoder_prefix_snapshot = resume_training_state.get(
            "cs_warmup_decoder_prefix_snapshot"
        )
        if not torch.is_tensor(decoder_prefix_snapshot):
            logger.warning(
                "--cs_warmup_freeze_baseline resume checkpoint lacks a valid "
                "decoder prefix snapshot."
            )
            raise ValueError(
                "--cs_warmup_freeze_baseline resume checkpoint lacks decoder prefix snapshot"
            )
        decoder_prefix_snapshot = decoder_prefix_snapshot.to(
            device=decoder_weight.device, dtype=decoder_weight.dtype
        )
    else:
        decoder_prefix_snapshot = decoder_weight[
            :, :old_decoder_input_size
        ].detach().clone()
    state = CSWarmupFreezeBaselineState(
        decoder_weight=decoder_weight,
        old_decoder_input_size=old_decoder_input_size,
        decoder_prefix_snapshot=decoder_prefix_snapshot,
    )
    state.restore_decoder_prefix()

    trainable_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    frozen_names = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]
    logger.warning("CS warmup freeze baseline enabled.")
    logger.warning(
        "Trainable parameters: " + ", ".join(trainable_names)
    )
    logger.warning(
        "Frozen parameters: " + ", ".join(frozen_names)
    )
    logger.warning(
        "Decoder mask: actionsMLP.0.weight "
        f"old_columns={old_decoder_input_size} "
        f"new_columns={model.coordination_state_size}"
    )
    logger.warning("Decoder prefix restore enabled.")
    return state


def _create_optimizer(args, model, cs_warmup_freeze_state=None):
    if cs_warmup_freeze_state is None:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        return optim.Adam(
            trainable_params, lr=args.lr_start, weight_decay=args.weight_decay
        )

    decoder_weight = cs_warmup_freeze_state.decoder_weight
    other_trainable_params = [
        param
        for param in model.parameters()
        if param.requires_grad and param is not decoder_weight
    ]
    param_groups = []
    if other_trainable_params:
        param_groups.append(
            {"params": other_trainable_params, "weight_decay": args.weight_decay}
        )
    param_groups.append({"params": [decoder_weight], "weight_decay": 0.0})
    return optim.Adam(param_groups, lr=args.lr_start, weight_decay=args.weight_decay)


def _capture_rng_state():
    rng_state = {
        "torch_rng_state": torch.get_rng_state().cpu(),
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "cuda_rng_state_all": None,
    }
    if torch.cuda.is_available():
        rng_state["cuda_rng_state_all"] = [
            state.cpu() for state in torch.cuda.get_rng_state_all()
        ]
    return rng_state


def _restore_rng_state(rng_state):
    required_keys = {
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "cuda_rng_state_all",
    }
    missing_keys = sorted(required_keys - set(rng_state))
    if missing_keys:
        logger.warning(
            f"Resume checkpoint RNG state is missing required keys: {missing_keys}."
        )
        raise ValueError("Resume checkpoint RNG state is missing required keys")

    torch_rng_state = rng_state["torch_rng_state"]
    if not torch.is_tensor(torch_rng_state):
        logger.warning("Resume checkpoint torch_rng_state must be a tensor.")
        raise ValueError("Resume checkpoint torch_rng_state must be a tensor")
    torch.set_rng_state(torch_rng_state.cpu())

    np.random.set_state(rng_state["numpy_random_state"])
    random.setstate(rng_state["python_random_state"])

    cuda_rng_state_all = rng_state["cuda_rng_state_all"]
    if cuda_rng_state_all is not None:
        if not torch.cuda.is_available():
            logger.warning(
                "Resume checkpoint contains CUDA RNG state, but CUDA is unavailable."
            )
            raise ValueError(
                "Resume checkpoint contains CUDA RNG state, but CUDA is unavailable"
            )
        normalized_cuda_rng_state_all = []
        for idx, state in enumerate(cuda_rng_state_all):
            if not torch.is_tensor(state):
                logger.warning(
                    "Resume checkpoint CUDA RNG state entry "
                    f"{idx} must be a tensor."
                )
                raise ValueError(
                    "Resume checkpoint CUDA RNG state entries must be tensors"
                )
            normalized_cuda_rng_state_all.append(
                state.detach().cpu().to(dtype=torch.uint8)
            )
        torch.cuda.set_rng_state_all(normalized_cuda_rng_state_all)


def _resolve_dataset_source_path(funcs, dir_name, args):
    last_path = None
    for func in funcs:
        path = pathlib.Path(args.dataset_dir, dir_name, func(args))
        last_path = path
        if path.exists():
            return path
        logger.warning(
            f"Could not find dataset file {path}. Trying legacy file name fallback."
        )
    raise FileNotFoundError(f"Could not find any dataset file. Last path: {last_path}")


def _file_signature(path):
    path = pathlib.Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _manifest_stage_dataset_state(manifest):
    return {
        "total_saved_samples": manifest.get("total_saved_samples"),
        "total_snapshots": manifest.get("total_snapshots"),
        "entries": [
            {
                "file_name": entry.get("file_name"),
                "sample_start": entry.get("sample_start"),
                "sample_end": entry.get("sample_end"),
                "saved_samples": entry.get("saved_samples"),
                "snapshot_count": entry.get("snapshot_count"),
                "graph_map_id_start": entry.get("graph_map_id_start"),
                "graph_map_id_end": entry.get("graph_map_id_end"),
                "path_signature": _file_signature(entry["path"]),
            }
            for entry in manifest.get("entries", [])
        ],
    }


def _build_sharded_dataset_state(
    args,
    train_dataset,
    *,
    validation_dataset,
    train_id_max,
    validation_id_max,
):
    manifests = train_dataset.training_index.manifests
    return {
        "mode": "sharded",
        "train_id_max": int(train_id_max),
        "validation_id_max": int(validation_id_max),
        "train_dataset_len": len(train_dataset),
        "validation_dataset_len": len(validation_dataset),
        "training_index": _file_signature(training_index_path(args)),
        "stages": {
            stage_name: _manifest_stage_dataset_state(manifest)
            for stage_name, manifest in sorted(manifests.items())
        },
    }


def _validation_rollout_graph_ids(
    *,
    use_shards,
    train_id_max,
    cur_validation_id_max,
    validation_id_max,
    validation_dataset,
):
    if not use_shards:
        return list(range(train_id_max, cur_validation_id_max)), int(
            validation_id_max - train_id_max
        )

    if not hasattr(validation_dataset, "episode_original_sample_ids"):
        logger.warning(
            "Sharded validation rollout requires validation_dataset to expose "
            "episode_original_sample_ids."
        )
        raise ValueError(
            "Sharded validation rollout requires episode_original_sample_ids"
        )

    requested_validation_episodes = int(cur_validation_id_max - train_id_max)
    if requested_validation_episodes < 0:
        logger.warning(
            "Validation rollout requested a negative number of validation "
            f"episodes: {requested_validation_episodes}."
        )
        raise ValueError("Validation rollout requested a negative number of episodes")

    total_validation_graphs = len(validation_dataset.episode_original_sample_ids)
    if requested_validation_episodes > total_validation_graphs:
        logger.warning(
            "Validation rollout requested more episodes than available in the "
            "sharded validation split: "
            f"requested={requested_validation_episodes}, "
            f"available={total_validation_graphs}."
        )
        raise ValueError("Validation rollout requested more episodes than available")

    return (
        list(validation_dataset.episode_original_sample_ids[:requested_validation_episodes]),
        total_validation_graphs,
    )


def _build_unsharded_dataset_state(
    args,
    *,
    hypergraph_model,
    load_additional_data,
    train_dataset,
    validation_dataset,
    train_id_max,
    validation_id_max,
):
    stages = {
        "processed_dataset": [
            _file_signature(
                _resolve_dataset_source_path(
                    [get_imitation_dataset_file_name],
                    "processed_dataset",
                    args,
                )
            )
        ]
    }
    if getattr(args, "load_positions_separately", False):
        stages["positions"] = [
            _file_signature(
                _resolve_dataset_source_path([get_pos_file_name], "positions", args)
            )
        ]
    if hypergraph_model:
        stages["hypergraphs"] = [
            _file_signature(
                _resolve_dataset_source_path(
                    [get_hypergraph_file_name], "hypergraphs", args
                )
            )
        ]
    if load_additional_data:
        stages["additional_data"] = [
            _file_signature(
                _resolve_dataset_source_path(
                    [get_additional_data_file_name], "additional_data", args
                )
            )
        ]
    return {
        "mode": "unsharded",
        "train_id_max": int(train_id_max),
        "validation_id_max": int(validation_id_max),
        "train_dataset_len": len(train_dataset),
        "validation_dataset_len": len(validation_dataset),
        "stages": stages,
    }


def _dataset_state_fingerprint(dataset_state):
    return hashlib.sha256(
        json.dumps(dataset_state, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validate_resume_dataset_state(current_dataset_state, saved_dataset_state):
    if not isinstance(current_dataset_state, dict) or not isinstance(
        saved_dataset_state, dict
    ):
        logger.warning("Resume dataset binding requires dict dataset_state payloads.")
        raise ValueError("Resume checkpoint dataset binding mismatch")
    if current_dataset_state == saved_dataset_state:
        return

    logger.warning(
        "Resume checkpoint dataset binding mismatch: "
        f"checkpoint_fingerprint={_dataset_state_fingerprint(saved_dataset_state)} "
        f"current_fingerprint={_dataset_state_fingerprint(current_dataset_state)}."
    )
    raise ValueError("Resume checkpoint dataset binding mismatch")


def _build_training_checkpoint_payload(
    *,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    best_validation_success_rate,
    best_validation_accuracy,
    best_val_file_name,
    cur_validation_id_max,
    threshold_val_success_rate,
    oe_improve_quality,
    dataset_state,
    cs_warmup_freeze_state,
    grad_scaler=None,
):
    training_state = {
        "best_validation_success_rate": best_validation_success_rate,
        "best_validation_accuracy": best_validation_accuracy,
        "best_val_file_name": best_val_file_name,
        "cur_validation_id_max": cur_validation_id_max,
        "threshold_val_success_rate": threshold_val_success_rate,
        "oe_improve_quality": oe_improve_quality,
        "cs_warmup_freeze_baseline": bool(args.cs_warmup_freeze_baseline),
        "cs_warmup_old_decoder_input_size": (
            cs_warmup_freeze_state.old_decoder_input_size
            if cs_warmup_freeze_state is not None
            else None
        ),
        "cs_warmup_decoder_prefix_snapshot": (
            cs_warmup_freeze_state.decoder_prefix_snapshot.detach().cpu()
            if cs_warmup_freeze_state is not None
            else None
        ),
    }
    return {
        "format_version": TRAINING_CHECKPOINT_VERSION,
        "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "grad_scaler_state_dict": (
            grad_scaler.state_dict() if grad_scaler is not None else None
        ),
        "epoch": int(epoch),
        "args": dict(vars(args)),
        "rng_state": _capture_rng_state(),
        "dataset_state": dataset_state,
        "training_state": training_state,
    }


def _save_training_checkpoint(
    checkpoint_path,
    *,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    best_validation_success_rate,
    best_validation_accuracy,
    best_val_file_name,
    cur_validation_id_max,
    threshold_val_success_rate,
    oe_improve_quality,
    dataset_state,
    cs_warmup_freeze_state,
    grad_scaler=None,
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = _build_training_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
        args=args,
        best_validation_success_rate=best_validation_success_rate,
        best_validation_accuracy=best_validation_accuracy,
        best_val_file_name=best_val_file_name,
        cur_validation_id_max=cur_validation_id_max,
        threshold_val_success_rate=threshold_val_success_rate,
        oe_improve_quality=oe_improve_quality,
        dataset_state=dataset_state,
        cs_warmup_freeze_state=cs_warmup_freeze_state,
        grad_scaler=grad_scaler,
    )
    torch.save(checkpoint_payload, checkpoint_path)


def _save_epoch_progress_checkpoints(
    *,
    save_intermediate_checkpoint,
    checkpoints_dir,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    best_validation_success_rate,
    best_validation_accuracy,
    best_val_file_name,
    cur_validation_id_max,
    threshold_val_success_rate,
    oe_improve_quality,
    dataset_state,
    cs_warmup_freeze_state,
    grad_scaler=None,
):
    save_kwargs = dict(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
        args=args,
        best_validation_success_rate=best_validation_success_rate,
        best_validation_accuracy=best_validation_accuracy,
        best_val_file_name=best_val_file_name,
        cur_validation_id_max=cur_validation_id_max,
        threshold_val_success_rate=threshold_val_success_rate,
        oe_improve_quality=oe_improve_quality,
        dataset_state=dataset_state,
        cs_warmup_freeze_state=cs_warmup_freeze_state,
        grad_scaler=grad_scaler,
    )
    if save_intermediate_checkpoint:
        checkpoint_path = pathlib.Path(checkpoints_dir, f"epoch_{epoch}.pt")
        _save_training_checkpoint(checkpoint_path, **save_kwargs)
    checkpoint_path = pathlib.Path(checkpoints_dir, "last.pt")
    _save_training_checkpoint(checkpoint_path, **save_kwargs)


def _restore_resume_training_state(
    model,
    optimizer,
    lr_scheduler,
    resume_checkpoint,
    grad_scaler=None,
):
    saved_args = resume_checkpoint["args"]
    saved_num_epochs = int(saved_args["num_epochs"])
    model_state_dict = extract_model_state_dict_from_checkpoint(
        resume_checkpoint,
        "<resume-checkpoint>",
        print_prefix="[resume] ",
    )
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    saved_lr_scheduler_state = resume_checkpoint["lr_scheduler_state_dict"]
    allow_horizon_extension = False
    if lr_scheduler._step_on_batch:
        allow_horizon_extension = (
            saved_lr_scheduler_state["max_steps"] != lr_scheduler.max_steps
        )
    elif lr_scheduler._step_on_epoch:
        current_num_epochs = int(
            getattr(lr_scheduler.scheduler, "T_max", saved_num_epochs)
        )
        allow_horizon_extension = current_num_epochs != saved_num_epochs
    lr_scheduler.load_state_dict(
        saved_lr_scheduler_state,
        allow_horizon_extension=allow_horizon_extension,
    )
    if allow_horizon_extension:
        if lr_scheduler._step_on_batch:
            logger.warning(
                "Resume checkpoint LR scheduler horizon extended: "
                f"checkpoint_max_steps={saved_lr_scheduler_state['max_steps']} "
                f"current_max_steps={lr_scheduler.max_steps}."
            )
        else:
            logger.warning(
                "Resume checkpoint LR scheduler horizon extended: "
                f"checkpoint_num_epochs={saved_num_epochs} "
                f"current_num_epochs={current_num_epochs}."
            )
    saved_grad_scaler_state = resume_checkpoint.get("grad_scaler_state_dict")
    if grad_scaler is not None:
        if saved_grad_scaler_state is None:
            logger.warning(
                "Resume checkpoint does not contain grad_scaler_state_dict; "
                "starting AMP GradScaler from a fresh state."
            )
        else:
            grad_scaler.load_state_dict(saved_grad_scaler_state)
    elif saved_grad_scaler_state is not None:
        logger.warning(
            "Resume checkpoint contains grad_scaler_state_dict, but current run "
            "does not use float16 AMP; ignoring saved scaler state."
        )
    _restore_rng_state(resume_checkpoint["rng_state"])
    training_state = resume_checkpoint["training_state"]
    return {
        "start_epoch": int(resume_checkpoint["epoch"]) + 1,
        "best_validation_success_rate": float(
            training_state["best_validation_success_rate"]
        ),
        "best_validation_accuracy": float(
            training_state["best_validation_accuracy"]
        ),
        "best_val_file_name": training_state["best_val_file_name"],
        "cur_validation_id_max": int(training_state["cur_validation_id_max"]),
        "threshold_val_success_rate": float(
            training_state["threshold_val_success_rate"]
        ),
        "oe_improve_quality": bool(training_state["oe_improve_quality"]),
        "training_state": training_state,
    }


def _create_tensorboard_writer(args, *, purge_step=None):
    if args.tensorboard_dir is None:
        return None
    if args.tensorboard_dir == "":
        logger.warning(
            "TensorBoard logging requires a non-empty --tensorboard_dir."
        )
        raise ValueError(
            "TensorBoard logging requires a non-empty --tensorboard_dir"
        )
    if args.tensorboard_flush_secs <= 0:
        logger.warning(
            "--tensorboard_flush_secs must be positive; got "
            f"{args.tensorboard_flush_secs}."
        )
        raise ValueError("--tensorboard_flush_secs must be positive")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        logger.warning(
            "TensorBoard logging requested but torch.utils.tensorboard is "
            "unavailable in this environment."
        )
        raise ValueError(
            "TensorBoard logging requested but SummaryWriter is unavailable"
        ) from exc

    tensorboard_dir = pathlib.Path(args.tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        flush_secs=args.tensorboard_flush_secs,
        purge_step=purge_step,
    )
    writer.add_text(
        "run/args_json",
        json.dumps(vars(args), sort_keys=True, indent=2),
        global_step=0,
    )
    writer.add_text("run/checkpoints_dir", str(args.checkpoints_dir), global_step=0)
    if getattr(args, "resume_checkpoint_path", None) is not None:
        writer.add_text(
            "run/resume_checkpoint_path",
            str(args.resume_checkpoint_path),
            global_step=0,
        )
    logger.info(
        f"TensorBoard logging enabled: dir={tensorboard_dir}, purge_step={purge_step}"
    )
    return writer


def _log_training_stage(stage_name, detail=None):
    message = f"[stage] {stage_name}"
    if detail:
        message += f": {detail}"
    logger.info(message)


def _count_trainable_parameters(model):
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _resolve_sequence_dataset_validation_mode(args):
    if not args.sequence_training:
        return "not_applicable", None

    if args.use_shards:
        if args.validate_sequence_training_dataset is True:
            return "deep_preflight", None

        try:
            payload = load_sequence_audit_stamp(args)
        except FileNotFoundError as exc:
            logger.warning(
                "Sharded sequence training requires a valid sequence audit stamp "
                "when --validate_sequence_training_dataset is not explicitly enabled. "
                "Expected stamp path: {}",
                sequence_audit_stamp_path(args),
            )
            raise ValueError(
                "Sharded sequence training requires a prior successful "
                "audit_sequence_dataset --use_shards run, or an explicit "
                "--validate_sequence_training_dataset."
            ) from exc
        except ValueError as exc:
            logger.warning(
                "Sharded sequence audit stamp validation failed for {}.",
                sequence_audit_stamp_path(args),
            )
            raise ValueError(
                "Sharded sequence training requires a valid sequence audit stamp. "
                "Re-run audit_sequence_dataset --use_shards, or pass "
                "--validate_sequence_training_dataset for an explicit deep check."
            ) from exc
        return "audit_stamp", payload

    if args.validate_sequence_training_dataset is False:
        return "skip", None
    return "deep_preflight", None


def _reset_runtime_metrics(*objects):
    for obj in objects:
        if obj is None:
            continue
        reset = getattr(obj, "reset_runtime_stats", None)
        if reset is not None:
            reset()
            continue
        reset = getattr(obj, "reset_metrics", None)
        if reset is not None:
            reset()


def _runtime_metrics_snapshot(obj):
    if obj is None:
        return {}
    snapshot = getattr(obj, "runtime_stats_snapshot", None)
    if snapshot is not None:
        return snapshot()
    snapshot = getattr(obj, "metrics_snapshot", None)
    if snapshot is not None:
        return snapshot()
    return {}


class EpochRuntimeTracker:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.batch_wait_sec = 0.0
        self.backward_sec = 0.0
        self.optimizer_step_sec = 0.0
        self.num_batches = 0

    def record_batch_wait(self, elapsed_sec):
        self.batch_wait_sec += float(elapsed_sec)

    def record_backward(self, elapsed_sec):
        self.backward_sec += float(elapsed_sec)

    def record_optimizer_step(self, elapsed_sec):
        self.optimizer_step_sec += float(elapsed_sec)

    def record_batch(self):
        self.num_batches += 1

    def metrics_snapshot(self):
        return {
            "batch_wait_sec": float(self.batch_wait_sec),
            "backward_sec": float(self.backward_sec),
            "optimizer_step_sec": float(self.optimizer_step_sec),
            "num_batches": int(self.num_batches),
        }


def _mean_batch_compute_sec(step_metrics, epoch_metrics):
    num_batches = int(epoch_metrics.get("num_batches", 0))
    if num_batches <= 0:
        return None
    total_compute_sec = float(step_metrics.get("state_restore_sec", 0.0))
    total_compute_sec += float(step_metrics.get("data_to_device_sec", 0.0))
    total_compute_sec += float(step_metrics.get("forward_sec", 0.0))
    total_compute_sec += float(step_metrics.get("state_stash_sec", 0.0))
    total_compute_sec += float(step_metrics.get("on_step_sec", 0.0))
    total_compute_sec += float(step_metrics.get("loss_compute_sec", 0.0))
    total_compute_sec += float(epoch_metrics.get("backward_sec", 0.0))
    total_compute_sec += float(epoch_metrics.get("optimizer_step_sec", 0.0))
    return total_compute_sec / num_batches


def _log_sequence_runtime_summary(
    label,
    *,
    dataset=None,
    collator=None,
    step_runtime=None,
    epoch_runtime=None,
    first_batch_latency_sec=None,
):
    dataset_metrics = _runtime_metrics_snapshot(dataset)
    collator_metrics = _runtime_metrics_snapshot(collator)
    step_metrics = _runtime_metrics_snapshot(step_runtime)
    epoch_metrics = _runtime_metrics_snapshot(epoch_runtime)
    parts = []
    if first_batch_latency_sec is not None:
        parts.append(f"first_batch_latency_sec={first_batch_latency_sec:.3f}")
    for key in (
        "shard_accesses",
        "shard_loads",
        "shard_switches",
        "unique_shards_touched",
        "shard_load_time_sec",
        "processed_payload_load_sec",
        "positions_payload_load_sec",
        "additional_data_payload_load_sec",
        "hypergraph_payload_load_sec",
        "shard_dataset_materialization_sec",
        "cache_hits",
        "cache_evictions",
        "cache_size",
    ):
        if key in dataset_metrics:
            value = dataset_metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
    for key in ("collate_calls", "collate_time_sec", "last_collate_time_sec"):
        if key in collator_metrics:
            value = collator_metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
    for key in (
        "timesteps",
        "state_restore_sec",
        "data_to_device_sec",
        "forward_sec",
        "state_stash_sec",
        "on_step_sec",
        "loss_compute_sec",
    ):
        if key in step_metrics:
            value = step_metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
    mean_batch_compute_sec = _mean_batch_compute_sec(step_metrics, epoch_metrics)
    for key in ("batch_wait_sec", "backward_sec", "optimizer_step_sec", "num_batches"):
        if key in epoch_metrics:
            value = epoch_metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
    if mean_batch_compute_sec is not None:
        parts.append(f"mean_batch_compute_sec={mean_batch_compute_sec:.3f}")
    if parts:
        logger.info("{} runtime summary: {}", label, ", ".join(parts))
    recent_shard_events = dataset_metrics.get("recent_shard_events")
    if recent_shard_events:
        logger.info(
            "{} shard trace: {}",
            label,
            " | ".join(str(event) for event in recent_shard_events[-8:]),
        )


def _sequence_runtime_metrics(
    *,
    prefix,
    dataset=None,
    collator=None,
    step_runtime=None,
    epoch_runtime=None,
    first_batch_latency_sec=None,
):
    metrics = {}
    if first_batch_latency_sec is not None:
        metrics[f"{prefix}_first_batch_latency_sec"] = float(first_batch_latency_sec)
    for key, value in _runtime_metrics_snapshot(dataset).items():
        if key == "unique_shards_touched":
            metrics[f"{prefix}_{key}"] = float(value)
        elif isinstance(value, (int, float)):
            metrics[f"{prefix}_{key}"] = float(value)
    for key, value in _runtime_metrics_snapshot(collator).items():
        if isinstance(value, (int, float)):
            metrics[f"{prefix}_{key}"] = float(value)
    for key, value in _runtime_metrics_snapshot(step_runtime).items():
        if isinstance(value, (int, float)):
            metrics[f"{prefix}_{key}"] = float(value)
    for key, value in _runtime_metrics_snapshot(epoch_runtime).items():
        if isinstance(value, (int, float)):
            metrics[f"{prefix}_{key}"] = float(value)
    mean_batch_compute_sec = _mean_batch_compute_sec(
        _runtime_metrics_snapshot(step_runtime),
        _runtime_metrics_snapshot(epoch_runtime),
    )
    if mean_batch_compute_sec is not None:
        metrics[f"{prefix}_mean_batch_compute_sec"] = float(mean_batch_compute_sec)
    return metrics


def _normalize_tensorboard_scalar(name, value):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            logger.warning(
                f"TensorBoard metric {name} must be a scalar tensor; got shape "
                f"{tuple(value.shape)}."
            )
            raise ValueError("TensorBoard metrics must be scalar tensors")
        return float(value.item())

    logger.warning(
        f"TensorBoard metric {name} must be numeric; got {type(value).__name__}."
    )
    raise ValueError("TensorBoard metrics must be numeric scalars")


def _log_tensorboard_scalars(writer, metrics, *, step):
    for name, value in sorted(metrics.items()):
        writer.add_scalar(name, _normalize_tensorboard_scalar(name, value), step)


@dataclass
class BatchMetricsCSVWriter:
    path: pathlib.Path
    file_handle: object
    writer: csv.DictWriter

    def write_batch(
        self,
        *,
        optimization_step,
        epoch,
        batch_idx,
        total_batches,
        progress_pct,
        elapsed_text,
        eta_text,
        train_batch_loss,
        train_batch_accuracy,
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timestamp = f"{timestamp}.{int((time.time() % 1) * 1000):03d}"
        self.writer.writerow(
            {
                "timestamp": timestamp,
                "optimization_step": int(optimization_step),
                "epoch": int(epoch),
                "batch_idx": int(batch_idx),
                "total_batches": int(total_batches),
                "progress_pct": float(progress_pct),
                "elapsed_text": elapsed_text,
                "eta_text": eta_text,
                "train_batch_loss": float(train_batch_loss),
                "train_batch_accuracy": (
                    ""
                    if train_batch_accuracy is None
                    else float(train_batch_accuracy)
                ),
            }
        )
        self.file_handle.flush()

    def close(self):
        self.file_handle.close()


@dataclass
class ValidationMetricsCSVWriter:
    path: pathlib.Path
    file_handle: object
    writer: csv.DictWriter

    def write_epoch_metrics(
        self,
        *,
        epoch,
        validation_accuracy,
        validation_success_rate,
        validation_average_makespan,
        validation_average_partial_success_rate,
        validation_average_sum_of_costs,
        best_validation_accuracy,
        best_validation_success_rate,
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timestamp = f"{timestamp}.{int((time.time() % 1) * 1000):03d}"
        self.writer.writerow(
            {
                "timestamp": timestamp,
                "epoch": int(epoch),
                "validation_accuracy": (
                    ""
                    if validation_accuracy is None
                    else float(validation_accuracy)
                ),
                "validation_success_rate": (
                    ""
                    if validation_success_rate is None
                    else float(validation_success_rate)
                ),
                "validation_average_makespan": (
                    ""
                    if validation_average_makespan is None
                    else float(validation_average_makespan)
                ),
                "validation_average_partial_success_rate": (
                    ""
                    if validation_average_partial_success_rate is None
                    else float(validation_average_partial_success_rate)
                ),
                "validation_average_sum_of_costs": (
                    ""
                    if validation_average_sum_of_costs is None
                    else float(validation_average_sum_of_costs)
                ),
                "best_validation_accuracy": float(best_validation_accuracy),
                "best_validation_success_rate": float(best_validation_success_rate),
            }
        )
        self.file_handle.flush()

    def close(self):
        self.file_handle.close()


def _default_batch_metrics_csv_path(args):
    logs_dir = pathlib.Path(args.dataset_dir) / "logs"
    run_name = pathlib.Path(args.checkpoints_dir).name
    return logs_dir / f"train_{run_name}_batch_metrics.csv"


def _default_validation_metrics_csv_path(args):
    logs_dir = pathlib.Path(args.dataset_dir) / "logs"
    run_name = pathlib.Path(args.checkpoints_dir).name
    return logs_dir / f"train_{run_name}_validation_metrics.csv"


def _create_batch_metrics_csv_writer(args):
    csv_path = _default_batch_metrics_csv_path(args)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    file_handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        file_handle,
        fieldnames=[
            "timestamp",
            "optimization_step",
            "epoch",
            "batch_idx",
            "total_batches",
            "progress_pct",
            "elapsed_text",
            "eta_text",
            "train_batch_loss",
            "train_batch_accuracy",
        ],
    )
    if not file_exists:
        writer.writeheader()
        file_handle.flush()
    logger.info(f"Batch metrics CSV logging enabled: path={csv_path}")
    return BatchMetricsCSVWriter(
        path=csv_path,
        file_handle=file_handle,
        writer=writer,
    )


def _create_validation_metrics_csv_writer(args):
    csv_path = _default_validation_metrics_csv_path(args)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    file_handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        file_handle,
        fieldnames=[
            "timestamp",
            "epoch",
            "validation_accuracy",
            "validation_success_rate",
            "validation_average_makespan",
            "validation_average_partial_success_rate",
            "validation_average_sum_of_costs",
            "best_validation_accuracy",
            "best_validation_success_rate",
        ],
    )
    if not file_exists:
        writer.writeheader()
        file_handle.flush()
    logger.info(f"Validation metrics CSV logging enabled: path={csv_path}")
    return ValidationMetricsCSVWriter(
        path=csv_path,
        file_handle=file_handle,
        writer=writer,
    )


def _format_batch_progress_extra(*, loss, accuracy):
    parts = [f"loss={float(loss):.6f}"]
    if accuracy is not None:
        parts.append(f"accuracy={float(accuracy):.6f}")
    return ", ".join(parts)


def _build_monitoring_metrics(
    *,
    results,
    epoch,
    optimizer,
    best_validation_success_rate,
    best_validation_accuracy,
    cur_validation_id_max,
    threshold_val_success_rate,
    oe_improve_quality,
):
    monitoring_metrics = dict(results)
    monitoring_metrics["epoch"] = int(epoch)
    monitoring_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    monitoring_metrics["best_validation_success_rate"] = float(
        best_validation_success_rate
    )
    monitoring_metrics["best_validation_accuracy"] = float(best_validation_accuracy)
    monitoring_metrics["cur_validation_id_max"] = int(cur_validation_id_max)
    monitoring_metrics["threshold_val_success_rate"] = float(
        threshold_val_success_rate
    )
    monitoring_metrics["oe_improve_quality"] = bool(oe_improve_quality)
    return monitoring_metrics


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning model.")
    parser = add_expert_dataset_args(parser)
    parser = add_imitation_dataset_args(parser)
    parser = add_hypergraph_generation_args(parser)
    parser = add_additional_data_args(parser)
    parser = add_training_args(parser)
    parser = add_sharded_downstream_args(parser)

    parser.add_argument("--wandb_project", type=str, default="hyper-mapf-train")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    validate_training_args_contract(args)
    print(args)
    _validate_and_warn_batch_limits(args)
    _validate_checkpoint_source_args(args)
    _log_training_stage(
        "args_validated",
        (
            f"dataset_dir={args.dataset_dir}, num_samples={args.num_samples}, "
            f"use_shards={args.use_shards}, sequence_training={args.sequence_training}, "
            f"device_arg={args.device}"
        ),
    )

    assert args.save_termination_state
    assert args.train_on_terminated_agents

    if args.device == -1:
        device = torch.device("cuda")
    elif args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    _validate_amp_args(args, device)
    _log_training_stage("device_resolved", f"device={device}")
    dataloader_kwargs = _build_dataloader_kwargs(args)
    logger.info(
        "DataLoader runtime config: "
        f"num_workers={dataloader_kwargs['num_workers']}, "
        f"pin_memory={dataloader_kwargs['pin_memory']}, "
        "persistent_workers="
        f"{dataloader_kwargs.get('persistent_workers', False)}, "
        "prefetch_factor="
        f"{dataloader_kwargs.get('prefetch_factor', None)}"
    )
    _warn_on_sharded_sequence_loader_profile(args)
    if args.amp:
        logger.info(
            "Automatic mixed precision enabled: "
            f"dtype={args.amp_dtype}, device={device}."
        )

    resume_checkpoint = None
    if args.resume_checkpoint_path is not None:
        print("Loading Resume Checkpoint.............")
        _log_training_stage(
            "resume_checkpoint_loading",
            f"path={args.resume_checkpoint_path}",
        )
        resume_checkpoint = _load_resume_training_checkpoint(
            args.resume_checkpoint_path, map_location=device
        )
        _validate_resume_checkpoint_args(args, resume_checkpoint)
        _log_training_stage(
            "resume_checkpoint_ready",
            f"epoch={resume_checkpoint['epoch']}",
        )

    stack_with_np = not args.use_lists

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(args)

    grid_config = _grid_config_generator(seeds[0], map_id=0)

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)

    from hmagat.modules.agents import (
        get_model,
        load_partial_state_dict,
        validate_partial_load_compatibility,
    )

    model, hypergraph_model, dataset_kwargs = get_model(args, device)
    _log_training_stage(
        "model_initialized",
        (
            f"model={args.imitation_learning_model}, "
            f"hypergraph_model={hypergraph_model}, "
            f"trainable_params={_count_trainable_parameters(model)}"
        ),
    )
    partial_load_summary = None

    load_additional_data, additional_data_idx = any_additional_data(args)
    common_dataset_kwargs = dict(
        edge_attr_opts=args.edge_attr_opts,
        additional_data_idx=additional_data_idx,
        **dataset_kwargs,
    )

    if args.use_shards:
        _log_training_stage("dataset_loading", "loading sharded training datasets")
        train_dataset, validation_dataset, train_id_max, validation_id_max = (
            build_sharded_snapshot_datasets(
                args,
                hypergraph_model=hypergraph_model,
                additional_data_idx=additional_data_idx,
                dataset_kwargs=common_dataset_kwargs,
            )
        )
        current_dataset_state = _build_sharded_dataset_state(
            args,
            train_dataset,
            validation_dataset=validation_dataset,
            train_id_max=train_id_max,
            validation_id_max=validation_id_max,
        )
    else:
        _log_training_stage("dataset_loading", "loading unsharded training datasets")
        dense_dataset = None
        hyper_edge_indices = None
        additional_data = None

        print("Loading Dataset.............")
        dense_dataset = load_dataset(
            [get_imitation_dataset_file_name],
            "processed_dataset",
            args,
        )
        if args.load_positions_separately:
            print("Loading Agent Positions.....")
            agent_pos = load_dataset(
                [get_pos_file_name],
                "positions",
                args,
            )
            dense_dataset = (*dense_dataset, agent_pos)
        if hypergraph_model:
            print("Loading Hypergraphs.........")
            hyper_edge_indices = load_dataset(
                [get_hypergraph_file_name], "hypergraphs", args
            )

        if load_additional_data:
            print("Loading Additional Data.....")
            additional_data = load_dataset(
                [get_additional_data_file_name], "additional_data", args
            )

        unique_map_ids = np.sort(np.unique(dense_dataset[4]))
        num_samples = len(unique_map_ids)

        # Data split
        train_id_max = int(
            num_samples * (1 - args.validation_fraction - args.test_fraction)
        )
        validation_id_max = train_id_max + int(num_samples * args.validation_fraction)

        train_id_max = min(train_id_max, num_samples)
        train_id_max = unique_map_ids[train_id_max]

        validation_id_max = min(validation_id_max, num_samples)
        validation_id_max = unique_map_ids[validation_id_max]

        def _divide_dataset(start, end):
            map_ids = dense_dataset[4]
            if not isinstance(map_ids, torch.Tensor):
                map_ids = torch.from_numpy(np.array(map_ids))
            mask = torch.logical_and(map_ids >= start, map_ids < end)
            hindices, add_data = None, None
            if hyper_edge_indices is not None:
                hindices, hton_indices = hyper_edge_indices
                hindices = list(compress(hindices, mask))
                hton_indices = list(compress(hton_indices, mask))
                hindices = (hindices, hton_indices)
            if additional_data is not None:
                add_data = list(compress(additional_data, mask))
            if isinstance(dense_dataset[0], torch.Tensor):
                ds = tuple(gd[mask] for gd in dense_dataset)
            else:
                ds = tuple(list(compress(gd, mask)) for gd in dense_dataset)
            return (
                ds,
                hindices,
                add_data,
            )

        (train_dataset, train_hindices, train_additional_data) = _divide_dataset(
            0, train_id_max
        )
        (validation_dataset, validation_hindices, validation_additional_data) = (
            _divide_dataset(train_id_max, validation_id_max)
        )
        # test_dataset = _divide_dataset(validation_id_max, torch.inf)

        train_kwargs = dict(additional_data=train_additional_data)
        validation_kwargs = dict(additional_data=validation_additional_data)

        if hypergraph_model:
            train_dataset = MAPFHypergraphDataset(
                train_dataset,
                train_hindices,
                **train_kwargs,
                **common_dataset_kwargs,
            )
            validation_dataset = MAPFHypergraphDataset(
                validation_dataset,
                validation_hindices,
                **validation_kwargs,
                **common_dataset_kwargs,
            )
        else:
            train_dataset = MAPFGraphDataset(
                train_dataset,
                **train_kwargs,
                **common_dataset_kwargs,
            )
            validation_dataset = MAPFGraphDataset(
                validation_dataset,
                **validation_kwargs,
                **common_dataset_kwargs,
            )
        current_dataset_state = _build_unsharded_dataset_state(
            args,
            hypergraph_model=hypergraph_model,
            load_additional_data=load_additional_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_id_max=train_id_max,
            validation_id_max=validation_id_max,
        )

    _log_training_stage(
        "dataset_ready",
        (
            f"train_len={len(train_dataset)}, validation_len={len(validation_dataset)}, "
            f"train_id_max={train_id_max}, validation_id_max={validation_id_max}"
        ),
    )

    if resume_checkpoint is not None:
        _validate_resume_dataset_state(
            current_dataset_state, resume_checkpoint["dataset_state"]
        )

    expert_makespans = None
    if args.oe_improve_quality:
        expert_makespans = check_or_create_expert_makespans(args)

    loss_function = get_loss_function(args)

    tensorboard_purge_step = (
        int(resume_checkpoint["epoch"]) + 1
        if resume_checkpoint is not None
        else None
    )
    tensorboard_writer = _create_tensorboard_writer(
        args, purge_step=tensorboard_purge_step
    )
    if tensorboard_writer is not None:
        _log_training_stage(
            "tensorboard_ready",
            f"dir={args.tensorboard_dir}, purge_step={tensorboard_purge_step}",
        )
    batch_metrics_csv_writer = _create_batch_metrics_csv_writer(args)
    _log_training_stage(
        "batch_metrics_csv_ready",
        f"path={batch_metrics_csv_writer.path}",
    )
    validation_metrics_csv_writer = _create_validation_metrics_csv_writer(args)
    _log_training_stage(
        "validation_metrics_csv_ready",
        f"path={validation_metrics_csv_writer.path}",
    )

    use_wandb = args.wandb_entity is not None
    if use_wandb:
        _log_training_stage(
            "wandb_init",
            f"project={args.wandb_project}, entity={args.wandb_entity}",
        )
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    sequence_validation_mode = "not_applicable"
    sequence_audit_stamp = None
    train_sequence_collator = None
    validation_sequence_collator = None
    train_sequence_step_runtime = None
    validation_sequence_step_runtime = None
    if args.sequence_training:
        sequence_validation_mode, sequence_audit_stamp = (
            _resolve_sequence_dataset_validation_mode(args)
        )
        _log_training_stage(
            "sequence_validation_mode",
            (
                f"mode={sequence_validation_mode}, use_shards={args.use_shards}, "
                f"flag={args.validate_sequence_training_dataset}"
            ),
        )
        if sequence_validation_mode == "audit_stamp":
            _log_training_stage(
                "sequence_audit_stamp_ready",
                f"path={sequence_audit_stamp_path(args)}",
            )
        if sequence_validation_mode == "deep_preflight":
            _log_training_stage(
                "sequence_preflight_start",
                (
                    f"train_snapshots={len(train_dataset)}, "
                    f"validation_snapshots={len(validation_dataset)}"
                ),
            )
            validate_sequence_dataset_assumptions(
                train_dataset,
                progress_label="Training sequence dataset preflight",
            )
            validate_sequence_dataset_assumptions(
                validation_dataset,
                progress_label="Validation sequence dataset preflight",
            )
            _log_training_stage("sequence_preflight_done")
        elif sequence_validation_mode == "skip":
            logger.warning(
                "Skipping sequence training dataset assumption checks because "
                "--no-validate_sequence_training_dataset was set. Run "
                "audit_sequence_dataset --use_shards before full training."
            )
        if args.use_shards:
            train_sequence_dataset = ShardedSequenceDataset.from_snapshot_dataset(
                train_dataset
            )
            validation_sequence_dataset = ShardedSequenceDataset.from_snapshot_dataset(
                validation_dataset
            )
            train_sequence_dataset.set_runtime_context(dataset_role="train_sequence")
            validation_sequence_dataset.set_runtime_context(
                dataset_role="validation_sequence"
            )
        else:
            train_sequence_dataset = MAPFSequenceDataset(train_dataset)
            validation_sequence_dataset = MAPFSequenceDataset(validation_dataset)
        train_sequence_collator = SequenceBatchCollator()
        validation_sequence_collator = SequenceBatchCollator()
        train_sequence_step_runtime = SequenceStepRuntimeTracker()
        validation_sequence_step_runtime = SequenceStepRuntimeTracker()
        logger.info(
            "Training sequence datasets prepared: "
            f"train_episodes={len(train_sequence_dataset)}, "
            f"validation_episodes={len(validation_sequence_dataset)}, "
            f"train_snapshots={len(train_dataset)}, "
            f"validation_snapshots={len(validation_dataset)}"
        )
        if args.use_shards:
            train_dl = TorchDataLoader(
                train_sequence_dataset,
                batch_sampler=ShardedEpisodeBatchSampler(
                    train_sequence_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                ),
                collate_fn=train_sequence_collator,
                **dataloader_kwargs,
            )
        else:
            train_dl = TorchDataLoader(
                train_sequence_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=train_sequence_collator,
                **dataloader_kwargs,
            )
        _log_training_stage(
            "train_dataloader_ready",
            (
                f"mode=sequence, batches_per_epoch~={len(train_dl)}, "
                f"batch_size={args.batch_size}"
            ),
        )
    else:
        logger.info(
            "Training snapshot datasets prepared: "
            f"train_snapshots={len(train_dataset)}, "
            f"validation_snapshots={len(validation_dataset)}"
        )
        train_dl = DataLoader(
            train_dataset, batch_size=args.batch_size, **dataloader_kwargs
        )
        _log_training_stage(
            "train_dataloader_ready",
            (
                f"mode=snapshot, batches_per_epoch~={len(train_dl)}, "
                f"batch_size={args.batch_size}"
            ),
        )
    if args.sequence_training:
        if args.use_shards:
            validation_sequence_dl = TorchDataLoader(
                validation_sequence_dataset,
                batch_sampler=ShardedEpisodeBatchSampler(
                    validation_sequence_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                ),
                collate_fn=validation_sequence_collator,
                **dataloader_kwargs,
            )
        else:
            validation_sequence_dl = TorchDataLoader(
                validation_sequence_dataset,
                batch_size=args.batch_size,
                collate_fn=validation_sequence_collator,
                **dataloader_kwargs,
            )
    else:
        validation_sequence_dl = None
    validation_dl = DataLoader(
        validation_dataset, batch_size=args.batch_size, **dataloader_kwargs
    )
    _log_training_stage(
        "validation_dataloader_ready",
        (
            f"snapshot_batches~={len(validation_dl)}, "
            f"sequence_batches~="
            f"{len(validation_sequence_dl) if validation_sequence_dl is not None else 'n/a'}, "
            f"batch_size={args.batch_size}"
        ),
    )

    if expert_makespans is not None:
        expert_makespans = expert_makespans[:train_id_max]

    best_validation_success_rate = 0.0
    best_validation_accuracy = 0.0
    best_val_file_name = "best_low_val.pt"
    best_val_acc_file_name = "best_acc_val.pt"
    checkpoint_path = pathlib.Path(f"{args.checkpoints_dir}", best_val_file_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cur_validation_id_max = min(train_id_max + args.initial_val_size, validation_id_max)

    oe_graph_dataset = None
    oe_grid_configs = []
    oe_hypergraph_indices = ([], [])
    oe_additional_data = None
    oe_improve_quality = args.oe_improve_quality_threshold == 0.0

    def multiprocess_run_expert(
        queue,
        done_event,
        expert,
        grid_config,
        save_termination_state,
        hypergraph_generator=None,
    ):
        if hypergraph_generator is not None:
            hypergraph_generator.initialized = False

        expert_results = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=save_termination_state,
            additional_data_func=hypergraph_generator,
        )
        queue.put((*expert_results, grid_config))
        if done_event is not None:
            done_event.wait()

    hypergraph_generator = None
    if hypergraph_model:
        hypergraph_generator = HyperedgeIndicesGenerator(
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

    if args.pretrain_weights_path is not None:
        print("Loading Weights.............")
        pretrain_path = pathlib.Path(args.pretrain_weights_path)
        state_dict = extract_model_state_dict_from_checkpoint(
            torch.load(pretrain_path, map_location=device),
            pretrain_path,
            print_prefix="[pretrain] ",
        )
        model.load_state_dict(state_dict)

    if args.load_partial_parameters_path is not None:
        print("Partially Loading Weights.............")
        partial_path = pathlib.Path(args.load_partial_parameters_path)
        state_dict = extract_model_state_dict_from_checkpoint(
            torch.load(partial_path, map_location=device),
            partial_path,
            print_prefix="[partial-load-source] ",
        )
        partial_load_summary = load_partial_state_dict(
            model, state_dict, print_prefix="[partial-load]"
        )
        validate_partial_load_compatibility(
            partial_load_summary, print_prefix="[partial-load]"
        )
    cs_warmup_freeze_state = _apply_cs_warmup_freeze_baseline(
        model,
        args,
        partial_load_summary,
        resume_training_state=(
            resume_checkpoint["training_state"] if resume_checkpoint is not None else None
        ),
    )
    optimizer = _create_optimizer(args, model, cs_warmup_freeze_state)
    grad_scaler = _create_grad_scaler(args)
    lr_scheduler = get_lr_scheduler(
        args, optimizer=optimizer, train_dataloader=train_dl
    )
    start_epoch = 0
    if resume_checkpoint is not None:
        resume_state = _restore_resume_training_state(
            model,
            optimizer,
            lr_scheduler,
            resume_checkpoint,
            grad_scaler=grad_scaler,
        )
        start_epoch = resume_state["start_epoch"]
        best_validation_success_rate = resume_state["best_validation_success_rate"]
        best_validation_accuracy = resume_state["best_validation_accuracy"]
        best_val_file_name = resume_state["best_val_file_name"]
        cur_validation_id_max = resume_state["cur_validation_id_max"]
        args.threshold_val_success_rate = resume_state["threshold_val_success_rate"]
        oe_improve_quality = resume_state["oe_improve_quality"]
        logger.warning(
            "Resuming training from checkpoint: "
            f"completed_epoch={resume_checkpoint['epoch']}, start_epoch={start_epoch}."
        )

    queue = mp.Queue()
    done_event = mp.Event()

    try:
        print("Starting Training....")
        optimization_step = int(getattr(lr_scheduler, "cur_step", 0))
        for epoch in range(start_epoch, args.num_epochs):
            total_loss = 0.0
            accuracies = None
            num_samples = 0
            n_batches = 0
            n_graphs = 0
            n_maps = 0
            train_first_batch_latency_sec = None
            train_epoch_runtime = EpochRuntimeTracker()

            model = model.train()
            train_progress = ProgressLogger(
                f"Training epoch {epoch}",
                _effective_num_batches(len(train_dl), args.max_train_batches),
                every_n=max(
                    1,
                    _effective_num_batches(len(train_dl), args.max_train_batches) // 20,
                ),
                every_seconds=30.0,
            )
            if args.sequence_training:
                _reset_runtime_metrics(
                    train_sequence_dataset,
                    train_sequence_collator,
                    train_sequence_step_runtime,
                )
                if hasattr(train_sequence_dataset, "set_runtime_context"):
                    train_sequence_dataset.set_runtime_context(
                        dataset_role="train_sequence",
                        phase="train",
                        epoch=epoch,
                    )
                    train_sequence_dataset.clear_runtime_batch_context()
            epoch_data_start_time = time.monotonic()
            next_batch_wait_start = epoch_data_start_time
            if args.sequence_training:
                for batch_idx, sequence_batch in enumerate(train_dl):
                    if hasattr(train_sequence_dataset, "set_runtime_context"):
                        train_sequence_dataset.set_runtime_context(
                            dataset_role="train_sequence",
                            phase="train",
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )
                    train_epoch_runtime.record_batch_wait(
                        time.monotonic() - next_batch_wait_start
                    )
                    if train_first_batch_latency_sec is None:
                        train_first_batch_latency_sec = (
                            time.monotonic() - epoch_data_start_time
                        )
                    optimizer.zero_grad()
                    train_sequence_graph_counts = getattr(
                        sequence_batch, "graph_counts", None
                    )
                    train_sequence_first_step_counts = getattr(
                        sequence_batch, "first_step_graph_counts", None
                    )
                    train_sequence_step_idx = 0
                    batch_accuracies = None
                    batch_num_samples = 0

                    def accumulate_step_metrics(out, data):
                        nonlocal accuracies, num_samples, n_graphs, n_maps
                        nonlocal train_sequence_step_idx
                        nonlocal batch_accuracies, batch_num_samples
                        new_acc = loss_function.get_accuracies(out, data, model)
                        if accuracies is None:
                            accuracies = _clone_metric_dict(new_acc)
                        else:
                            for key in accuracies:
                                accuracies[key] += new_acc[key]
                        if batch_accuracies is None:
                            batch_accuracies = _clone_metric_dict(new_acc)
                        else:
                            for key in batch_accuracies:
                                batch_accuracies[key] += new_acc[key]
                        step_num_samples = loss_function.get_num_supervised_samples(
                            data
                        )
                        num_samples += step_num_samples
                        batch_num_samples += step_num_samples
                        if train_sequence_graph_counts is not None:
                            n_graphs += train_sequence_graph_counts[
                                train_sequence_step_idx
                            ]
                        else:
                            n_graphs += len(data.ptr) - 1
                        if train_sequence_first_step_counts is not None:
                            n_maps += train_sequence_first_step_counts[
                                train_sequence_step_idx
                            ]
                        else:
                            n_maps += torch.sum(data.first_step).cpu().item()
                        train_sequence_step_idx += 1

                    with _autocast_context(args, device):
                        loss = compute_sequence_loss(
                            model,
                            sequence_batch,
                            loss_function,
                            device=device,
                            detach_state=args.sequence_detach_state,
                            truncated_bptt_length=args.truncated_bptt_length,
                            on_step=accumulate_step_metrics,
                            runtime_tracker=train_sequence_step_runtime,
                        )
                    total_loss += loss.item()

                    if grad_scaler is not None:
                        backward_start_time = time.monotonic()
                        grad_scaler.scale(loss).backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=grad_scaler,
                            global_step=optimization_step,
                        )
                        train_epoch_runtime.record_backward(
                            time.monotonic() - backward_start_time
                        )
                        optimizer_step_start_time = time.monotonic()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        train_epoch_runtime.record_optimizer_step(
                            time.monotonic() - optimizer_step_start_time
                        )
                    else:
                        backward_start_time = time.monotonic()
                        loss.backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=None,
                            global_step=optimization_step,
                        )
                        train_epoch_runtime.record_backward(
                            time.monotonic() - backward_start_time
                        )
                        optimizer_step_start_time = time.monotonic()
                        optimizer.step()
                        train_epoch_runtime.record_optimizer_step(
                            time.monotonic() - optimizer_step_start_time
                        )
                    if cs_warmup_freeze_state is not None:
                        cs_warmup_freeze_state.restore_decoder_prefix()

                    if tensorboard_writer is not None:
                        batch_accuracy = None
                        if (
                            batch_accuracies is not None
                            and "train_accuracy" in batch_accuracies
                            and batch_num_samples > 0
                        ):
                            batch_accuracy = (
                                float(batch_accuracies["train_accuracy"])
                                / float(batch_num_samples)
                            )
                        _log_tensorboard_scalars(
                            tensorboard_writer,
                            {
                                "train_batch_loss": float(loss.item()),
                                "train_batch_accuracy": (
                                    float(batch_accuracy)
                                    if batch_accuracy is not None
                                    else 0.0
                                ),
                            },
                            step=optimization_step,
                        )
                    else:
                        batch_accuracy = None
                        if (
                            batch_accuracies is not None
                            and "train_accuracy" in batch_accuracies
                            and batch_num_samples > 0
                        ):
                            batch_accuracy = (
                                float(batch_accuracies["train_accuracy"])
                                / float(batch_num_samples)
                            )
                    batch_metrics_csv_writer.write_batch(
                        optimization_step=optimization_step,
                        epoch=epoch,
                        batch_idx=batch_idx + 1,
                        total_batches=train_progress.total,
                        progress_pct=100.0
                        * float(batch_idx + 1)
                        / float(train_progress.total),
                        elapsed_text=train_progress.elapsed_text(),
                        eta_text=train_progress.eta_text(batch_idx + 1),
                        train_batch_loss=loss.item(),
                        train_batch_accuracy=batch_accuracy,
                    )
                    optimization_step += 1
                    n_batches += 1
                    train_epoch_runtime.record_batch()
                    lr_scheduler.step_on_batch()
                    train_progress.update(
                        batch_idx + 1,
                        extra=_format_batch_progress_extra(
                            loss=loss.item(),
                            accuracy=batch_accuracy,
                        ),
                    )
                    next_batch_wait_start = time.monotonic()
                    if _batch_limit_reached(batch_idx + 1, args.max_train_batches):
                        break
            else:
                for batch_idx, data in enumerate(train_dl):
                    train_epoch_runtime.record_batch_wait(
                        time.monotonic() - next_batch_wait_start
                    )
                    transfer_start_time = time.monotonic()
                    data = data.to(device)
                    transfer_elapsed = time.monotonic() - transfer_start_time
                    optimizer.zero_grad()

                    forward_start_time = time.monotonic()
                    with _autocast_context(args, device):
                        out = model(data.x, data)
                        loss = loss_function(out, data, model)
                    forward_elapsed = time.monotonic() - forward_start_time
                    total_loss += loss.item()

                    if grad_scaler is not None:
                        backward_start_time = time.monotonic()
                        grad_scaler.scale(loss).backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=grad_scaler,
                            global_step=optimization_step,
                        )
                        train_epoch_runtime.record_backward(
                            time.monotonic() - backward_start_time
                        )
                        optimizer_step_start_time = time.monotonic()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        train_epoch_runtime.record_optimizer_step(
                            time.monotonic() - optimizer_step_start_time
                        )
                    else:
                        backward_start_time = time.monotonic()
                        loss.backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=None,
                            global_step=optimization_step,
                        )
                        train_epoch_runtime.record_backward(
                            time.monotonic() - backward_start_time
                        )
                        optimizer_step_start_time = time.monotonic()
                        optimizer.step()
                        train_epoch_runtime.record_optimizer_step(
                            time.monotonic() - optimizer_step_start_time
                        )
                    if cs_warmup_freeze_state is not None:
                        cs_warmup_freeze_state.restore_decoder_prefix()

                    new_acc = loss_function.get_accuracies(out, data, model)
                    if accuracies is None:
                        accuracies = new_acc
                    else:
                        for key in accuracies:
                            accuracies[key] += new_acc[key]
                    batch_num_samples = loss_function.get_num_supervised_samples(data)
                    num_samples += batch_num_samples
                    if tensorboard_writer is not None:
                        batch_accuracy = None
                        if "train_accuracy" in new_acc and batch_num_samples > 0:
                            batch_accuracy = float(new_acc["train_accuracy"]) / float(
                                batch_num_samples
                            )
                        _log_tensorboard_scalars(
                            tensorboard_writer,
                            {
                                "train_batch_loss": float(loss.item()),
                                "train_batch_accuracy": (
                                    float(batch_accuracy)
                                    if batch_accuracy is not None
                                    else 0.0
                                ),
                            },
                            step=optimization_step,
                        )
                    else:
                        batch_accuracy = None
                        if "train_accuracy" in new_acc and batch_num_samples > 0:
                            batch_accuracy = float(new_acc["train_accuracy"]) / float(
                                batch_num_samples
                            )
                    batch_metrics_csv_writer.write_batch(
                        optimization_step=optimization_step,
                        epoch=epoch,
                        batch_idx=batch_idx + 1,
                        total_batches=train_progress.total,
                        progress_pct=100.0
                        * float(batch_idx + 1)
                        / float(train_progress.total),
                        elapsed_text=train_progress.elapsed_text(),
                        eta_text=train_progress.eta_text(batch_idx + 1),
                        train_batch_loss=loss.item(),
                        train_batch_accuracy=batch_accuracy,
                    )
                    optimization_step += 1
                    n_batches += 1
                    train_epoch_runtime.record_batch()
                    n_graphs += len(data.ptr) - 1
                    n_maps += torch.sum(data.first_step).cpu().item()
                    lr_scheduler.step_on_batch()
                    train_progress.update(
                        batch_idx + 1,
                        extra=_format_batch_progress_extra(
                            loss=loss.item(),
                            accuracy=batch_accuracy,
                        ),
                    )
                    next_batch_wait_start = time.monotonic()
                    if _batch_limit_reached(batch_idx + 1, args.max_train_batches):
                        break

            if oe_graph_dataset is not None:
                oe_dataset_kwargs = dict(additional_data=oe_additional_data)
                if hypergraph_model:
                    oe_dl = DataLoader(
                        MAPFHypergraphDataset(
                            oe_graph_dataset,
                            oe_hypergraph_indices,
                            **oe_dataset_kwargs,
                            **common_dataset_kwargs,
                        ),
                        batch_size=args.batch_size,
                        **dataloader_kwargs,
                    )
                else:
                    oe_dl = DataLoader(
                        MAPFGraphDataset(
                            oe_graph_dataset,
                            **oe_dataset_kwargs,
                            **common_dataset_kwargs,
                        ),
                        batch_size=args.batch_size,
                        **dataloader_kwargs,
                    )

                for data in oe_dl:
                    data = data.to(device)
                    optimizer.zero_grad()

                    with _autocast_context(args, device):
                        out = model(data.x, data)
                        loss = loss_function(out, data, model)

                    total_loss += loss.item()

                    if grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=grad_scaler,
                            global_step=optimization_step,
                        )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        loss.backward()
                        _apply_gradient_clipping(
                            args,
                            model,
                            optimizer=optimizer,
                            grad_scaler=None,
                            global_step=optimization_step,
                        )
                        optimizer.step()
                    if cs_warmup_freeze_state is not None:
                        cs_warmup_freeze_state.restore_decoder_prefix()

                    new_acc = loss_function.get_accuracies(out, data, model)
                    for key in accuracies:
                        accuracies[key] += new_acc[key]
                    num_samples += loss_function.get_num_supervised_samples(data)
                    optimization_step += 1
                    n_batches += 1
                    n_graphs += len(data.ptr) - 1
                    n_maps += torch.sum(data.first_step).cpu().item()
                    lr_scheduler.step_on_batch()
            lr_scheduler.step_on_epoch()

            for key in accuracies:
                if "first_step_first_agent" in key:
                    accuracies[key] = accuracies[key] / n_maps
                elif "first_agent" in key:
                    accuracies[key] = accuracies[key] / n_graphs
                else:
                    accuracies[key] = accuracies[key] / num_samples

            print(
                f"Epoch {epoch}, Mean Loss: {total_loss / n_batches}, Mean Accuracy: {accuracies['train_accuracy']}"
            )
            if args.sequence_training:
                _log_sequence_runtime_summary(
                    f"Training epoch {epoch}",
                    dataset=train_sequence_dataset,
                    collator=train_sequence_collator,
                    step_runtime=train_sequence_step_runtime,
                    epoch_runtime=train_epoch_runtime,
                    first_batch_latency_sec=train_first_batch_latency_sec,
                )

            results = {"train_loss": total_loss / n_batches} | accuracies
            if args.sequence_training:
                results = results | _sequence_runtime_metrics(
                    prefix="train_sequence",
                    dataset=train_sequence_dataset,
                    collator=train_sequence_collator,
                    step_runtime=train_sequence_step_runtime,
                    epoch_runtime=train_epoch_runtime,
                    first_batch_latency_sec=train_first_batch_latency_sec,
                )

            # Save resumable progress immediately after the train phase of the
            # epoch so that a long validation stage cannot erase completed work.
            _save_epoch_progress_checkpoints(
                save_intermediate_checkpoint=args.save_intmd_checkpoints,
                checkpoints_dir=args.checkpoints_dir,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                args=args,
                best_validation_success_rate=best_validation_success_rate,
                best_validation_accuracy=best_validation_accuracy,
                best_val_file_name=best_val_file_name,
                cur_validation_id_max=cur_validation_id_max,
                threshold_val_success_rate=args.threshold_val_success_rate,
                oe_improve_quality=oe_improve_quality,
                dataset_state=current_dataset_state,
                cs_warmup_freeze_state=cs_warmup_freeze_state,
                grad_scaler=grad_scaler,
            )
            if (not args.skip_validation) and (
                (epoch + 1) % args.validation_every_epochs == 0
            ):
                model = model.eval()

                num_completed = 0
                all_makespan = []
                all_partial_success_rate = []
                all_sum_of_costs = []

                print("-------------------")
                print("Starting Validation", flush=True)

                if not args.skip_validation_accuracy:
                    val_accuracies = None
                    val_samples = 0
                    n_graphs = 0
                    n_maps = 0
                    validation_first_batch_latency_sec = None
                    validation_epoch_runtime = EpochRuntimeTracker()

                    with torch.no_grad():
                        if args.sequence_training:
                            _reset_runtime_metrics(
                                validation_sequence_dataset,
                                validation_sequence_collator,
                                validation_sequence_step_runtime,
                            )
                            if hasattr(validation_sequence_dataset, "set_runtime_context"):
                                validation_sequence_dataset.set_runtime_context(
                                    dataset_role="validation_sequence",
                                    phase="validation_accuracy",
                                    epoch=epoch,
                                )
                                validation_sequence_dataset.clear_runtime_batch_context()
                            validation_data_start_time = time.monotonic()
                            validation_next_batch_wait_start = (
                                validation_data_start_time
                            )
                            validation_progress = ProgressLogger(
                                f"Validation accuracy epoch {epoch}",
                                _effective_num_batches(
                                    len(validation_sequence_dl),
                                    args.max_validation_batches,
                                ),
                                every_n=max(
                                    1,
                                    _effective_num_batches(
                                        len(validation_sequence_dl),
                                        args.max_validation_batches,
                                    )
                                    // 20,
                                ),
                                every_seconds=30.0,
                            )
                            for batch_idx, sequence_batch in enumerate(
                                validation_sequence_dl
                            ):
                                if hasattr(
                                    validation_sequence_dataset,
                                    "set_runtime_context",
                                ):
                                    validation_sequence_dataset.set_runtime_context(
                                        dataset_role="validation_sequence",
                                        phase="validation_accuracy",
                                        epoch=epoch,
                                        batch_idx=batch_idx,
                                    )
                                validation_epoch_runtime.record_batch_wait(
                                    time.monotonic()
                                    - validation_next_batch_wait_start
                                )
                                if validation_first_batch_latency_sec is None:
                                    validation_first_batch_latency_sec = (
                                        time.monotonic() - validation_data_start_time
                                    )
                                validation_sequence_graph_counts = getattr(
                                    sequence_batch, "graph_counts", None
                                )
                                validation_sequence_first_step_counts = getattr(
                                    sequence_batch, "first_step_graph_counts", None
                                )
                                validation_sequence_step_idx = 0

                                def accumulate_validation_metrics(out, data):
                                    nonlocal val_accuracies, val_samples, n_graphs, n_maps
                                    nonlocal validation_sequence_step_idx
                                    new_acc = loss_function.get_accuracies(
                                        out, data, model, "validation"
                                    )

                                    if val_accuracies is None:
                                        val_accuracies = new_acc
                                    else:
                                        for key in val_accuracies:
                                            val_accuracies[key] += new_acc[key]
                                    val_samples += (
                                        loss_function.get_num_supervised_samples(data)
                                    )
                                    if validation_sequence_graph_counts is not None:
                                        n_graphs += validation_sequence_graph_counts[
                                            validation_sequence_step_idx
                                        ]
                                    else:
                                        n_graphs += len(data.ptr) - 1
                                    if (
                                        validation_sequence_first_step_counts
                                        is not None
                                    ):
                                        n_maps += validation_sequence_first_step_counts[
                                            validation_sequence_step_idx
                                        ]
                                    else:
                                        n_maps += torch.sum(data.first_step).cpu().item()
                                    validation_sequence_step_idx += 1

                                with _autocast_context(args, device):
                                    compute_sequence_loss(
                                        model,
                                        sequence_batch,
                                        loss_function,
                                        device=device,
                                        detach_state=True,
                                        on_step=accumulate_validation_metrics,
                                        runtime_tracker=validation_sequence_step_runtime,
                                    )
                                validation_epoch_runtime.record_batch()
                                validation_progress.update(batch_idx + 1)
                                validation_next_batch_wait_start = time.monotonic()
                                if _batch_limit_reached(
                                    batch_idx + 1,
                                    args.max_validation_batches,
                                ):
                                    break
                            _log_sequence_runtime_summary(
                                f"Validation accuracy epoch {epoch}",
                                dataset=validation_sequence_dataset,
                                collator=validation_sequence_collator,
                                step_runtime=validation_sequence_step_runtime,
                                epoch_runtime=validation_epoch_runtime,
                                first_batch_latency_sec=validation_first_batch_latency_sec,
                            )
                            results = results | _sequence_runtime_metrics(
                                prefix="validation_sequence",
                                dataset=validation_sequence_dataset,
                                collator=validation_sequence_collator,
                                step_runtime=validation_sequence_step_runtime,
                                epoch_runtime=validation_epoch_runtime,
                                first_batch_latency_sec=validation_first_batch_latency_sec,
                            )
                        else:
                            validation_progress = ProgressLogger(
                                f"Validation accuracy epoch {epoch}",
                                _effective_num_batches(
                                    len(validation_dl),
                                    args.max_validation_batches,
                                ),
                                every_n=max(
                                    1,
                                    _effective_num_batches(
                                        len(validation_dl),
                                        args.max_validation_batches,
                                    )
                                    // 20,
                                ),
                                every_seconds=30.0,
                            )
                            for batch_idx, data in enumerate(validation_dl):
                                data = data.to(device)
                                with _autocast_context(args, device):
                                    out = model(data.x, data)
                                new_acc = loss_function.get_accuracies(
                                    out, data, model, "validation"
                                )

                                if val_accuracies is None:
                                    val_accuracies = new_acc
                                else:
                                    for key in val_accuracies:
                                        val_accuracies[key] += new_acc[key]
                                val_samples += loss_function.get_num_supervised_samples(
                                    data
                                )
                                n_graphs += len(data.ptr) - 1
                                n_maps += torch.sum(data.first_step).cpu().item()
                                validation_progress.update(batch_idx + 1)
                                if _batch_limit_reached(
                                    batch_idx + 1,
                                    args.max_validation_batches,
                                ):
                                    break
                    for key in val_accuracies:
                        if "first_step_first_agent" in key:
                            val_accuracies[key] = val_accuracies[key] / n_maps
                        elif "first_agent" in key:
                            val_accuracies[key] = val_accuracies[key] / n_graphs
                        else:
                            val_accuracies[key] = val_accuracies[key] / val_samples

                    val_accuracy = val_accuracies["validation_accuracy"]
                    results = results | val_accuracies
                    if val_accuracy > best_validation_accuracy:
                        best_validation_accuracy = val_accuracy
                        checkpoint_path = pathlib.Path(
                            args.checkpoints_dir, best_val_acc_file_name
                        )
                        _save_training_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            epoch=epoch,
                            args=args,
                            best_validation_success_rate=best_validation_success_rate,
                            best_validation_accuracy=best_validation_accuracy,
                            best_val_file_name=best_val_file_name,
                            cur_validation_id_max=cur_validation_id_max,
                            threshold_val_success_rate=args.threshold_val_success_rate,
                            oe_improve_quality=oe_improve_quality,
                            dataset_state=current_dataset_state,
                            cs_warmup_freeze_state=cs_warmup_freeze_state,
                            grad_scaler=grad_scaler,
                        )

                rollout_graph_ids, total_validation_graphs = _validation_rollout_graph_ids(
                    use_shards=args.use_shards,
                    train_id_max=train_id_max,
                    cur_validation_id_max=cur_validation_id_max,
                    validation_id_max=validation_id_max,
                    validation_dataset=validation_dataset,
                )
                for rollout_idx, graph_id in enumerate(rollout_graph_ids):
                    rollout_grid_config = _grid_config_generator(
                        seeds[graph_id], map_id=graph_id
                    )
                    print(
                        (
                            f"Validation rollout start: graph_idx={rollout_idx}/"
                            f"{total_validation_graphs}, graph_id={graph_id}, "
                            f"map_seed={getattr(rollout_grid_config, 'seed', 'na')}, "
                            f"sampling_seed={getattr(rollout_grid_config, 'sampling_seed', getattr(rollout_grid_config, 'seed', 'na'))}, "
                            f"max_episode_steps={getattr(rollout_grid_config, 'max_episode_steps', 'na')}"
                        ),
                        flush=True,
                    )
                    success, env, observations = run_model_on_grid(
                        model=model,
                        device=device,
                        grid_config=rollout_grid_config,
                        args=args,
                        dataset_kwargs=dataset_kwargs,
                        hypergraph_model=hypergraph_model,
                        use_target_vec=args.use_target_vec,
                        aux_func=aux_func,
                        debug_label=(
                            f"Validation rollout graph_idx={rollout_idx}/"
                            f"{total_validation_graphs}, graph_id={graph_id}"
                        ),
                        debug_slow_step_sec=5.0,
                    )

                    makespan = aux_func.makespan
                    costs = aux_func.costs

                    partial_success_rate = np.mean(env.was_on_goal)
                    sum_of_costs = np.sum(costs)

                    all_makespan.append(makespan)
                    all_partial_success_rate.append(partial_success_rate)
                    all_sum_of_costs.append(sum_of_costs)

                    if success:
                        num_completed += 1
                    print(
                        f"Validation Graph {rollout_idx}/{total_validation_graphs}, "
                        f"Current Success Rate: {num_completed / (rollout_idx + 1)}",
                        flush=True,
                    )
                success_rate = num_completed / len(rollout_graph_ids)
                results = results | {
                    "validation_success_rate": success_rate,
                    "validation_average_makespan": np.mean(all_makespan),
                    "validation_average_partial_success_rate": np.mean(
                        all_partial_success_rate
                    ),
                    "validation_average_sum_of_costs": np.mean(all_sum_of_costs),
                }

                if success_rate > best_validation_success_rate:
                    best_validation_success_rate = success_rate
                    if success_rate >= args.threshold_val_success_rate:
                        print(
                            "Success rate passed threshold -- Increasing Validation Size",
                            flush=True,
                        )
                        args.threshold_val_success_rate = 1.1
                        cur_validation_id_max = validation_id_max
                        best_val_file_name = "best.pt"
                        best_validation_success_rate = 0.0
                    checkpoint_path = pathlib.Path(
                        f"{args.checkpoints_dir}", best_val_file_name
                    )
                    _save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        args=args,
                        best_validation_success_rate=best_validation_success_rate,
                        best_validation_accuracy=best_validation_accuracy,
                        best_val_file_name=best_val_file_name,
                        cur_validation_id_max=cur_validation_id_max,
                        threshold_val_success_rate=args.threshold_val_success_rate,
                        oe_improve_quality=oe_improve_quality,
                        dataset_state=current_dataset_state,
                        cs_warmup_freeze_state=cs_warmup_freeze_state,
                        grad_scaler=grad_scaler,
                    )

                print("Finshed Validation")
                print("------------------")

                if args.run_online_expert and (epoch + 1 >= args.run_oe_after):
                    print("---------------------")
                    print("Running Online Expert")

                    rng = np.random.default_rng(args.dataset_seed + epoch + 1)
                    if args.recursive_oe:
                        oe_ids = rng.integers(
                            train_id_max + len(oe_grid_configs), size=args.num_run_oe
                        )
                    else:
                        oe_ids = rng.integers(train_id_max, size=args.num_run_oe)

                    oe_dataset = []
                    oe_hindices = []
                    oe_add_data = []
                    oe_makespans = []
                    num_oe_success = 0
                    num_oe_improve = 0

                    if oe_improve_quality:
                        aux_func_train(
                            env=None,
                            observations=None,
                            actions=None,
                            oe_period=args.oe_improve_quality_period,
                        )

                    for i, graph_id in enumerate(oe_ids):
                        print(f"Running model on {i}/{args.num_run_oe} ", end="")
                        if graph_id >= train_id_max:
                            grid_config = oe_grid_configs[graph_id - train_id_max]
                        else:
                            if args.use_shards:
                                if not hasattr(train_dataset, "episode_original_sample_ids"):
                                    logger.warning(
                                        "Sharded online-expert rollout requires "
                                        "train_dataset to expose episode_original_sample_ids."
                                    )
                                    raise ValueError(
                                        "Sharded online-expert rollout requires episode_original_sample_ids"
                                    )
                                graph_id = train_dataset.episode_original_sample_ids[
                                    graph_id
                                ]
                            grid_config = _grid_config_generator(
                                seeds[graph_id], map_id=graph_id
                            )
                        success, env, observations = run_model_on_grid(
                            model=model,
                            device=device,
                            grid_config=grid_config,
                            args=args,
                            dataset_kwargs=dataset_kwargs,
                            hypergraph_model=hypergraph_model,
                            use_target_vec=args.use_target_vec,
                            max_episodes=args.max_episode_steps,
                            aux_func=aux_func_train if oe_improve_quality else None,
                        )
                        grid_configs_to_run = []
                        if success:
                            num_oe_success += 1
                            if (
                                oe_improve_quality
                                and num_oe_improve < args.oe_improve_quality_max_num
                            ):
                                model_makespan = aux_func_train.makespan
                                expert_makespan = expert_makespans[graph_id]
                                if (
                                    model_makespan
                                    >= args.oe_improve_quality_buffer
                                    * expert_makespan
                                ):
                                    # Will run oe to improve solution quality
                                    num_oe_improve += 1
                                    cur_makespan = model_makespan
                                    for gc_to_run in aux_func_train.grid_configs:
                                        cur_makespan -= args.oe_improve_quality_period
                                        # Setting max_episode_steps such that only better solutions are found
                                        gc_to_run.max_episode_steps = int(
                                            cur_makespan
                                            / args.oe_improve_quality_buffer
                                        )
                                        grid_configs_to_run.append(gc_to_run)
                        else:
                            grid_configs_to_run.append(
                                generate_grid_config_from_env(env)
                            )

                        if len(grid_configs_to_run) > 0:
                            print(f"-- Running OE ", end="")
                            for grid_config in grid_configs_to_run:
                                expert = expert_algorithm(inference_config)

                                all_actions, all_observations, all_terminated = (
                                    None,
                                    None,
                                    None,
                                )
                                expert_results = None
                                hindices = []

                                if args.run_expert_in_separate_fork:
                                    done_event.clear()
                                    p = mp.Process(
                                        target=multiprocess_run_expert,
                                        args=(
                                            queue,
                                            done_event,
                                            expert,
                                            grid_config,
                                            args.save_termination_state,
                                            hypergraph_generator,
                                        ),
                                    )
                                    p.start()

                                    if args.max_runtime_oe is not None:
                                        start_time = time.time()

                                    while p.is_alive():
                                        if args.max_runtime_oe is not None:
                                            if (
                                                time.time() - start_time
                                                > args.max_runtime_oe
                                            ):
                                                print(f"-- Timeout")
                                                p.terminate()
                                                break
                                        try:
                                            expert_results = queue.get(timeout=3)
                                            done_event.set()
                                            p.join(timeout=0.5)
                                            if p.exitcode is None:
                                                p.terminate()
                                            break
                                        except queue_module.Empty:
                                            p.join(timeout=0.5)
                                            if p.exitcode is not None:
                                                break
                                else:
                                    multiprocess_run_expert(
                                        queue,
                                        None,
                                        expert,
                                        grid_config,
                                        args.save_termination_state,
                                        hypergraph_generator,
                                    )
                                    expert_results = queue.get()

                                if expert_results is not None:
                                    (all_actions, all_observations, all_terminated) = (
                                        expert_results[:3]
                                    )
                                    grid_config = expert_results[-1]
                                    if hypergraph_model:
                                        hindices = expert_results[-2]
                                    if all(all_terminated[-1]):
                                        print(f"-- Success")
                                        oe_dataset.append(
                                            (
                                                all_observations,
                                                all_actions,
                                                all_terminated,
                                            )
                                        )
                                        oe_hindices.extend(hindices)
                                        grid_config.max_episode_steps = (
                                            args.max_episode_steps
                                        )
                                        oe_grid_configs.append(grid_config)
                                        oe_makespans.append(len(all_actions))
                                        if load_additional_data:
                                            add_data = generate_additional_data(
                                                grid_config=grid_config,
                                                all_actions=all_actions,
                                                num_previous_actions=args.add_data_num_previous_actions,
                                                cost_to_go=args.add_data_cost_to_go,
                                                normalized_cost_to_go=args.normalize_cost_to_go,
                                                greedy_action=args.add_data_greedy_action,
                                                clamp_value=args.clamp_cost_to_go,
                                                clamped_values_doubled=args.clamped_values_doubled,
                                            )
                                            oe_add_data.extend(add_data)
                                    else:
                                        print(f"-- Fail")
                                        break
                                else:
                                    print(f"-- Error")
                                    break
                        else:
                            print(f"-- Success")
                    oe_success_rate = num_oe_success / len(oe_ids)
                    if (
                        not oe_improve_quality
                        and args.oe_improve_quality
                        and oe_success_rate >= args.oe_improve_quality_threshold
                    ):
                        print("Enabling Online Expert for Solution Quality Improvement")
                        oe_improve_quality = True
                        if args.oe_improve_quality_expert is not None:
                            args.expert_algorithm = args.oe_improve_quality_expert
                            print(f"Switching expert to {args.expert_algorithm}.")
                            expert_algorithm, inference_config = (
                                get_expert_algorithm_and_config(args)
                            )
                    while queue.qsize() > 0:
                        # Popping remaining elements, although no elements should remain
                        expert_results = queue.get()
                        hindices = []
                        (
                            all_actions,
                            all_observations,
                            all_terminated,
                        ) = expert_results[:3]
                        grid_config = expert_results[-1]
                        if hypergraph_model:
                            hindices = expert_results[-2]

                        if all(all_terminated[-1]):
                            oe_dataset.append(
                                (all_observations, all_actions, all_terminated)
                            )
                            oe_hindices.extend(hindices)
                            grid_config.max_episode_steps = args.max_episode_steps
                            oe_grid_configs.append(grid_config)
                            oe_makespans.append(len(all_actions))
                            if load_additional_data:
                                add_data = generate_additional_data(
                                    grid_config=grid_config,
                                    all_actions=all_actions,
                                    num_previous_actions=args.add_data_num_previous_actions,
                                    cost_to_go=args.add_data_cost_to_go,
                                    normalized_cost_to_go=args.normalize_cost_to_go,
                                    greedy_action=args.add_data_greedy_action,
                                    clamp_value=args.clamp_cost_to_go,
                                    clamped_values_doubled=args.clamped_values_doubled,
                                )
                                oe_add_data.extend(add_data)

                    if len(oe_dataset) > 0:
                        print(f"Adding {len(oe_dataset)} OE grids to the dataset")
                        oe_hindices = tuple(list(h) for h in zip(*oe_hindices))
                        for i in range(len(oe_hypergraph_indices)):
                            oe_hypergraph_indices[i].extend(oe_hindices[i])
                        new_oe_graph_dataset = generate_graph_dataset(
                            dataset=oe_dataset,
                            comm_radius=args.comm_radius,
                            obs_radius=args.obs_radius,
                            num_samples=None,
                            save_termination_state=True,
                            use_edge_attr=dataset_kwargs["use_edge_attr"],
                            print_prefix=None,
                            num_neighbour_cutoff=args.num_neighbour_cutoff,
                            neighbour_cutoff_method=args.neighbour_cutoff_method,
                            distance_metric=args.distance_metric,
                            random_edge_probs=args.random_edge_probs,
                            stack_with_np=stack_with_np,
                        )
                        if oe_graph_dataset is None:
                            oe_graph_dataset = new_oe_graph_dataset
                        else:
                            if isinstance(oe_graph_dataset[0], torch.Tensor):
                                oe_graph_dataset = tuple(
                                    torch.concat(
                                        [oe_graph_dataset[i], new_oe_graph_dataset[i]],
                                        dim=0,
                                    )
                                    for i in range(len(oe_graph_dataset))
                                )
                            else:
                                oe_graph_dataset = tuple(
                                    oe_graph_dataset[i] + new_oe_graph_dataset[i]
                                    for i in range(len(oe_graph_dataset))
                                )
                        if len(oe_add_data) > 0:
                            if oe_additional_data is None:
                                oe_additional_data = oe_add_data
                            else:
                                oe_additional_data.extend(oe_add_data)

                        if expert_makespans is not None:
                            oe_makespans = np.array(oe_makespans)
                            expert_makespans = np.concatenate(
                                [expert_makespans, oe_makespans], axis=0
                            )

                    print("Finished Online Expert")
                    print("----------------------")

            _save_epoch_progress_checkpoints(
                save_intermediate_checkpoint=args.save_intmd_checkpoints,
                checkpoints_dir=args.checkpoints_dir,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                args=args,
                best_validation_success_rate=best_validation_success_rate,
                best_validation_accuracy=best_validation_accuracy,
                best_val_file_name=best_val_file_name,
                cur_validation_id_max=cur_validation_id_max,
                threshold_val_success_rate=args.threshold_val_success_rate,
                oe_improve_quality=oe_improve_quality,
                dataset_state=current_dataset_state,
                cs_warmup_freeze_state=cs_warmup_freeze_state,
                grad_scaler=grad_scaler,
            )

            monitoring_metrics = _build_monitoring_metrics(
                results=results,
                epoch=epoch,
                optimizer=optimizer,
                best_validation_success_rate=best_validation_success_rate,
                best_validation_accuracy=best_validation_accuracy,
                cur_validation_id_max=cur_validation_id_max,
                threshold_val_success_rate=args.threshold_val_success_rate,
                oe_improve_quality=oe_improve_quality,
            )
            if tensorboard_writer is not None:
                _log_tensorboard_scalars(
                    tensorboard_writer,
                    monitoring_metrics,
                    step=epoch,
                )
            if any(key.startswith("validation_") for key in monitoring_metrics):
                validation_metrics_csv_writer.write_epoch_metrics(
                    epoch=epoch,
                    validation_accuracy=monitoring_metrics.get("validation_accuracy"),
                    validation_success_rate=monitoring_metrics.get(
                        "validation_success_rate"
                    ),
                    validation_average_makespan=monitoring_metrics.get(
                        "validation_average_makespan"
                    ),
                    validation_average_partial_success_rate=monitoring_metrics.get(
                        "validation_average_partial_success_rate"
                    ),
                    validation_average_sum_of_costs=monitoring_metrics.get(
                        "validation_average_sum_of_costs"
                    ),
                    best_validation_accuracy=best_validation_accuracy,
                    best_validation_success_rate=best_validation_success_rate,
                )
            if use_wandb:
                wandb.log(monitoring_metrics)
    finally:
        if batch_metrics_csv_writer is not None:
            batch_metrics_csv_writer.close()
        if validation_metrics_csv_writer is not None:
            validation_metrics_csv_writer.close()
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("fork")  # TODO: Maybe add this as an cmd line option
    main()
