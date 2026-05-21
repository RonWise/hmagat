import pathlib

import torch
from loguru import logger


TRAINING_CHECKPOINT_TYPE = "hmagat-training"
TRAINING_CHECKPOINT_VERSION = 2


def _is_raw_model_state_dict(candidate):
    if not isinstance(candidate, dict) or not candidate:
        return False
    return all(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items())


def extract_model_state_dict_from_checkpoint(
    checkpoint,
    checkpoint_path,
    *,
    print_prefix="",
):
    checkpoint_path = str(checkpoint_path)
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get("checkpoint_type") == TRAINING_CHECKPOINT_TYPE
    ):
        state_dict = checkpoint.get("model_state_dict")
        if not _is_raw_model_state_dict(state_dict):
            logger.warning(
                f"{print_prefix}Training checkpoint {checkpoint_path} has an invalid "
                "model_state_dict payload."
            )
            raise ValueError("Training checkpoint has an invalid model_state_dict")
        logger.warning(
            f"{print_prefix}Training checkpoint payload detected in {checkpoint_path}; "
            "loading model_state_dict."
        )
        return state_dict

    if _is_raw_model_state_dict(checkpoint):
        logger.warning(
            f"{print_prefix}Legacy raw model state_dict checkpoint detected in "
            f"{checkpoint_path}; loading weights directly."
        )
        return checkpoint

    logger.warning(
        f"{print_prefix}Unsupported checkpoint format in {checkpoint_path}."
    )
    raise ValueError("Unsupported checkpoint format")


def load_model_state_dict_from_checkpoint_path(
    checkpoint_path,
    *,
    map_location,
    print_prefix="",
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return extract_model_state_dict_from_checkpoint(
        checkpoint,
        checkpoint_path,
        print_prefix=print_prefix,
    )


def resolve_evaluation_checkpoint_path(
    checkpoints_dir,
    *,
    model_epoch_num=None,
    map_location,
):
    checkpoints_dir = pathlib.Path(checkpoints_dir)
    if model_epoch_num is not None:
        return checkpoints_dir / f"epoch_{model_epoch_num}.pt"

    last_checkpoint_path = checkpoints_dir / "last.pt"
    if last_checkpoint_path.exists():
        last_checkpoint = torch.load(last_checkpoint_path, map_location=map_location)
        if (
            isinstance(last_checkpoint, dict)
            and last_checkpoint.get("checkpoint_type") == TRAINING_CHECKPOINT_TYPE
        ):
            training_state = last_checkpoint.get("training_state")
            if not isinstance(training_state, dict):
                logger.warning(
                    "Training checkpoint last.pt is missing training_state; cannot "
                    "resolve evaluation checkpoint reliably."
                )
                raise ValueError("Training checkpoint last.pt is missing training_state")
            best_val_file_name = training_state.get("best_val_file_name")
            if not isinstance(best_val_file_name, str) or not best_val_file_name:
                logger.warning(
                    "Training checkpoint last.pt is missing best_val_file_name; "
                    "cannot resolve evaluation checkpoint reliably."
                )
                raise ValueError(
                    "Training checkpoint last.pt is missing best_val_file_name"
                )
            checkpoint_path = checkpoints_dir / best_val_file_name
            if not checkpoint_path.exists():
                logger.warning(
                    "Training checkpoint last.pt points to a missing best-validation "
                    f"checkpoint: {checkpoint_path}."
                )
                raise ValueError(
                    "Training checkpoint last.pt points to a missing best-validation checkpoint"
                )
            logger.warning(
                "Resolved evaluation checkpoint from last.pt training state: "
                f"{checkpoint_path}."
            )
            return checkpoint_path

        logger.warning(
            "Found last.pt in checkpoints_dir, but it is not a supported training "
            "checkpoint payload; falling back to legacy best.pt/best_low_val.pt "
            "resolution."
        )

    best_checkpoint_path = checkpoints_dir / "best.pt"
    if best_checkpoint_path.exists():
        return best_checkpoint_path
    best_low_val_path = checkpoints_dir / "best_low_val.pt"
    if best_low_val_path.exists():
        return best_low_val_path

    logger.warning(
        "Could not resolve evaluation checkpoint: neither last.pt best-validation "
        "reference nor best.pt/best_low_val.pt exists."
    )
    raise ValueError("Could not resolve evaluation checkpoint")


def load_partial_checkpoint_into_model(
    model,
    checkpoint_path,
    *,
    map_location,
    print_prefix="",
    **validate_kwargs,
):
    from hmagat.modules.agents import (
        load_partial_state_dict,
        validate_partial_load_compatibility,
    )

    state_dict = load_model_state_dict_from_checkpoint_path(
        checkpoint_path,
        map_location=map_location,
        print_prefix=f"{print_prefix}[partial-load-source] ",
    )
    summary = load_partial_state_dict(model, state_dict, print_prefix=print_prefix)
    validate_partial_load_compatibility(
        summary,
        print_prefix=print_prefix,
        **validate_kwargs,
    )
    return summary
