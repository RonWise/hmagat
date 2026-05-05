import torch
from loguru import logger


def _detach_episode_states(episode_states):
    for episode_idx, state in list(episode_states.items()):
        episode_states[episode_idx] = state.detach()


def _restore_active_episode_state(model, episode_states, active_episode_indices):
    if not hasattr(model, "_coordination_state"):
        logger.warning(
            "Sequence state restore skipped: model has no _coordination_state "
            "attribute."
        )
        return
    if len(episode_states) == 0:
        return

    states = []
    for episode_idx in active_episode_indices.tolist():
        if episode_idx not in episode_states:
            logger.warning(
                "Sequence state restore fallback: no stored coordination state for "
                f"active episode {episode_idx}; resetting model coordination state."
            )
            model.reset_coordination_state()
            return
        states.append(episode_states[episode_idx])
    model._coordination_state = torch.cat(states, dim=0)


def _stash_active_episode_state(model, episode_states, data, active_episode_indices):
    state = getattr(model, "_coordination_state", None)
    if state is None:
        logger.warning(
            "Sequence state stash skipped: model has no current coordination state."
        )
        return

    ptr = data.ptr.detach().cpu().tolist()
    for batch_idx, episode_idx in enumerate(active_episode_indices.tolist()):
        start = ptr[batch_idx]
        end = ptr[batch_idx + 1]
        episode_states[episode_idx] = state[start:end]


def compute_sequence_loss(
    model,
    sequence_batch,
    loss_function,
    device=None,
    reset_state=True,
    detach_state=False,
    truncated_bptt_length=None,
    on_step=None,
):
    if truncated_bptt_length is not None and truncated_bptt_length <= 0:
        raise ValueError("truncated_bptt_length must be positive or None.")

    if reset_state and hasattr(model, "reset_coordination_state"):
        model.reset_coordination_state()
    elif reset_state:
        logger.warning(
            "Sequence training requested reset_state=True, but model has no "
            "reset_coordination_state method."
        )

    previous_simulation = getattr(model, "simulation", None)
    previous_detach_state = getattr(model, "detach_coordination_state", None)
    if hasattr(model, "in_simulation"):
        model.in_simulation(True)
    else:
        logger.warning(
            "Sequence training could not enable simulation mode: model has no "
            "in_simulation method."
        )
    if hasattr(model, "set_coordination_state_detach"):
        model.set_coordination_state_detach(detach_state)
    elif not detach_state:
        logger.warning(
            "Sequence training requested detach_state=False, but model has no "
            "set_coordination_state_detach method; temporal BPTT may be disabled."
        )

    try:
        total_loss = None
        num_timesteps = 0
        episode_states = {}
        active_episode_indices = getattr(sequence_batch, "active_episode_indices", None)
        for timestep, data in enumerate(sequence_batch.timesteps):
            if device is not None:
                data = data.to(device)
            if active_episode_indices is not None:
                active_indices = active_episode_indices[timestep]
                _restore_active_episode_state(model, episode_states, active_indices)
            out = model(data.x, data)
            if active_episode_indices is not None:
                _stash_active_episode_state(model, episode_states, data, active_indices)
            if on_step is not None:
                on_step(out, data)
            loss = loss_function(out, data, model)
            total_loss = loss if total_loss is None else total_loss + loss
            num_timesteps += 1
            if (
                truncated_bptt_length is not None
                and num_timesteps % truncated_bptt_length == 0
            ):
                _detach_episode_states(episode_states)
                current_state = getattr(model, "_coordination_state", None)
                if current_state is not None:
                    model._coordination_state = current_state.detach()

        if total_loss is None:
            raise ValueError("Cannot compute sequence loss for an empty sequence batch.")

        return total_loss / num_timesteps
    finally:
        if (
            previous_simulation is not None
            and previous_simulation is not True
            and hasattr(model, "in_simulation")
        ):
            model.in_simulation(previous_simulation)
        if (
            previous_detach_state is not None
            and hasattr(model, "set_coordination_state_detach")
        ):
            model.set_coordination_state_detach(previous_detach_state)
