import time

import torch
from loguru import logger


class SequenceStepRuntimeTracker:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.total_timesteps = 0
        self.state_restore_sec = 0.0
        self.data_to_device_sec = 0.0
        self.forward_sec = 0.0
        self.state_stash_sec = 0.0
        self.on_step_sec = 0.0
        self.loss_compute_sec = 0.0

    def record_timestep(self):
        self.total_timesteps += 1

    def record_state_restore(self, elapsed_sec):
        self.state_restore_sec += float(elapsed_sec)

    def record_data_to_device(self, elapsed_sec):
        self.data_to_device_sec += float(elapsed_sec)

    def record_forward(self, elapsed_sec):
        self.forward_sec += float(elapsed_sec)

    def record_state_stash(self, elapsed_sec):
        self.state_stash_sec += float(elapsed_sec)

    def record_on_step(self, elapsed_sec):
        self.on_step_sec += float(elapsed_sec)

    def record_loss_compute(self, elapsed_sec):
        self.loss_compute_sec += float(elapsed_sec)

    def metrics_snapshot(self):
        return {
            "timesteps": int(self.total_timesteps),
            "state_restore_sec": float(self.state_restore_sec),
            "data_to_device_sec": float(self.data_to_device_sec),
            "forward_sec": float(self.forward_sec),
            "state_stash_sec": float(self.state_stash_sec),
            "on_step_sec": float(self.on_step_sec),
            "loss_compute_sec": float(self.loss_compute_sec),
        }


def _detach_episode_states(episode_states):
    for episode_idx, state in enumerate(episode_states):
        if state is not None:
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
    missing_episode_indices = []
    for episode_idx in active_episode_indices:
        state = episode_states[episode_idx]
        if state is None:
            missing_episode_indices.append(int(episode_idx))
            continue
        states.append(state)
    if not states:
        return
    if missing_episode_indices:
        logger.warning(
            "Sequence state restore fallback: no stored coordination state for "
            f"active episodes {missing_episode_indices}; resetting model "
            "coordination state."
        )
        model.reset_coordination_state()
        return
    model._coordination_state = torch.cat(states, dim=0)


def _stash_active_episode_state(model, episode_states, ptr_list, active_episode_indices):
    state = getattr(model, "_coordination_state", None)
    if state is None:
        logger.warning(
            "Sequence state stash skipped: model has no current coordination state."
        )
        return

    for batch_idx, episode_idx in enumerate(active_episode_indices):
        start = ptr_list[batch_idx]
        end = ptr_list[batch_idx + 1]
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
    runtime_tracker=None,
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
        episode_states = [None] * int(len(sequence_batch.sequence_lengths))
        active_episode_indices = getattr(sequence_batch, "active_episode_indices", None)
        active_episode_index_lists = getattr(
            sequence_batch, "active_episode_index_lists", None
        )
        timestep_ptr_lists = getattr(sequence_batch, "timestep_ptr_lists", None)
        for timestep, data in enumerate(sequence_batch.timesteps):
            if device is not None:
                transfer_start_time = time.monotonic()
                data = data.to(device)
                if runtime_tracker is not None:
                    runtime_tracker.record_data_to_device(
                        time.monotonic() - transfer_start_time
                    )
            active_indices = None
            if active_episode_indices is not None:
                if active_episode_index_lists is not None:
                    active_indices = active_episode_index_lists[timestep]
                else:
                    active_indices = tuple(active_episode_indices[timestep].tolist())
                state_restore_start_time = time.monotonic()
                _restore_active_episode_state(model, episode_states, active_indices)
                if runtime_tracker is not None:
                    runtime_tracker.record_state_restore(
                        time.monotonic() - state_restore_start_time
                    )
            forward_start_time = time.monotonic()
            out = model(data.x, data)
            if runtime_tracker is not None:
                runtime_tracker.record_forward(time.monotonic() - forward_start_time)
            if active_indices is not None:
                if timestep_ptr_lists is not None:
                    ptr_list = timestep_ptr_lists[timestep]
                else:
                    ptr_list = tuple(data.ptr.detach().cpu().tolist())
                state_stash_start_time = time.monotonic()
                _stash_active_episode_state(
                    model, episode_states, ptr_list, active_indices
                )
                if runtime_tracker is not None:
                    runtime_tracker.record_state_stash(
                        time.monotonic() - state_stash_start_time
                    )
            if on_step is not None:
                on_step_start_time = time.monotonic()
                on_step(out, data)
                if runtime_tracker is not None:
                    runtime_tracker.record_on_step(time.monotonic() - on_step_start_time)
            loss_start_time = time.monotonic()
            loss = loss_function(out, data, model)
            if runtime_tracker is not None:
                runtime_tracker.record_loss_compute(
                    time.monotonic() - loss_start_time
                )
                runtime_tracker.record_timestep()
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
