from typing import Optional
import math
import warnings
import torch.optim as optim
from loguru import logger


class BaseLRScheduler:
    def __init__(
        self,
        scheduler: optim.lr_scheduler._LRScheduler,
        step_on_batch: bool = False,
        step_on_epoch: bool = False,
        max_steps: Optional[int] = None,
    ):
        self.scheduler = scheduler
        self._step_on_batch = step_on_batch
        self._step_on_epoch = step_on_epoch
        self.cur_step = 0
        self.max_steps = max_steps

    def step_on_batch(self, *args, **kwargs):
        if self._step_on_batch:
            if self.max_steps is not None:
                if self.cur_step >= self.max_steps:
                    return
                self.cur_step += 1
            self.scheduler.step(*args, **kwargs)

    def step_on_epoch(self, *args, **kwargs):
        if self._step_on_epoch:
            self.scheduler.step(*args, **kwargs)

    def state_dict(self):
        return {
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step_on_batch": self._step_on_batch,
            "step_on_epoch": self._step_on_epoch,
            "cur_step": self.cur_step,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state_dict, *, allow_horizon_extension=False):
        required_keys = {
            "scheduler_state_dict",
            "step_on_batch",
            "step_on_epoch",
            "cur_step",
            "max_steps",
        }
        missing_keys = sorted(required_keys - set(state_dict))
        if missing_keys:
            logger.warning(
                "LR scheduler checkpoint is missing required keys: "
                f"{missing_keys}."
            )
            raise ValueError("LR scheduler checkpoint is missing required keys")

        if bool(state_dict["step_on_batch"]) != self._step_on_batch:
            logger.warning(
                "LR scheduler checkpoint step_on_batch mismatch: "
                f"{state_dict['step_on_batch']} != {self._step_on_batch}."
            )
            raise ValueError("LR scheduler checkpoint step_on_batch mismatch")
        if bool(state_dict["step_on_epoch"]) != self._step_on_epoch:
            logger.warning(
                "LR scheduler checkpoint step_on_epoch mismatch: "
                f"{state_dict['step_on_epoch']} != {self._step_on_epoch}."
            )
            raise ValueError("LR scheduler checkpoint step_on_epoch mismatch")
        if (not allow_horizon_extension) and state_dict["max_steps"] != self.max_steps:
            logger.warning(
                "LR scheduler checkpoint max_steps mismatch: "
                f"{state_dict['max_steps']} != {self.max_steps}."
            )
            raise ValueError("LR scheduler checkpoint max_steps mismatch")

        if not allow_horizon_extension:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            self.cur_step = int(state_dict["cur_step"])
            return

        self._load_state_dict_with_horizon_extension(state_dict)

    def _load_state_dict_with_horizon_extension(self, state_dict):
        saved_cur_step = int(state_dict["cur_step"])
        saved_scheduler_state_dict = state_dict["scheduler_state_dict"]
        if "last_epoch" not in saved_scheduler_state_dict:
            logger.warning(
                "LR scheduler checkpoint is missing scheduler_state_dict.last_epoch."
            )
            raise ValueError(
                "LR scheduler checkpoint is missing scheduler_state_dict.last_epoch"
            )

        if self._step_on_batch:
            if self.max_steps is None:
                logger.warning(
                    "LR scheduler horizon extension requires max_steps for "
                    "batch-stepped schedulers."
                )
                raise ValueError(
                    "LR scheduler horizon extension requires max_steps for batch-stepped schedulers"
                )
            if saved_cur_step > self.max_steps:
                logger.warning(
                    "LR scheduler checkpoint progress exceeds current max_steps: "
                    f"{saved_cur_step} > {self.max_steps}."
                )
                raise ValueError(
                    "LR scheduler checkpoint progress exceeds current max_steps"
                )
            progress = saved_cur_step
        else:
            progress = int(saved_scheduler_state_dict["last_epoch"])
            if progress < 0:
                logger.warning(
                    "LR scheduler checkpoint has a negative last_epoch progress."
                )
                raise ValueError(
                    "LR scheduler checkpoint has a negative last_epoch progress"
                )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`\.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The epoch parameter in `scheduler.step\(\)` was not necessary.*",
                category=UserWarning,
            )
            self.scheduler.step(progress)

        if hasattr(self.scheduler, "_step_count"):
            self.scheduler._step_count = progress + 1
        self.cur_step = saved_cur_step


def get_estimated_total_number_of_steps(args, train_dataloader, fraction_of_oe=0.9):
    train_batches_per_epoch = len(train_dataloader)
    max_train_batches = getattr(args, "max_train_batches", None)
    if max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, max_train_batches)

    number_of_steps = train_batches_per_epoch * args.num_epochs
    if args.skip_validation:
        return number_of_steps
    if args.validation_every_epochs <= 0:
        raise ValueError("validation_every_epochs must be positive")

    max_num_oe_runs = (
        args.num_epochs // args.validation_every_epochs
        - (args.run_oe_after // args.validation_every_epochs)
        - 1
    )
    max_num_oe_runs = max(max_num_oe_runs, 0)
    avg_num_oe_runs = max_num_oe_runs * 0.5
    num_oe_batches_per_run = (
        fraction_of_oe * args.num_run_oe + args.batch_size - 1
    ) // args.batch_size
    num_oe_steps = (
        avg_num_oe_runs * num_oe_batches_per_run * args.validation_every_epochs
    )
    num_oe_steps = math.ceil(num_oe_steps)

    return number_of_steps + num_oe_steps


def get_lr_scheduler(args, optimizer, train_dataloader):
    if args.lr_scheduler == "cosine-annealing":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr_end
        )
        return BaseLRScheduler(lr_scheduler, step_on_epoch=True)
    elif args.lr_scheduler == "one-cycle":
        number_of_steps = get_estimated_total_number_of_steps(args, train_dataloader)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr_start,
            total_steps=number_of_steps,
        )
        return BaseLRScheduler(
            lr_scheduler, step_on_batch=True, max_steps=number_of_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}.")
