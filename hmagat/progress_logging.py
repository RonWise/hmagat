import time

from loguru import logger


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class ProgressLogger:
    def __init__(
        self,
        label,
        total,
        *,
        every_n=1,
        every_seconds=30.0,
        requested_total=None,
    ):
        self.label = label
        self.total = max(1, int(total))
        self.every_n = max(1, int(every_n))
        self.every_seconds = every_seconds
        self.requested_total = requested_total
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time

        requested = (
            f", requested={requested_total}" if requested_total is not None else ""
        )
        logger.info(f"{self.label} started: total={self.total}{requested}")

    def elapsed_text(self):
        return _format_duration(time.monotonic() - self.start_time)

    def eta_text(self, current):
        current = int(current)
        elapsed = time.monotonic() - self.start_time
        average = elapsed / max(1, current)
        remaining = max(0, self.total - current)
        return _format_duration(remaining * average)

    def update(self, current, *, extra=None, force=False):
        current = int(current)
        now = time.monotonic()
        should_log = (
            force
            or current == 1
            or current >= self.total
            or current % self.every_n == 0
            or now - self.last_log_time >= self.every_seconds
        )
        if not should_log:
            return

        elapsed = now - self.start_time
        average = elapsed / max(1, current)
        remaining = max(0, self.total - current)
        eta = remaining * average
        percent = min(100.0, 100.0 * current / self.total)

        suffix = f", {extra}" if extra else ""
        logger.info(
            f"{self.label}: {current}/{self.total} ({percent:.2f}%), "
            f"elapsed={_format_duration(elapsed)}, eta={_format_duration(eta)}"
            f"{suffix}"
        )
        self.last_log_time = now
