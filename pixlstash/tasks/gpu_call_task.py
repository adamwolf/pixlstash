"""Interactive task that runs one callable on the GPU worker thread.

torch's Apple Metal backend crashes or hangs when two threads use it at once
(see ``docs/apple-metal-thread-safety.md``), so every piece of Metal work has to
run on one thread. The task runner's single GPU worker is that thread. Code that
is not itself a task - the anomaly-region route running the tagger - reaches it
through :meth:`~pixlstash.task_runner.TaskRunner.run_on_gpu_worker`, which wraps
the call in this task and waits for it. Code that must not wait - the idle unload
sweep, run from a progress poll - submits the task itself.

It runs on the GPU queue at ``URGENT`` priority, so it skips ahead of queued
background work but still waits for the task already running to finish.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pixlstash.tasks.base_task import BaseTask, QueueType, TaskPriority


class GpuCallTask(BaseTask):
    """Call ``fn(*args, **kwargs)`` on the GPU worker and keep what it returns.

    The call never leaves the GPU worker thread. CPU spillover is not a move to
    another thread but a flag a task reads to choose the CPU device, and the
    runner only offers it from its VRAM gate, which ``TaskRunner._run`` skips
    for GPU-queue tasks. This task also declines spillover and estimates no
    VRAM, so the gate would let it through at once even if it were consulted.

    A GPU out-of-memory error gets the retries every GPU-queue task gets
    (``BaseTask.VRAM_OOM_ATTEMPTS``): the runner flushes the allocator cache,
    pauses ``TaskRunner.VRAM_OOM_RETRY_PAUSE_S`` and calls *fn* again. *fn*
    must therefore be safe to call more than once, as an inference pass, a
    model load or a cache flush is. A call that is not, such as an image plugin
    run that reports progress as it goes, passes ``retry_vram_oom=False`` and
    gets one attempt. After the last attempt the waiter gets the error itself,
    kept in :attr:`exception`. A call that ``run_on_gpu_worker`` makes inline,
    from a GPU task, has no retry of its own: an OOM it raises is the enclosing
    task's to handle.

    Whatever *fn* raises stays on this task for the waiter, ``SystemExit`` and
    ``KeyboardInterrupt`` included, and the worker keeps running: *fn* may be a
    third-party plugin, and ``BaseTask.run`` records an exception that is not
    an ``Exception`` as the task's failure (``TaskInterruptedError``) instead of
    letting it end the worker thread.

    The runner does not flush the device cache after a call
    (``FLUSH_DEVICE_CACHE_AFTER_RUN`` is ``False``): flushing after every call
    would cost a ``gc.collect()`` each time and return the buffers the next
    call would reuse.

    Args:
        fn: The callable to run.
        args: Positional arguments for *fn*.
        kwargs: Keyword arguments for *fn*.
        retry_vram_oom: Whether a GPU out-of-memory error calls *fn* again.

    Attributes:
        exception: What the latest call of *fn* raised, or ``None`` when it
            returned. ``BaseTask.error`` keeps only the message.
    """

    FLUSH_DEVICE_CACHE_AFTER_RUN = False

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        retry_vram_oom: bool = True,
    ):
        super().__init__(
            task_type="GpuCallTask",
            params={"fn": getattr(fn, "__qualname__", repr(fn))},
        )
        self._fn = fn
        self._args = args
        self._kwargs = kwargs or {}
        self.exception: Optional[BaseException] = None
        if not retry_vram_oom:
            # Read by ``BaseTask.run`` and the runner's OOM notice; one attempt
            # means the first OOM is the last.
            self.VRAM_OOM_ATTEMPTS = 1

    @property
    def priority(self) -> TaskPriority:
        return TaskPriority.URGENT

    @property
    def queue_type(self) -> QueueType:
        return QueueType.GPU

    def allow_cpu_spillover(self) -> bool:
        """A call has no CPU mode to switch to; *fn* picks its own device."""
        return False

    def estimated_vram_mb(self) -> int:
        """Zero, so the VRAM gate never holds the call back."""
        return 0

    def _run_task(self) -> Any:
        """Call *fn* and return its result.

        Returns:
            Whatever *fn* returned.

        Raises:
            BaseException: Whatever *fn* raised, kept in :attr:`exception`.
        """
        self.exception = None
        try:
            return self._fn(*self._args, **self._kwargs)
        except BaseException as exc:
            # Kept whole for the waiter. ``BaseTask.run`` records the failure,
            # ``TaskRunner._run`` logs it.
            self.exception = exc
            raise
