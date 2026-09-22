"""One thread owns the simulator: it builds it, steps it, and closes it.

robosuite renders camera observations inside `env.step()` through an EGL context, and an EGL
context is bound to the thread that made it current. Any render from another thread produces
garbage: measured on NutAssemblySquare, stepping from two alternating worker threads gave frames
that differed from the single-thread reference by ~71 grey levels on average, with the agentview
and wrist images identical to each other. The plant used to build the env on the main thread and
step it from a ROS timer on a 2-thread MultiThreadedExecutor, so each step landed on whichever
worker was free: most frames were right, some were the wrong camera or half-drawn, and the policy
consumed them. Lift tolerated it; Square (a precise insertion) scored 0/10 in the loop against
85% co-located.

So nothing else may touch the env. `SimThread` runs `build()` on its own thread, then calls
`step()` every `period_s` of wall time, then `close()` on that same thread when stopped. Timing
keeps the plant's existing soft real-time contract: no catch-up; if a step overruns, the next one
starts from now rather than firing a burst to make up the lost ticks (`overruns` counts them).

Pure Python (threading + a clock), so the fast suite tests it with a fake env.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


class SimThread:
    """Build, step at a fixed wall-clock period, and close — all on one dedicated thread."""

    def __init__(self, build: Callable[[], None], step: Callable[[], None],
                 close: Callable[[], None], period_s: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 name: str = 'evh_sim') -> None:
        self._build, self._step, self._close = build, step, close
        self.period_s = period_s
        self._clock, self._sleep = clock, sleep
        self._stop = threading.Event()
        self.ready = threading.Event()        # set once build() has returned (or raised)
        self.error: BaseException | None = None
        self.steps = 0
        self.overruns = 0
        self.thread_id: int | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> SimThread:
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout_s)

    def _run(self) -> None:
        self.thread_id = threading.get_ident()
        try:
            self._build()
        except BaseException as exc:          # surfaced to the owner through .error
            self.error = exc
            self.ready.set()
            return
        self.ready.set()
        try:
            next_t = self._clock()
            while not self._stop.is_set():
                self._step()
                self.steps += 1
                next_t += self.period_s
                now = self._clock()
                if next_t < now:
                    self.overruns += 1
                    next_t = now                 # no catch-up: the same contract as a ROS timer
                else:
                    self._sleep(next_t - now)
        except BaseException as exc:
            self.error = exc
        finally:
            self._close()
