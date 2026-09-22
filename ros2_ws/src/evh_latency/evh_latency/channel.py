"""The relay's channel model: per-message delay and loss, with no ROS in it.

Split out of `latency_node.py` so the statistics the paper's RQ2 turns on (tail shape, loss
burstiness) are unit-testable directly, over millions of samples, instead of through a spun-up
relay.

Delay: `latency_ms` plus jitter drawn from `jitter_model` (see latency_node's docstring for why a
heavy-tailed lognormal exists next to the light-tailed gaussian/uniform). Never negative.

Loss, two models at the same AVERAGE rate `drop_prob`:

  iid       each message is dropped independently with probability drop_prob.

  gilbert   Gilbert–Elliott, the standard two-state bursty-loss channel: the link alternates
            between GOOD (nothing lost) and BAD (everything lost). BAD periods last `burst_ms` on
            average, GOOD periods `burst_ms * (1 - drop_prob) / drop_prob`, both exponential, so
            the long-run fraction of time spent BAD is drop_prob. Equal average loss with
            different burst lengths is exactly the comparison RQ2 needs: iid spreads the same
            loss thinly, a long burst concentrates it into outages.

The GOOD/BAD state is SHARED by every relay without any communication between them. A real WiFi
outage hits the camera frames, proprio and actions together; independent per-relay states would
turn one outage into three uncorrelated partial ones. So the state is a deterministic function of
(seed, time): time is cut into `EPOCH_S`-long epochs, each epoch's state trajectory is generated
from its own RNG seeded by (seed, epoch index), and any relay asking about time t on the same
clock gets the same answer. Bursts are defined in milliseconds rather than messages for the same
reason: the topics run at different rates, and only time is common to all of them. (Each epoch
starts GOOD, a negligible artefact once an hour.)
"""
from __future__ import annotations

import bisect
import random

EPOCH_S = 3600.0


class GilbertElliott:
    """Shared, deterministic GOOD/BAD state as a function of time (seconds)."""

    def __init__(self, loss: float, burst_ms: float, seed: int = 0) -> None:
        if not 0.0 < loss < 1.0:
            raise ValueError(f'gilbert loss must be in (0, 1), got {loss}')
        if burst_ms <= 0.0:
            raise ValueError(f'burst_ms must be positive, got {burst_ms}')
        self.loss = loss
        self.bad_mean_s = burst_ms / 1e3
        self.good_mean_s = self.bad_mean_s * (1.0 - loss) / loss
        self.seed = seed
        self._epoch: int | None = None
        self._edges: list[float] = []     # state flips, seconds into the epoch; starts GOOD
        self._rng: random.Random | None = None

    def bad(self, t_s: float) -> bool:
        epoch = int(t_s // EPOCH_S)
        offset = t_s - epoch * EPOCH_S
        if epoch != self._epoch:
            self._epoch, self._edges = epoch, []
            self._rng = random.Random(f'{self.seed}:{epoch}')
        while not self._edges or self._edges[-1] <= offset:
            last = self._edges[-1] if self._edges else 0.0
            mean = self.good_mean_s if len(self._edges) % 2 == 0 else self.bad_mean_s
            self._edges.append(last + self._rng.expovariate(1.0 / mean))
        # an odd number of flips before `offset` means we are in a BAD period
        return bisect.bisect_right(self._edges, offset) % 2 == 1


class Channel:
    """Per-message delay and drop decisions for one relay."""

    def __init__(self, latency_ms: float = 0.0, jitter_ms: float = 0.0,
                 jitter_model: str = 'gaussian', drop_prob: float = 0.0,
                 loss_model: str = 'iid', burst_ms: float = 100.0, seed: int = 0) -> None:
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.jitter_model = jitter_model
        self.drop_prob = drop_prob
        self.loss_model = loss_model
        if loss_model not in ('iid', 'gilbert'):
            raise ValueError(f"loss_model must be iid|gilbert, got {loss_model!r}")
        self._rng = random.Random(seed)
        self._ge = (GilbertElliott(drop_prob, burst_ms, seed)
                    if loss_model == 'gilbert' and drop_prob > 0.0 else None)

    def dropped(self, t_s: float) -> bool:
        """Whether a message entering the relay at time `t_s` is lost."""
        if self.drop_prob <= 0.0:
            return False
        if self._ge is not None:
            return self._ge.bad(t_s)
        return self._rng.random() < self.drop_prob

    def delay_ms(self) -> float:
        """This message's one-way delay. Never negative."""
        d = self.latency_ms
        if self.jitter_ms > 0.0:
            if self.jitter_model == 'uniform':
                d += self._rng.uniform(-self.jitter_ms, self.jitter_ms)
            elif self.jitter_model == 'lognormal':
                # heavy-tailed and one-sided: a link is occasionally much slower than nominal and
                # never faster. Scaled so the MEAN excess is jitter_ms, keeping the knob
                # comparable to the light-tailed models while the tail behaves nothing like them.
                sigma = 1.0
                d += self.jitter_ms * self._rng.lognormvariate(-0.5 * sigma * sigma, sigma)
            else:
                d += self._rng.gauss(0.0, self.jitter_ms)
        return max(0.0, d)
