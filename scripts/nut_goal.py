"""Goal-from-cloud task: robosuite NutAssembly with both nuts on the table; the episode's goal says
which one to put on its peg, and only the cloud is told (paper 2, docs/query-planner-plan.md).

Square showed the cloud map adds ~17/360 over a robot-only BEV planner (outputs/stress/): the
robot's own camera already sees everything the task needs. Here it cannot: both nuts are always
visible (robosuite places the square nut at y in [0.11, 0.225], the round one at [-0.225, -0.11]),
and the goal is semantic, standing in for an instruction a cloud VLM parses. A robot-only policy
can only guess (~50%), so the cloud's contribution is measurable by construction.

    goal 0   SquareNut -> peg1 (square peg, y = +0.10)
    goal 1   RoundNut  -> peg2 (round peg,  y = -0.10)

A goal switch flips the goal once, at `switch_tick`, unless the goal nut has already been lifted
(then it is cancelled: a re-target after the grasp would need a put-down the oracle does not
have). The robot learns of a switch only from a cloud memory requested after it, so the switch
is the clean delay probe: ticks spent on the old goal vs link delay.

GoalWorld is the one place the goal lives, shared by the demo collector and the eval harness. It
wraps the env (or a MovingCube around it) and adds to every observation

    goal            int, the current goal (the harness hands it to the cloud half only)
    goal_nut_pos    the current goal nut's position (3)
    goal_nut_quat   and quaternion (4, xyzw), so code written for one nut (oracle frames,
                    trackers, the failure tracer) reads these keys and follows the goal
"""
from __future__ import annotations

import numpy as np

ENV_NAME = 'NutAssembly'
NUTS = ('SquareNut', 'RoundNut')                   # goal i = robosuite nut id i, onto peg i+1
PEGS_XY = ((0.23, 0.10), (0.23, -0.10))            # peg1, peg2 (world x, y; measured); fixed
LIFTED_M = 0.005                                   # nut above its start height: held


class GoalWorld:
    def __init__(self, env, goal: int = 0, switch_tick: int | None = None) -> None:
        self.env = env
        self.goal, self.switch_tick = goal, switch_tick
        self.switched = False
        self.t = 0
        self._z0 = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, goal: int | None = None, switch_tick: int | None = None):
        if goal is not None:
            self.goal = goal
        self.switch_tick = switch_tick
        self.switched = False
        self.t = 0
        obs = self.env.reset()
        self._z0 = {n: float(obs[f'{n}_pos'][2]) for n in NUTS}
        return self._add(obs)

    def lifted(self, obs, goal: int) -> bool:
        n = NUTS[goal]
        return float(obs[f'{n}_pos'][2]) > self._z0[n] + LIFTED_M

    def step(self, action):
        obs, *rest = self.env.step(action)
        self.t += 1
        if self.switch_tick is not None and self.t == self.switch_tick:
            if not self.lifted(obs, self.goal):
                self.goal, self.switched = 1 - self.goal, True
        return (self._add(obs), *rest)

    def success(self) -> bool:
        """The goal nut is on its peg (and the gripper away); the other nut does not matter."""
        self.env._check_success()                  # updates objects_on_pegs
        return bool(self.env.objects_on_pegs[self.goal])

    def _add(self, obs: dict) -> dict:
        n = NUTS[self.goal]
        obs = dict(obs)
        obs.update(goal=self.goal, goal_nut_pos=np.asarray(obs[f'{n}_pos']),
                   goal_nut_quat=np.asarray(obs[f'{n}_quat']))
        return obs
