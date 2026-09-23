"""Registers FinanceHeadroom-v0 with Gymnasium so it's reachable via gym.make()."""
from gymnasium.envs.registration import register

from .environment import FinanceHeadroomEnv

register(
    id="FinanceHeadroom-v0",
    entry_point="finance_headroom.env.environment:FinanceHeadroomEnv",
)

__all__ = ["FinanceHeadroomEnv"]
