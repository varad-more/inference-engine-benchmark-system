"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
from typing import Generator

import pytest


# Use asyncio as the default event loop for all async tests
@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.DefaultEventLoopPolicy:
    return asyncio.DefaultEventLoopPolicy()
