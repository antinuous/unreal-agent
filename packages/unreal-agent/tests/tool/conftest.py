from collections.abc import AsyncIterator, Callable

import pytest
from unreal_agent.operation import LocalOperationManager


@pytest.fixture
async def new_manager() -> AsyncIterator[Callable[[], LocalOperationManager]]:
    """Create managers that are closed and drained when the test ends."""
    managers: list[LocalOperationManager] = []

    def create() -> LocalOperationManager:
        manager = LocalOperationManager()
        managers.append(manager)
        return manager

    yield create
    for manager in managers:
        await manager.aclose()
