import asyncio
import errno
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterator

import pytest
from unreal_agent.primitives.process import (
    ProcessParentPipes,
    signal_process_invocation,
    terminate_process,
)
from unreal_agent.primitives.process_signal import (
    DARWIN_PROCESS_FLAG_EXITING,
    DARWIN_PROCESS_STATE_ZOMBIE,
    KinfoProc,
    kinfo_proc,
    normalize_process_group_signal_error,
    process_group_has_live_members,
)

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Darwin process inspection")

SIGKILL = 9
SIGTERM = 15


@pytest.fixture
def group_members() -> Iterator[Callable[[int], subprocess.Popen[bytes]]]:
    started: list[subprocess.Popen[bytes]] = []

    def start(process_group: int) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(["/bin/sleep", "30"], process_group=process_group)
        started.append(process)
        return process

    yield start
    for process in started:
        if process.returncode is None:
            process.kill()
            process.wait()


async def wait_for_zombie(pid: int) -> None:
    deadline = time.monotonic() + 5
    delay = 0.001
    while True:
        info = kinfo_proc(pid)
        if info.p_stat == DARWIN_PROCESS_STATE_ZOMBIE:
            return
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"process {pid} state = {info.p_stat}, want zombie"
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.05)


async def _sigterm(process: subprocess.Popen[bytes]) -> BaseException | None:
    return signal_process_invocation(process, SIGTERM)


async def _sigkill(process: subprocess.Popen[bytes]) -> BaseException | None:
    return signal_process_invocation(process, SIGKILL)


async def _terminate(process: subprocess.Popen[bytes]) -> BaseException | None:
    return await terminate_process(process, ProcessParentPipes(), 0)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(_sigterm, id="SIGTERM"),
        pytest.param(_sigkill, id="SIGKILL"),
        pytest.param(_terminate, id="terminate"),
    ],
)
async def test_process_termination_with_unreaped_zombie(
    group_members: Callable[[int], subprocess.Popen[bytes]],
    run: Callable[[subprocess.Popen[bytes]], Awaitable[BaseException | None]],
) -> None:
    process = group_members(0)
    process.kill()
    await wait_for_zombie(process.pid)
    assert await run(process) is None
    assert process.wait() == -SIGKILL


async def test_signal_process_invocation_with_zombie_leader_and_live_group_member(
    group_members: Callable[[int], subprocess.Popen[bytes]],
) -> None:
    leader = group_members(0)
    member = group_members(leader.pid)
    leader.kill()
    await wait_for_zombie(leader.pid)
    permission = PermissionError(errno.EPERM, "operation not permitted")
    assert normalize_process_group_signal_error(leader.pid, permission) is permission
    assert signal_process_invocation(leader, SIGKILL) is None
    await wait_for_zombie(member.pid)
    assert signal_process_invocation(leader, SIGKILL) is None
    assert member.wait() == -SIGKILL


async def test_normalize_process_group_signal_error(
    group_members: Callable[[int], subprocess.Popen[bytes]],
) -> None:
    process = group_members(0)

    def errors() -> list[OSError]:
        return [OSError(code, "") for code in (errno.EPERM, errno.ESRCH, errno.EINVAL)]

    assert normalize_process_group_signal_error(process.pid, None) is None
    for want in errors():
        assert normalize_process_group_signal_error(process.pid, want) is want
    process.kill()
    await wait_for_zombie(process.pid)
    for want in errors()[1:]:
        assert normalize_process_group_signal_error(process.pid, want) is want
    assert normalize_process_group_signal_error(process.pid, errors()[0]) is None
    assert process.wait() == -SIGKILL
    assert normalize_process_group_signal_error(process.pid, errors()[0]) is None


RUNNING, STOPPED, EXEC_FLAG = 2, 4, 0x4000
LIVE = KinfoProc(p_stat=RUNNING, p_flag=EXEC_FLAG)
STOPPED_MEMBER = KinfoProc(p_stat=STOPPED)
ZOMBIE = KinfoProc(p_stat=DARWIN_PROCESS_STATE_ZOMBIE)
EXITING = KinfoProc(p_stat=RUNNING, p_flag=EXEC_FLAG | DARWIN_PROCESS_FLAG_EXITING)


@pytest.mark.parametrize(
    ("members", "want"),
    [
        pytest.param([], False, id="empty"),
        pytest.param([ZOMBIE], False, id="zombie"),
        pytest.param([EXITING], False, id="exiting before zombie"),
        pytest.param([LIVE], True, id="live"),
        pytest.param([STOPPED_MEMBER], True, id="stopped"),
        pytest.param([ZOMBIE, LIVE], True, id="zombie leader with live member"),
    ],
)
def test_process_group_has_live_members(members: list[KinfoProc], want: bool) -> None:
    assert process_group_has_live_members(members) is want
