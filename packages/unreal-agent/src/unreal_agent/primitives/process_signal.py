"""Darwin process-group inspection through `sysctl(kern.proc.pgrp)`."""

import ctypes
import ctypes.util
import errno
import os
import sys
from dataclasses import dataclass
from typing import Final

from unreal_agent.primitives._errors import join, wrap

# Values from Darwin's sys/proc.h and sys/sysctl.h, not exported by Python.
DARWIN_PROCESS_STATE_ZOMBIE: Final = 5  # SZOMB
DARWIN_PROCESS_FLAG_EXITING: Final = 0x00002000  # P_WEXIT
_CTL_KERN: Final = 1
_KERN_PROC: Final = 14
_KERN_PROC_PID: Final = 1
_KERN_PROC_PGRP: Final = 2
# struct kinfo_proc layout on 64-bit Darwin (arm64 and x86_64).
_KINFO_PROC_SIZE: Final = 648
_P_FLAG_OFFSET: Final = 32
_P_STAT_OFFSET: Final = 36
_P_PID_OFFSET: Final = 40


@dataclass(slots=True, frozen=True)
class KinfoProc:
    p_pid: int = 0
    p_stat: int = 0
    p_flag: int = 0


def process_group_has_live_members(members: list[KinfoProc]) -> bool:
    return any(
        member.p_stat != DARWIN_PROCESS_STATE_ZOMBIE
        and member.p_flag & DARWIN_PROCESS_FLAG_EXITING == 0
        for member in members
    )


def normalize_process_group_signal_error(
    process_group_id: int, signal_error: BaseException | None
) -> BaseException | None:
    """Forgive Darwin's EPERM for a group that holds only exiting processes or zombies.

    P_WEXIT is visible before the process reaches SZOMB. Every member is checked:
    a dead leader does not imply its descendants are dead.
    """
    if sys.platform != "darwin":
        return signal_error
    if not (isinstance(signal_error, OSError) and signal_error.errno == errno.EPERM):
        return signal_error
    try:
        members = kinfo_proc_slice(_KERN_PROC_PGRP, process_group_id)
    except OSError as error:
        return join(signal_error, wrap(f"inspect process group {process_group_id}", error))
    if process_group_has_live_members(members):
        return signal_error
    return None


def kinfo_proc(pid: int) -> KinfoProc:
    members = kinfo_proc_slice(_KERN_PROC_PID, pid)
    if len(members) != 1:
        raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH))
    return members[0]


def kinfo_proc_slice(kind: int, argument: int) -> list[KinfoProc]:
    sysctl = _libc().sysctl
    mib = (ctypes.c_int * 4)(_CTL_KERN, _KERN_PROC, kind, argument)
    while True:
        size = ctypes.c_size_t(0)
        if sysctl(mib, 4, None, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        capacity = size.value + 4 * _KINFO_PROC_SIZE
        buffer = ctypes.create_string_buffer(capacity)
        size = ctypes.c_size_t(capacity)
        if sysctl(mib, 4, buffer, ctypes.byref(size), None, 0) != 0:
            code = ctypes.get_errno()
            if code == errno.ENOMEM:
                continue
            raise OSError(code, os.strerror(code))
        break
    raw = buffer.raw[: size.value]
    members: list[KinfoProc] = []
    for start in range(0, len(raw) - _KINFO_PROC_SIZE + 1, _KINFO_PROC_SIZE):
        record = raw[start : start + _KINFO_PROC_SIZE]
        members.append(
            KinfoProc(
                p_pid=int.from_bytes(
                    record[_P_PID_OFFSET : _P_PID_OFFSET + 4], "little", signed=True
                ),
                p_stat=int.from_bytes(
                    record[_P_STAT_OFFSET : _P_STAT_OFFSET + 1], "little", signed=True
                ),
                p_flag=int.from_bytes(
                    record[_P_FLAG_OFFSET : _P_FLAG_OFFSET + 4], "little", signed=True
                ),
            )
        )
    return members


_LIBC: ctypes.CDLL | None = None


def _libc() -> ctypes.CDLL:
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _LIBC.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        _LIBC.sysctl.restype = ctypes.c_int
    return _LIBC
