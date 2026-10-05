"""Durable session and turn identity."""

from enum import StrEnum

from unreal_agent.gojson import Struct, Timestamp

__all__ = ["ID", "Session", "Turn", "TurnID", "TurnType"]

type ID = str
type TurnID = str


class TurnType(StrEnum):
    REGULAR = "regular"
    COMPACTION = "compaction"


class Session(Struct):
    ID: str = ""
    CreatedAt: Timestamp = Timestamp()


class Turn(Struct):
    ID: TurnID = ""
    PreviousTurnID: TurnID = ""
    Type: TurnType = ""
