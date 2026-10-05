from unreal_agent import gojson
from unreal_agent.gojson import Timestamp
from unreal_agent.session import Session, Turn, TurnType


def test_session_json_matches_go() -> None:
    session = Session(ID="golden-session", CreatedAt=Timestamp.parse("2026-08-27T10:00:00Z"))
    encoded = gojson.encode(session)
    assert encoded == b'{"ID":"golden-session","CreatedAt":"2026-08-27T10:00:00Z"}'
    assert gojson.decode(encoded, Session) == session


def test_turn_json_matches_go() -> None:
    turn = Turn(ID="child-turn-2", PreviousTurnID="child-turn-1", Type=TurnType.COMPACTION)
    encoded = gojson.encode(turn)
    assert encoded == b'{"ID":"child-turn-2","PreviousTurnID":"child-turn-1","Type":"compaction"}'
    assert gojson.decode(encoded, Turn) == turn


def test_zero_values_encode_like_go() -> None:
    assert gojson.encode(Turn()) == b'{"ID":"","PreviousTurnID":"","Type":""}'
    assert gojson.encode(Session()) == b'{"ID":"","CreatedAt":"0001-01-01T00:00:00Z"}'
