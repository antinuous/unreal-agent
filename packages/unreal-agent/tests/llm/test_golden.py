from pathlib import Path

import msgspec
import pytest
from unreal_agent import gojson
from unreal_agent.gojson import Raw
from unreal_agent.inbox import Input
from unreal_agent.llm import Item, Response
from unreal_agent.session import Session, Turn

TESTDATA = Path(__file__).parent / "testdata"
FIXTURES = ["golden-session.session.jsonl", "golden-fork.session.jsonl"]


class _Record(msgspec.Struct):
    type: str
    data: Raw


class _SessionHeader(msgspec.Struct):
    Session: Raw


class _ItemRecord(msgspec.Struct):
    Item: Raw


class _StoreItem(msgspec.Struct):
    Kind: str
    Data: Raw


class _ModelResponse(msgspec.Struct):
    Response: Raw


class _ResponseOutput(msgspec.Struct):
    Output: list[Raw] = []


def _round_trip(raw: Raw, kind: type) -> bytes:
    original = bytes(raw)
    encoded = gojson.encode(gojson.decode(original, kind))
    if b'"Type":"reasoning"' in original:
        assert b'"Type":"reasoning"' not in encoded, "retired item type survived migration"
        assert b'"Type":"provider"' in encoded
        assert gojson.encode(gojson.decode(encoded, kind)) == encoded
    else:
        assert encoded == original
    return original


def _check_fixture(path: Path) -> dict[str, int]:
    counts = {"session": 0, "turn": 0, "input": 0, "response": 0, "item": 0}
    for line in path.read_bytes().splitlines():
        record = msgspec.json.decode(line, type=_Record)
        if record.type == "session":
            _round_trip(msgspec.json.decode(record.data, type=_SessionHeader).Session, Session)
            counts["session"] += 1
            continue
        if record.type != "item":
            continue
        item = msgspec.json.decode(
            msgspec.json.decode(record.data, type=_ItemRecord).Item, type=_StoreItem
        )
        match item.Kind:
            case "turn":
                _round_trip(item.Data, Turn)
                counts["turn"] += 1
            case "input":
                _round_trip(item.Data, Input)
                gojson.decode(bytes(item.Data), Input).validate()
                counts["input"] += 1
            case "model_response":
                response = msgspec.json.decode(item.Data, type=_ModelResponse).Response
                _round_trip(response, Response)
                counts["response"] += 1
                for output in msgspec.json.decode(response, type=_ResponseOutput).Output:
                    _round_trip(output, Item)
                    counts["item"] += 1
    return counts


@pytest.mark.parametrize("name", FIXTURES)
def test_golden_fixture_round_trips_byte_identical(name: str) -> None:
    counts = _check_fixture(TESTDATA / name)
    assert counts["session"] == 1
    assert counts["turn"] >= 1
    assert counts["input"] == 1
    assert counts["response"] >= 1
    assert counts["item"] >= 2


def test_failed_response_round_trips_with_empty_stop() -> None:
    from unreal_agent import gojson
    from unreal_agent.llm import Failure, Response, StopReason

    encoded = gojson.encode(Response(ID="r", Failure=Failure(Code="c", Message="m")))
    assert b'"Stop":""' in encoded
    decoded = gojson.decode(encoded, Response)
    assert decoded.Stop is StopReason.NONE
    assert gojson.encode(decoded) == encoded
