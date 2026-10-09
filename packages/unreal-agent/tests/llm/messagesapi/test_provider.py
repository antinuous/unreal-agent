import pytest
from unreal_agent import gojson
from unreal_agent.gojson import Raw
from unreal_agent.llm import Item, ItemType, ProviderItem, Response
from unreal_agent.llm.messagesapi import decode_response, request_body
from unreal_agent.llm.messagesapi.request import request_input_block

from tests.llm._helpers import LEGACY_PAYLOADS, PASSTHROUGH_PAYLOADS
from tests.llm.messagesapi._helpers import response_body, valid_request


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"thinking","thinking":"Consider tools","signature":"signed",'
        '"extra":9007199254740993}',
        '{"type":"redacted_thinking","data":"opaque","extra":1e1000}',
    ],
)
def test_thinking_persists_and_replays(raw: str) -> None:
    response = decode_response(response_body("end_turn", f"[{raw}]"))
    provider = (response.Output or [])[0].Data
    assert (response.Output or [])[0].Type == ItemType.PROVIDER
    assert isinstance(provider, ProviderItem)
    assert bytes(provider.Raw) == raw.encode()
    restored = gojson.decode(gojson.encode(response), Response)
    assert restored == response, "persistence changed provider output"
    request = valid_request()
    request.Input.extend(restored.Output or [])
    assert raw.encode() in request_body(request), "replay changed provider state"


@pytest.mark.parametrize("data", ["{}", '{"Summary":["visible"]}', '{"Raw":null}'])
def test_legacy_reasoning_without_raw_cannot_replay(data: str) -> None:
    item = gojson.decode('{"Type":"reasoning","Data":' + data + "}", Item)
    request = valid_request()
    request.Input.append(item)
    with pytest.raises(ValueError, match="provider item Raw is required"):
        request_body(request)


@pytest.mark.parametrize("raw", LEGACY_PAYLOADS)
def test_legacy_reasoning_replay(raw: str) -> None:
    item = gojson.decode(
        '{"Type":"reasoning","Data":{"Summary":["display only"],"Raw":' + raw + "}}", Item
    )
    assert bytes(request_input_block(item)) == raw.encode(), "legacy payload changed"


@pytest.mark.parametrize("raw", PASSTHROUGH_PAYLOADS)
def test_provider_payload_passes_through(raw: str) -> None:
    item = Item(
        Type=ItemType.PROVIDER, Data=ProviderItem(Type="display_metadata", Raw=Raw(raw.encode()))
    )
    assert bytes(request_input_block(item)) == raw.encode(), "provider payload changed"
