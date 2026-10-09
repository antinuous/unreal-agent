import pytest
from unreal_agent import gojson
from unreal_agent.contextbuilder import new_builder
from unreal_agent.gojson import Raw
from unreal_agent.llm import (
    Item,
    ItemType,
    Model,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Response,
)
from unreal_agent.llm.responsesapi import decode_response, request_body
from unreal_agent.llm.responsesapi.request import request_input_item

from tests.llm._helpers import LEGACY_PAYLOADS, PASSTHROUGH_PAYLOADS
from tests.llm.responsesapi._helpers import valid_request


@pytest.mark.parametrize(
    "raw",
    [
        '{"id":"r","type":"reasoning","summary":[],"encrypted_content":"opaque","extra":1e1000}',
        '{"id":"r","type":"reasoning","summary":[{"type":"summary_text","text":"Consider tools"}],'
        '"encrypted_content":"opaque","extra":9007199254740993}',
    ],
)
def test_reasoning_persists_and_replays(raw: str) -> None:
    response = decode_response(
        b'{"id":"response","status":"completed","output":[' + raw.encode() + b"]}"
    )
    provider = (response.Output or [])[0].Data
    assert (response.Output or [])[0].Type == ItemType.PROVIDER
    assert isinstance(provider, ProviderItem)
    assert bytes(provider.Raw) == raw.encode()
    restored = gojson.decode(gojson.encode(response), Response)
    assert restored == response, "persistence changed the response"
    builder = new_builder()
    builder.set_model(Model(ID="test"))
    builder.add_model_response(restored)
    built = builder.build()
    assert raw.encode() in request_body(built.request, ""), "replay changed provider state"


@pytest.mark.parametrize(
    "kind",
    [
        "future_block",
        "local_shell_call",
        "shell_call",
        "computer_call",
        "custom_tool_call",
        "mcp_approval_request",
        "apply_patch_call",
        "tool_search_call",
    ],
)
def test_response_rejects_unsupported_provider_output(kind: str) -> None:
    with pytest.raises(ValueError, match=f'unsupported output item type "{kind}"'):
        decode_response(
            b'{"id":"response","status":"completed","output":[{"type":"' + kind.encode() + b'"}]}'
        )


def test_reasoning_summary_preserves_paragraphs() -> None:
    response = decode_response(
        b'{"id":"response","status":"completed","output":[{"id":"r","type":"reasoning",'
        b'"summary":[{"type":"summary_text","text":"first"},'
        b'{"type":"summary_text","text":"second"}]}]}'
    )
    provider = (response.Output or [])[0].Data
    assert isinstance(provider, ProviderItem)
    assert provider.Display == ProviderDisplay(
        Kind=ProviderDisplayKind.REASONING, Text="first\n\nsecond"
    )


@pytest.mark.parametrize("data", ["{}", '{"Summary":["visible"]}', '{"Raw":null}'])
def test_legacy_reasoning_without_raw_cannot_replay(data: str) -> None:
    item = gojson.decode('{"Type":"reasoning","Data":' + data + "}", Item)
    request = valid_request()
    request.Input.append(item)
    with pytest.raises(ValueError, match="provider item Raw is required"):
        request_body(request, "")


@pytest.mark.parametrize("raw", LEGACY_PAYLOADS)
def test_legacy_reasoning_replay(raw: str) -> None:
    item = gojson.decode(
        '{"Type":"reasoning","Data":{"Summary":["display only"],"Raw":' + raw + "}}", Item
    )
    assert bytes(request_input_item(item)) == raw.encode(), "legacy payload changed"


@pytest.mark.parametrize("raw", PASSTHROUGH_PAYLOADS)
def test_provider_payload_passes_through(raw: str) -> None:
    item = Item(
        Type=ItemType.PROVIDER, Data=ProviderItem(Type="display_metadata", Raw=Raw(raw.encode()))
    )
    assert bytes(request_input_item(item)) == raw.encode(), "provider payload changed"
