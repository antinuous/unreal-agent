import pytest
from unreal_agent import gojson
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode, Inbox, Input, InputKind, Settings
from unreal_agent.llm import ReasoningEffort


async def submit_and_receive(inbox: Inbox, input: Input) -> Input:
    await inbox.submit(input)
    return inbox.get_nowait()


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE, ControlMode.HEARTBEAT]
)
async def test_inbox_control_messages(mode: ControlMode) -> None:
    want = ControlMessage(Mode=mode, Reason="stop now")
    payload = gojson.encode(want)
    input = Input(ID="stop", Kind=InputKind.CONTROL, Payload=Raw(payload))
    got = (await submit_and_receive(Inbox(), input)).decode_control_message()
    assert got == want


@pytest.mark.parametrize("effort", [effort for effort in ReasoningEffort if effort])
async def test_inbox_settings_controls(effort: ReasoningEffort) -> None:
    want = ControlMessage(
        Mode=ControlMode.UPDATE_SETTINGS, Parameters=Settings(ReasoningEffort=effort)
    )
    payload = gojson.encode(want)
    assert payload == (
        b'{"Mode":"settings","Reason":"","Parameters":{"ReasoningEffort":"%s"}}' % effort.encode()
    )
    input = Input(ID="settings", Kind=InputKind.CONTROL, Payload=Raw(payload))
    got = (await submit_and_receive(Inbox(), input)).decode_control_message()
    assert got == want


@pytest.mark.parametrize(
    "settings",
    [
        Settings(),
        Settings(Model="model"),
        Settings(MaxOutputTokens=2048),
        Settings(Model="model", MaxOutputTokens=2048, ReasoningEffort=ReasoningEffort.HIGH),
    ],
)
async def test_inbox_partial_settings_controls(settings: Settings) -> None:
    want = ControlMessage(Mode=ControlMode.UPDATE_SETTINGS, Parameters=settings)
    payload = gojson.encode(want)
    input = Input(ID="settings", Kind=InputKind.CONTROL, Payload=Raw(payload))
    got = (await submit_and_receive(Inbox(), input)).decode_control_message()
    assert got == want


def test_settings_encoding_matches_go() -> None:
    settings = Settings(Model="model", MaxOutputTokens=2048, ReasoningEffort=ReasoningEffort.HIGH)
    assert gojson.encode(settings) == (
        b'{"Model":"model","MaxOutputTokens":2048,"ReasoningEffort":"high"}'
    )
    assert gojson.encode(Settings()) == b"{}"


def test_control_message_encoding_omits_zero_parameters() -> None:
    assert (
        gojson.encode(ControlMessage(Mode=ControlMode.STOP_HARD, Reason="r"))
        == b'{"Mode":"hard","Reason":"r"}'
    )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{}",
        '{"Mode":"unknown"}',
        '{"Mode":42}',
        '{"Mode":"hard"',
        '{"Mode":"heartbeat","Reason":"waiting","extra":true}',
        '{"Mode":"heartbeat"}',
        '{"Mode":"heartbeat","Reason":""}',
        '{"Mode":"settings","Parameters":[]}',
        '{"Mode":"settings","Parameters":"high"}',
        '{"Mode":"settings","Parameters":{"ReasoningEffort":"default"}}',
        '{"Mode":"settings","Parameters":{"ReasoningEffort":42}}',
        '{"Mode":"settings","Parameters":{"Model":" "}}',
        '{"Mode":"settings","Parameters":{"MaxOutputTokens":0}}',
        '{"Mode":"settings","Parameters":{"MaxOutputTokens":-1}}',
        '{"Mode":"settings","Parameters":{"SystemPrompt":"be concise"}}',
        '{"Mode":"settings","Parameters":{"SystemPrompt":""}}',
        '{"Mode":"settings","Parameters":{"Model":"model","SystemPrompt":"be concise"}}',
        '{"Mode":"settings","Parameters":{"ReasoningEffort":"high","extra":true}}',
        '{"Mode":"settings","Parameters":{"ReasoningEffort":"high"},"extra":true}',
        '{"Mode":"hard","Parameters":{"ReasoningEffort":"high"}}',
        '{"Mode":"when_idle","Parameters":{"ReasoningEffort":"high"}}',
        '{"Mode":"heartbeat","Reason":"waiting","Parameters":{"ReasoningEffort":"high"}}',
    ],
)
async def test_inbox_rejects_invalid_control_messages(payload: str) -> None:
    input = Input(ID="stop", Kind=InputKind.CONTROL, Payload=Raw(payload.encode()))
    with pytest.raises(ValueError, match=r"^submit input: "):
        await Inbox().submit(input)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("null", 'unsupported control mode ""'),
        ('{"Mode":"soft"}', 'unsupported control mode "soft"'),
        ('{"Mode":"hard","extra":true}', "decode control message: "),
        ('{"Mode":"heartbeat","Reason":null}', "heartbeat reason is empty"),
        ('{"Mode":"settings"}', "settings parameters are missing"),
        ('{"Mode":"settings","Parameters":null}', "settings parameters are missing"),
        ('{"Mode":"settings","Parameters":[]}', "decode settings parameters: "),
        ('{"Mode":"settings","Parameters":{"ReasoningEffort":"turbo"}}', '"turbo"'),
        ('{"Mode":"settings","Parameters":{"Model":" "}}', "model ID is empty"),
        (
            '{"Mode":"settings","Parameters":{"MaxOutputTokens":0}}',
            "max output tokens must be positive",
        ),
        ('{"Mode":"hard","Parameters":null}', 'control mode "hard" does not accept parameters'),
    ],
)
def test_control_message_errors(payload: str, message: str) -> None:
    input = Input(ID="stop", Kind=InputKind.CONTROL, Payload=Raw(payload.encode()))
    with pytest.raises(ValueError, match=message):
        input.decode_control_message()


def test_external_input_is_not_a_control_message() -> None:
    input = Input(Kind=InputKind.EXTERNAL, Payload=Raw(b'{"Mode":"hard"}'))
    with pytest.raises(ValueError, match='control message input has kind "external"'):
        input.decode_control_message()
