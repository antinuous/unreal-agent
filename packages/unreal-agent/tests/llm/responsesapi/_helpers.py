from unreal_agent.gojson import Raw
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Reasoning,
    Request,
    Role,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
)

COMPLETED_RESPONSE = (
    b'{"id":"resp-1","status":"completed","output":[{"id":"fc-1","type":"function_call",'
    b'"call_id":"call-1","name":"Bash","arguments":"{\\"command\\":\\"pwd\\"}","status":"completed"},'
    b'{"id":"rs-1","type":"reasoning","encrypted_content":"opaque","summary":[{"type":"summary_text",'
    b'"text":"Check directory"}]}],"usage":{"input_tokens":12,"input_tokens_details":'
    b'{"cached_tokens":5},"output_tokens":8,"output_tokens_details":{"reasoning_tokens":4}}}'
)


FALLBACK_CALL = (
    b'{"id":"fc-1","type":"function_call","call_id":"call-1","name":"Bash",'
    b'"arguments":"{\\"command\\":\\"pwd\\"}","status":"completed"}'
)


FALLBACK_MESSAGE = (
    b'{"id":"msg-1","type":"message","status":"completed","role":"assistant","phase":"final_answer",'
    b'"content":[{"type":"output_text","text":"fallback"}]}'
)


def detailed_request() -> Request:
    return Request(
        Model=Model(ID="gpt-test", MaxOutputTokens=128),
        Input=[
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text="Be concise.")),
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="weather?")),
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text="Use tools.")),
            Item(
                ProviderID="previous-message",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="Checking.", Phase="commentary"),
            ),
            Item(
                ProviderID="previous-reasoning",
                Type=ItemType.REASONING,
                Data=Reasoning(
                    Summary=["Checked the request."],
                    Raw=Raw(
                        b'{"id":"previous-reasoning","type":"reasoning","status":"completed",'
                        b'"summary":[{"type":"summary_text","text":"Checked the request."}],'
                        b'"content":[{"type":"reasoning_text","text":"verbatim"}],'
                        b'"encrypted_content":"previous-opaque"}'
                    ),
                ),
            ),
            Item(
                ProviderID="previous-tool-call",
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name="weather", Arguments='{"city":"London"}'),
            ),
            Item(
                Type=ItemType.TOOL_RESULT,
                Data=ToolResult(
                    CallID="call-1",
                    Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="sunny")],
                ),
            ),
        ],
        Tools=[
            Tool(
                Type=ToolType.FUNCTION,
                Name="weather",
                Description="Get weather",
                Parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            )
        ],
    )


def valid_request() -> Request:
    return Request(
        Model=Model(ID="gpt-test"),
        Input=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello"))],
    )
