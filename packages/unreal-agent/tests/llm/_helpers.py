LEGACY_PAYLOADS = [
    '{"type":"reasoning","encrypted_content":"opaque"}',
    '{"type":"thinking","thinking":"original","signature":"signed"}',
    '{"type":"redacted_thinking","data":"opaque"}',
    '{"vendor_payload":{"n":9007199254740993}}',
]

PASSTHROUGH_PAYLOADS = [
    '{"type":"future_block","data":1e1000}',
    '{"type":42,"data":9007199254740993}',
    '{"vendor_payload":{"data":"opaque"}}',
]
