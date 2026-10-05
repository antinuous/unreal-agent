"""Property test for image preparation.

testdata/fuzz_seeds.json holds seeds encoded by Go's image encoders. Pillow
cannot reproduce the 16-bit and paletted TIFF variants byte for byte.
"""

import base64
import io
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from PIL import Image
from unreal_agent.operation import ViewImageConfig
from unreal_agent.operation.image import prepare_view_image

_SEEDS = json.loads((Path(__file__).parent / "testdata" / "fuzz_seeds.json").read_text())


def _with_seeds[F: Callable[..., Any]](test: F) -> F:
    for seed in _SEEDS:
        data = base64.b64decode(seed["Data"])
        test = example(data, seed["Split"], seed["MaxSize"], seed["MaxWidth"], seed["MaxHeight"])(
            test
        )
    return test


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@_with_seeds
@given(
    st.binary(max_size=64 << 10),
    st.integers(0, 65535),
    st.integers(0, 65535),
    st.integers(0, 255),
    st.integers(0, 255),
)
def test_prepare_view_image(
    data: bytes, split: int, max_size: int, max_width: int, max_height: int
) -> None:
    config = ViewImageConfig(
        MaxSize=max_size + 1,
        MaxWidth=max_width + 1,
        MaxHeight=max_height + 1,
        MaxSourcePixels=4096,
    )
    offset = split % (len(data) + 1)
    result, error = prepare_view_image(
        threading.Event(), [data[:offset], data[offset:]], config, None
    )
    if error is not None:
        assert result.Content == "", "failed preparation returned encoded content"
        return
    assert len(result.Content) <= config.MaxSize
    assert 0 < result.ScaleRatio <= 1
    assert result.OriginalWidth > 0
    assert result.OriginalHeight > 0
    assert result.Error == ""
    encoded = base64.b64decode(result.Content, validate=True)
    decoded = Image.open(io.BytesIO(encoded))
    decoded.load()
    want_format = "JPEG" if result.OriginalMIMEType == "image/jpeg" else "PNG"
    assert decoded.format == want_format
    assert result.EncodedMIMEType == "image/" + want_format.lower()
    assert decoded.width <= config.MaxWidth
    assert decoded.height <= config.MaxHeight
