import os
from typing import Any

import pytest
from hypothesis import settings

# Property tests pin derandomized, database-free settings for fast, repeatable
# runs. A fuzz profile scales each test's own max_examples, draws random
# inputs, and stores failures in .hypothesis/ so they replay on the next run.
_FUZZ_SCALE = {"fuzz": 25, "fuzz-ci": 10}

for _name in _FUZZ_SCALE:
    settings.register_profile(
        _name, settings.get_profile("default"), deadline=None, print_blob=True
    )

_profile = os.environ.get("HYPOTHESIS_PROFILE", "")
if _profile:
    settings.load_profile(_profile)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    scale = _FUZZ_SCALE.get(settings.get_current_profile_name())
    if scale is None:
        return
    profile = settings()
    for item in items:
        test: Any = getattr(item, "obj", None)
        if not getattr(test, "is_hypothesis_test", False):
            continue
        current: settings = test._hypothesis_internal_use_settings
        test._hypothesis_internal_use_settings = settings(
            current,
            max_examples=current.max_examples * scale,
            derandomize=False,
            database=profile.database,
            print_blob=True,
        )
