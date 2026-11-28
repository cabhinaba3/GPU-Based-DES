# SPDX-License-Identifier: GPL-2.0-only
from tickdes import _core


def test_native_core_loads():
    assert _core.__doc__ == "tickdes native core"
