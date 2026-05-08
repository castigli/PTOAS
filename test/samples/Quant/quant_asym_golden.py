#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden reference for the TQuant INT8_ASYM kernel."""

import numpy as np
from pathlib import Path
import sys

_ROWS = 32
_COLS = 32
for search_root in (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1],
):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    src_name, scale_name, off_name = meta.inputs
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style="signed")
    scale = float_values(generator, meta.elem_counts[scale_name], style="positive")
    off = float_values(generator, meta.elem_counts[off_name], style="signed_small")
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[scale_name] = scale
    buffers[off_name] = off
    write_buffers(meta, buffers)
    scale_2d = scale.reshape(_ROWS, 1)
    off_2d = off.reshape(_ROWS, 1)
    out = np.clip(
        np.round(
            src.reshape(_ROWS, _COLS) * scale_2d
            + off_2d
        ),
        0, 255,
    ).astype(np.uint8)
    write_golden(meta, {single_output(meta): out})


if __name__ == "__main__":
    main()
