#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden reference for the TDequant int8 kernel.

Formula: dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]

The scale and offset tiles are 32x8 f32 row-vectors; only the first column
of each row is used (matching the ISA BRC_B32 broadcast semantics).

Note: the i8→f32 conversion on A2/A3 goes through an intermediate half-
precision step, which may introduce rounding. The tolerance in
dequant_i8_compare.py accounts for this.
"""

import numpy as np
from pathlib import Path
import sys

_ROWS       = 32
_COLS       = 32
_PARA_COLS  = 8   # columns in the scale/offset tile buffer

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
    int_values,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    src_name, scale_name, offset_name = meta.inputs
    generator = rng()
    # i8 source values: bitwise style gives values in [-128, 128)
    src    = int_values(generator, meta.elem_counts[src_name],
                        np.int8, style='bitwise')
    scale  = float_values(generator, meta.elem_counts[scale_name], style="positive")
    offset = float_values(generator, meta.elem_counts[offset_name], style="signed_small")

    buffers = default_buffers(meta)
    buffers[src_name]    = src
    buffers[scale_name]  = scale
    buffers[offset_name] = offset
    write_buffers(meta, buffers)

    src_f32    = src.reshape(_ROWS, _COLS).astype(np.float32)
    scale_row  = scale.reshape(_ROWS, _PARA_COLS)[:, 0]   # first col per row
    offset_row = offset.reshape(_ROWS, _PARA_COLS)[:, 0]  # first col per row

    out = (src_f32 - offset_row[:, None]) * scale_row[:, None]
    write_golden(meta, {single_output(meta): out.astype(np.float32)})


if __name__ == "__main__":
    main()
