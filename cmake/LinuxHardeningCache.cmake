# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Shared Linux hardening flags for LLVM/PTOAS package builds.
# This cache file is intended to be passed via `cmake -C ...` so release and
# delivery builds inherit the same compiler options that codecheck expects.

foreach(_flag_var CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  set(_flag_value "${${_flag_var}}")
  foreach(_hardening_flag
      -D_FORTIFY_SOURCE=2
      -fstack-protector-strong
      -ftrapv)
    if(NOT " ${_flag_value} " MATCHES "(^| )${_hardening_flag}( |$)")
      string(APPEND _flag_value " ${_hardening_flag}")
    endif()
  endforeach()
  string(STRIP "${_flag_value}" _flag_value)
  set(${_flag_var} "${_flag_value}" CACHE STRING "Linux hardening flags" FORCE)
endforeach()
