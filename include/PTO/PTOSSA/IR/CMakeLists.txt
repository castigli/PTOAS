# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Drive tablegen from PTOSSAOps.td (which includes PTOSSADialect.td and PTO/IR/PTOOps.td)
set(LLVM_TARGET_DEFINITIONS PTOSSAOps.td)

# Generate Dialect definitions
mlir_tablegen(PTOSSADialect.h.inc -gen-dialect-decls -dialect=pto_ssa)
mlir_tablegen(PTOSSADialect.cpp.inc -gen-dialect-defs -dialect=pto_ssa)

# Generate Operations definitions
mlir_tablegen(PTOSSAOps.h.inc -gen-op-decls)
mlir_tablegen(PTOSSAOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PTOSSAOpsIncGen)
