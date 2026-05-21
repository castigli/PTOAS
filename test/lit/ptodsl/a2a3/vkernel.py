# RUN: %python %s | FileCheck %s
from ptodsl import pto

# TODO: Move to a test utility class
def print_module(handle):
    # Get the MLIR module from the handle
    mod = handle.mlir_module()
    # Get the func name from the module
    func = mod.body.operations[0]
    print("// -----")
    print("// TEST_FUNCTION:", func.name.value)
    print(mod.operation.get_asm())
    return handle

# CHECK-LABEL: TEST_FUNCTION: miminal_kernel
# CHECK: module attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a2a3"} {
@print_module
@pto.jit(
    name="miminal_kernel",
    kernel_kind="vector",
    target="a2a3",
    func_attr="pto.aicore",
)
def miminal_kernel():
    c0   = pto.const(0)

