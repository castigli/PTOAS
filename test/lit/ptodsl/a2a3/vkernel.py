# RUN: PYTHONPATH=%S/.. %python %s | FileCheck %s
from ptodsl import pto
from utils import print_module

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

