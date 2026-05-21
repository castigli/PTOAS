"""

"""
def print_module(handle):
    # Get the MLIR module from the handle
    mod = handle.mlir_module()
    # Get the func name from the module
    func = mod.body.operations[0]
    print("// -----")
    print("// TEST_FUNCTION:", func.name.value)
    print(mod.operation.get_asm())
    return handle
