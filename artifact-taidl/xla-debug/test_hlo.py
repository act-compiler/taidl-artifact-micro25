import jax
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_client
from jax.extend import ffi
from jax.extend.backend import get_backend
import os
import sys
from pathlib import Path

build_dir = Path("build").resolve()
sys.path.append(str(build_dir))

import print_handler_ext


def compile_hlo_text(hlo_text: str, backend):
    for name, target in print_handler_ext.registrations().items():
        ffi.register_ffi_target(name, target, "cpu")

    options = xla_client.CompileOptions()
    hlo_module = xla_client._xla.hlo_module_from_text(hlo_text)
    computation = xla_client.XlaComputation(hlo_module.as_serialized_hlo_module_proto())
    mlir_module = xla_client._xla.mlir.xla_computation_to_mlir_module(computation)
    executable = backend.compile(mlir_module, compile_options=options)
    print("Compiled executable from HLO text.")
    return executable


hlo_text = """
ENTRY main {
  prefix = u8[2] constant({120, 121})
  data = f64[2]{0} constant({3.2, 4.5})
  ROOT result = (s32[]) custom-call(prefix, data), custom_call_target="print_handler", api_version=API_VERSION_TYPED_FFI
}
"""

backend = get_backend("cpu")
executable = compile_hlo_text(hlo_text, backend)

print("Executing compiled HLO...")
result = executable.execute([])
result[0].block_until_ready()

print("Execution finished.")
