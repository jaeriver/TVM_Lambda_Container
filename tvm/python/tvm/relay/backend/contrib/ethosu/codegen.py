# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Codegen for Arm(R) Ethos(TM)-U"""
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.expr_functor import ExprMutator
from tvm.ir.transform import Pass

# pylint: disable=unused-import
from tvm.relay.backend.contrib.ethosu.op import op_attrs
from tvm.relay.backend.contrib.ethosu import op


class OptimizeLUTs(ExprMutator):
    """A pass to merge an identity operator with a LUT based activation function with
    a preceding operator provided that operator can do a table lookup for the activation
    in the hardware"""

    def __init__(self):
        super().__init__()
        self.lut_ops = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
        }

    def create_op_with_lut(self, call):
        """Extract the parameters and attributes from the NPU operator and create
        a new operator with LUT.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The new operator with LUT.
        """
        identity = call
        ethosu_op = call.args[0]
        lut = identity.args[1]
        activation = identity.attrs.activation

        new_attrs = dict(ethosu_op.attrs)
        new_attrs["activation"] = activation

        # Assume that LUT is always the last argument
        new_args = ethosu_op.args[:-1] + [lut]
        assert ethosu_op.op.name in self.lut_ops.keys()

        return self.lut_ops[ethosu_op.op.name](*new_args, **new_attrs)

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Recursively visit call nodes in the input graph and if an ethosu.identity
        operator with LUT is found and the preceding operator has a LUT attribute, create
        a new NPU operator.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The input call node in the case the current call node does
            not refer to an Op. Else, a new call node with a new operator.
        """
        new_call = call
        lut_activations = ["TANH", "LUT"]

        if isinstance(call.op, tvm.ir.Op) and isinstance(call.args[0], tvm.relay.expr.Call):
            producer_op = call.args[0]
            # Check if the producer can do a LUT operation
            if (
                producer_op.op.name in self.lut_ops.keys()
                and call.op.name == "contrib.ethosu.identity"
                and call.attrs.activation in lut_activations
            ):
                # Check the producer doesn't already have a LUT
                has_lut = producer_op.attrs.activation in lut_activations
                if not has_lut:
                    new_call = self.create_op_with_lut(call)

        new_call = super().visit_call(new_call)

        return new_call


@relay.transform.function_pass(opt_level=1, name="LUTsOptimizer")
class LUTsOptimizer(Pass):
    """Register LUTsOptimizer as a relay pass."""

    def transform_function(
        self, func: tvm.relay.function.Function, mod: tvm.IRModule, _
    ) -> tvm.IRModule:
        """Visit relay nodes in the given module.

        Parameters
        ----------
        func : tvm.relay.function.Function
            The function to apply the optimization pass for multiple LUTs to.
        mod : tvm.IRModule
            The module to apply the optimization pass for multiple LUTs to.

        Returns
        -------
        mod : tvm.IRModule
            New module with optimized LUTs.
        """
        assert len(mod.functions.items()) == 1, "Module can only contain one function."
        return OptimizeLUTs().visit(func)


@tvm._ffi.register_func("relay.ext.ethos-u")
def ethosu_compiler(external_function):
    """The entry-point to a compile a external relay function of
    NPU compatible operators to generated command stream.
    Such generated command stream would be used to create c-source r
    runtime module that interfaces with NPU driver.
    """
    assert isinstance(external_function, tvm.ir.function.BaseFunc)
    func_name = external_function.attrs["global_symbol"]
    # There should only be a single input
    assert len(external_function.params) == 1
    input_size = util.calculate_size_bytes(external_function.params[0])
    output_size = util.calculate_size_bytes(external_function.body)
    cmms, encoded_constants, scratch_size = _compile(external_function)
    ethosu_runtime = tvm._ffi.get_global_func("runtime.module.ethos-u.create")
    return ethosu_runtime(func_name, cmms, encoded_constants, scratch_size, input_size, output_size)


@tvm._ffi.register_func("relay.ext.ethos-u.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


def _compile(ext_func):
    """
    This is the main wrapper that accepts an external
    relay function and runs all the passes to lower it down
    to command stream
    Parameters
    ----------
    ext_func : tvm.relay.function.Function
        The partitioned relay function
    Returns
    -------
    cs : str
        An hex string of the bytes of command stream
    encoded_constants : str
        An hex string of the bytes that includes concat'd
        encoded weights, encoded biases and scales.
    scratch_size : int
        The size of the scratch buffer needed.
    """
    mod = tvm.IRModule()
    mod["main"] = ext_func
    mod = LegalizeEthosU()(mod)
    mod = LUTsOptimizer()(mod)
    mod = relay.transform.InferType()(mod)
    # We are currently using copy_constants scheduler In the long run,
    # this should be a single intelligent and a composite scheduler
    # that can perform scheduling based on user inputs such as
    # scratch memory size.
    tir_mod, params = lower_to_tir(mod["main"], copy_constants())
    cmms, encoded_constants, scratch_size = tir_to_cs_translator.translate(tir_mod, params)
    return cmms, encoded_constants, scratch_size