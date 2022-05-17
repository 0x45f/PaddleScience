# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import six
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.static import global_scope


def l2_norm_square(x, scale=None):
    if scale is None:
        l2_norm = paddle.norm(x, p=2)
    else:
        l2_norm = paddle.norm(x * scale, p=2) / scale
    return l2_norm * l2_norm


def compute_eq_loss(inputs, outputs, labels_var):
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    u = outputs[:, 0]
    v = outputs[:, 1]
    w = outputs[:, 2]
    p = outputs[:, 3]
    u_n = labels_var[0]
    v_n = labels_var[1]
    w_n = labels_var[2]
    jac0, = paddle.static.gradients([u], [inputs])  # du/dx, du/dy, du/dz
    jac1, = paddle.static.gradients([v], [inputs])  # dv/dx, dv/dy, dv/dz
    jac2, = paddle.static.gradients([w], [inputs])  # dw/dx, dw/dy, dw/dz
    jac3, = paddle.static.gradients([p], [inputs])  # dp/dx, dp/dy, dp/dz
    hes0, = paddle.static.gradients(
        [jac0[:, 0]], [inputs])  # du*du/dx*dx, du*du/dx*dy, du*du/dx*dz
    hes1, = paddle.static.gradients(
        [jac0[:, 1]], [inputs])  # du*du/dy*dx, du*du/dy*dy, du*du/dy*dz
    hes2, = paddle.static.gradients(
        [jac0[:, 2]], [inputs])  # du*du/dz*dx, du*du/dz*dy, du*du/dz*dz
    hes3, = paddle.static.gradients(
        [jac1[:, 0]], [inputs])  # dv*dv/dx*dx, dv*dv/dx*dy, dv*dv/dx*dz
    hes4, = paddle.static.gradients(
        [jac1[:, 1]], [inputs])  # dv*dv/dy*dx, dv*dv/dy*dy, dv*dv/dy*dz
    hes5, = paddle.static.gradients(
        [jac1[:, 2]], [inputs])  # dv*dv/dz*dx, dv*dv/dz*dy, dv*dv/dz*dz
    hes6, = paddle.static.gradients(
        [jac2[:, 0]], [inputs])  # dw*dw/dx*dx, dw*dw/dx*dy, dw*dw/dx*dz
    hes7, = paddle.static.gradients(
        [jac2[:, 1]], [inputs])  # dw*dw/dy*dx, dw*dw/dy*dy, dw*dw/dy*dz
    hes8, = paddle.static.gradients(
        [jac2[:, 2]], [inputs])  # dw*dw/dz*dx, dw*dw/dz*dy, dw*dw/dz*dz

    nu = 0.01
    rho = 1.0
    dt = 1.0
    continuty = jac0[:, 0] + jac1[:, 1] + jac2[:, 2]
    # + u / dt - u_n / dt
    momentum_x = u / dt - u_n / dt + u * jac0[:, 0] + v * jac0[:, 1] + w * jac0[:, 2] - \
                nu / rho * hes0[:, 0] - nu / rho * hes1[:, 1] - nu / rho * hes2[:, 2] + \
                1.0 / rho * jac3[:, 0]
    momentum_y = v / dt - v_n / dt + u * jac1[:, 0] + v * jac1[:, 1] + w * jac1[:, 2] - \
                nu / rho * hes3[:, 0] - nu / rho * hes4[:, 1] - nu / rho * hes5[:, 2] + \
                1.0 / rho * jac3[:, 1]
    momentum_z = w / dt - w_n / dt + u * jac2[:, 0] + v * jac2[:, 1] + w * jac2[:, 2] - \
                nu / rho * hes6[:, 0] - nu / rho * hes7[:, 1] - nu / rho * hes8[:, 2] + \
                1.0 / rho * jac3[:, 2]

    rhs = 0
    wgt = np.sqrt(0.01)

    eq_loss = l2_norm_square((continuty - rhs)*wgt) + \
            l2_norm_square((momentum_x - rhs)*wgt) + \
            l2_norm_square((momentum_y - rhs)*wgt) + \
            l2_norm_square((momentum_z - rhs)*wgt)
    return eq_loss


# Convert the program into graph, apply the calculation graph optimizations, and turn back to the program
def compile_and_convert_back_to_program(program=None,
                                        feed=None,
                                        fetch_list=None,
                                        fetch_var_name='fetch',
                                        scope=None,
                                        use_prune=False,
                                        loss_name=None):
    def _add_fetch_ops(program, fetch_list, fetch_var_name):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        # append fetch_operators
        if not fluid.executor.has_fetch_operators(global_block, fetch_list,
                                                  fetch_var_name, 'fetch'):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types), (
                        "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})
        return tmp_program

    def _remove_fetch_ops(program):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == 'fetch':
                global_block._remove_op(idx)

        return tmp_program

    def _compile(program, loss_name=None):
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        exec_strategy.num_threads = 1

        compiled_program = paddle.static.CompiledProgram(
            program).with_data_parallel(
                loss_name=loss_name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

        return compiled_program

    if program is None:
        program = default_main_program()

    if scope is None:
        scope = global_scope()

    executor = paddle.static.Executor()

    fetch_list = executor._check_fetch_list(fetch_list)
    fetch_list, optimize_ops = executor._split_optimize_ops_in_fetch_list(
        fetch_list)

    if optimize_ops:
        raise ValueError("Unsupport to fetch optimize OP.")

    if use_prune:
        program = executor._prune_program(program, feed, fetch_list,
                                          optimize_ops)
        feed = executor._update_feed(program, feed)

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = _compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope,
                              paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    #ir_graph.draw(save_path='./', name='compiled_graph')
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    #paddle.static.save(final_program, "final")
    return final_program
