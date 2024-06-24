# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

import paddle
from paddle.framework import core
from paddle.fluid.framework import program_guard, device_guard
from paddle.fluid import unique_name, layers
from paddle.fluid.clip import append_gradient_clip_ops
from paddle.distributed.auto_parallel.utils import set_var_dist_attr
from paddle.distributed.auto_parallel.utils import naive_set_dist_op_attr_for_program_by_mesh_and_mapping
from paddle.distributed.auto_parallel.process_group import get_world_process_group
world_process_group = get_world_process_group()


def _is_the_backward_op(op):
    OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
    OpRole = core.op_proto_and_checker_maker.OpRole
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)


def _is_the_optimizer_op(op):
    OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
    OpRole = core.op_proto_and_checker_maker.OpRole
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Optimize)

def _is_the_grad_allreduce_op(op, grad_names):

    if op.type != "c_allreduce_sum":
        return False
    output_var_name = op.output_arg_names[0]
    if output_var_name not in grad_names:
        return False

    return True


def _remove_and_get_optimizer_op(main_program, params_grads, allreduce_in_update):
    # 1 create tmp block
    # 2 mv optimizer op from global program to tmp block
    # 3 del the op from dist_context
    from paddle.distributed.fleet.meta_optimizers.common import OpRole
    main_block = main_program.global_block()
    temp_block = main_program._create_block()
    removed_op_idx = []
    optimize_ops_desc = []
    grad_names = [g.name for p, g in params_grads]
    for idx, op in enumerate(main_block.ops):
        if allreduce_in_update and _is_the_grad_allreduce_op(op, grad_names):
            removed_op_idx.append(idx)

        if _is_the_optimizer_op(op):
            # append optimizer op to tmp block
            new_op_desc = temp_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            optimize_ops_desc.append(new_op_desc)
            removed_op_idx.append(idx)

    for idx in removed_op_idx[::-1]:
        main_block._remove_op(idx)

    return optimize_ops_desc


def _remove_op_role_var(param, grad):
    op_maker = core.op_proto_and_checker_maker
    op = grad.op
    if op.has_attr(op_maker.kOpRoleVarAttrName()):
        op._remove_attr(op_maker.kOpRoleVarAttrName())


def _get_gm_cond_var(main_program, k_steps):
    main_block = main_program.global_block()
    # Add const var
    k_step_var = layers.create_global_var(
        name="gradient_merge_k",
        shape=[1],
        value=int(k_steps),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    zero_var = layers.create_global_var(
        name="gradient_merge_zero",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    # Add step var & cond var
    step_var = layers.create_global_var(
        name="gradient_merge_step",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    cond_var = main_block.create_var(
        name="gradient_merge_cond", shape=[1], dtype='bool')

    with device_guard("cpu"):
        # step_var = (step_var + 1) % k_step
        layers.increment(x=step_var, value=1.0, in_place=True)
        elementwise_mod_op = main_block.append_op(
            type='elementwise_mod',
            inputs={'X': step_var,
                    'Y': k_step_var},
            outputs={'Out': step_var},
            attrs={'axis': -1,
                   'use_mkldnn': False})

        # cond_var = (step_var == 0)
        equal_op = main_block.append_op(
            type='equal',
            inputs={'X': step_var,
                    'Y': zero_var},
            outputs={'Out': cond_var})

    return cond_var


def _append_gradient_merge_backward_op(
        main_program,
        startup_program,
        params_grads: List[Tuple[Any, Any]],
        cond_var_name: str) -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    main_block = main_program.global_block()
    startup_block = startup_program.global_block()

    # step1: remove grad.op's op_role_var
    for param, grad in params_grads:
        assert (
            param.type != core.VarDesc.VarType.SELECTED_ROWS
        ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

        _remove_op_role_var(param, grad)

    param_to_gradient_merge = {}
    new_params_to_grads = []
    # step2: create gradient_merge var and init with 0
    for param, grad in params_grads:
        param_name = param.name
        param_var = main_block.var(param_name)
        assert (param_var is not None)
        gradient_merge_var = main_block.create_var(
            name=param_name + "@GRAD@GradientMerge",
            shape=param_var.shape,
            dtype=param_var.dtype,
            persistable=True)
        param_to_gradient_merge[param_name] = gradient_merge_var

        startup_gradient_merge_var = startup_block.create_var(
            name=param_name + "@GRAD@GradientMerge",
            shape=param_var.shape,
            dtype=param_var.dtype,
            persistable=True)
        startup_block.append_op(
            type="fill_constant",
            outputs={"Out": startup_gradient_merge_var},
            attrs={
                "shape": param_var.shape,
                "dtype": param_var.dtype,
                "value": float(0),
            })

        # grad_merge += grad
        new_grad_op = main_block.append_op(
            type="elementwise_add",
            inputs={'X': grad,
                    'Y': gradient_merge_var},
            outputs={'Out': gradient_merge_var},
            attrs={'axis': -1,
                   'use_mkldnn': False})
        new_params_to_grads.append([param, gradient_merge_var])

    return new_params_to_grads, param_to_gradient_merge


def _create_cond_block_and_update_optimizer(
        main_program,
        cond_var,
        new_params_to_grads: List[Tuple[Any, Any]],
        param_to_gradient_merge: Dict[str, Any],
        optimize_ops_desc: List[Any],
        k_steps,
        avg,
        allreduce_in_update):
    def true_apply_gradient():
        cur_block_idx = main_program.current_block_idx
        cur_block = main_program.current_block()

        # cur_block's forward_block & backward_block is itself
        cur_block._set_forward_block_idx(cur_block_idx)
        op_maker = core.op_proto_and_checker_maker
        if avg:
            for param, new_grad in new_params_to_grads:
                # grad /= k_steps
                cur_block.append_op(
                    type='scale',
                    inputs={'X': new_grad},
                    outputs={'Out': new_grad},
                    attrs={
                        'scale': 1.0 / k_steps,
                        'bias': 0.0,
                        'bias_after_scale': False
                    })
                new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                      op_maker.OpRole.Optimize)

        # gradient allreduce
        if allreduce_in_update:
            for param, new_grad in new_params_to_grads:
                cur_block.append_op(
                    type='c_allreduce_sum',
                    inputs={'X': [new_grad]},
                    outputs={'Out': [new_grad]},
                    attrs={
                        'ring_id': world_process_group.id,
                        'use_calc_stream': True,
                        op_maker.kOpRoleAttrName(): op_maker.OpRole.Optimize
                    })

        # append optimizer ops
        for op_desc in optimize_ops_desc:
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            #update input/output
            for input_name in new_op_desc.input_arg_names():
                if input_name in new_params_to_grads:
                    new_op_desc._rename_input(input_name,
                                              new_params_to_grads[input_name])

            for output_name in new_op_desc.output_arg_names():
                if output_name in new_params_to_grads:
                    new_op_desc._rename_output(output_name,
                                               new_params_to_grads[output_name])

            # remove op_role_var
            if new_op_desc.has_attr(op_maker.kOpRoleVarAttrName()):
                new_op_desc.remove_attr(op_maker.kOpRoleVarAttrName())

            # op's update Grad
            if core.grad_var_suffix() in new_op_desc.input_arg_names():
                grad_value = new_op_desc.input("Grad")[0]
                # TODO FIXME(xym) support fp16
                grad_merge_value = grad_value + '@GradientMerge'
                new_op_desc.set_input("Grad", [grad_merge_value])

        main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()

        # clear gradient_merge_vars
        for param, new_grad in new_params_to_grads:
            layers.fill_constant(
                shape=new_grad.shape,
                dtype=new_grad.dtype,
                value=0.0,
                out=new_grad)
            new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                  op_maker.OpRole.Optimize)

    layers.cond(cond_var, true_fn=true_apply_gradient, false_fn=None)


def parse_program(main_program, startup_program, params_grads, k_steps, avg, allreduce_in_update = False ):
    # 1 create gradient_merge_cond
    cond_var = _get_gm_cond_var(main_program, k_steps)

    # 2 remove optimizer_op from main_program
    optimize_ops_desc = _remove_and_get_optimizer_op(main_program, params_grads, allreduce_in_update)

    # back to block 0
    main_program._rollback()

    # 3 append gradient merge backward op to main_program
    new_params_to_grads, param_to_gradient_merge = _append_gradient_merge_backward_op(
        main_program, startup_program, params_grads, cond_var.name)

    # 4 create ConditionalBlock and append gradient merge optimizer ops
    _create_cond_block_and_update_optimizer(
        main_program, cond_var, new_params_to_grads, param_to_gradient_merge,
        optimize_ops_desc, k_steps, avg, allreduce_in_update)
