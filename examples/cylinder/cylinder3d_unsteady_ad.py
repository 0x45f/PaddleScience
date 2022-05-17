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

import numpy as np
import paddlescience as psci
import paddle
from paddle.incubate.autograd import prim2orig, enable_prim, prim_enabled
from utils import l2_norm_square, compute_eq_loss, compile_and_convert_back_to_program

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
enable_prim()

# define start time and time step
start_time = 100
time_step = 1


# load real data
def GetRealPhyInfo(time, need_cord=False, need_physic=False):
    real_data = np.load("./flow_unsteady_re200/flow_re200_" + str(time) +
                        "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    if need_cord is False and need_physic is False:
        print("Error: you need to get cord or get physic infomation")
        exit()
    elif need_cord is True and need_physic is True:
        return real_data
    elif need_cord is True and need_physic is False:
        return real_data[:, 0:3]
    elif need_cord is False and need_physic is True:
        return real_data[:, 3:7]
    else:
        pass


# get init physic infomation
def GenInitPhyInfo(xyz):
    uvw = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvw[i][0] = 1.0
    return uvw


cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2), extent=(25, 8, 2), circle_center=cc, circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
geo_disc = geo.discretize(npoints=40000, method="sampling")
# the real_cord need to be added in geo_disc
real_cord = GetRealPhyInfo(start_time, need_cord=True)
geo_disc.user = real_cord

# N-S equation
pde = psci.pde.NavierStokes(
    nu=0.01,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[0.01, 0.01, 0.01, 0.01])

pde.set_time_interval([100.0, 110.0])

# boundary condition on left side: u=10, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=1.0, weight=1.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0, weight=1.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0, weight=1.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# pde discretization 
pde_disc = pde.discretize(
    time_method="implicit", time_step=time_step, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# create inputs/labels and its attributes
inputs, inputs_attr = algo.create_inputs(pde_disc)
labels, labels_attr = algo.create_labels(pde_disc)

main_program = paddle.static.Program()
startup_program = paddle.static.Program()

with paddle.static.program_guard(main_program, startup_program):
    algo.net.make_network_static()
    inputs_var = []
    labels_var = []
    outputs_var = []

    # inputs
    for i in range(len(inputs)):
        input = paddle.static.data(
            name='input' + str(i), shape=inputs[i].shape, dtype='float32')
        input.stop_gradient = False
        inputs_var.append(input)

    # labels
    for i in range(len(labels)):
        # Hard code here for label shape. Shape may change when random seed changed 
        if i in [0, 1, 2]:
            shape = (37174, )
        else:
            shape = (3415, )
        label = paddle.static.data(
            name='label' + str(i), shape=shape, dtype='float32')
        label.stop_gradient = False
        labels_var.append(label)

    for var in inputs_var:
        ret = algo.net.nn_func(var)
        outputs_var.append(ret)

    # bc loss
    name2index = {'u': 0, 'v': 1, 'w': 2, 'p': 3}
    bc_loss = 0.0
    name_list = []
    for i, name_b in enumerate(inputs_attr["bc"].keys()):
        # from outputs_var[1] to outputs_var[3]
        out_el = outputs_var[i + 1]
        for j in range(len(pde_disc.bc[name_b])):
            rhs_b = labels_attr["bc"][name_b][j]["rhs"]
            wgt_b = labels_attr["bc"][name_b][j]["weight"]
            index = name2index.get(pde_disc.bc[name_b][j].name)

            bc_loss += l2_norm_square(
                (out_el[:, index] - rhs_b) * np.sqrt(wgt_b), 10000)

    # inputs_var[0] eq loss
    output_var_0_eq_loss = compute_eq_loss(inputs_var[0], outputs_var[0],
                                           labels_var[0:3])

    # inputs_var[4] eq loss
    input_i = inputs_var[4]
    out_i = outputs_var[4]
    output_var_4_eq_loss = compute_eq_loss(inputs_var[4], outputs_var[4],
                                           labels_var[7:10])
    # data_loss
    data_loss = l2_norm_square(outputs_var[4][:, 0]-labels_var[3]) + \
                l2_norm_square(outputs_var[4][:, 1]-labels_var[4]) + \
                l2_norm_square(outputs_var[4][:, 2]-labels_var[5]) + \
                l2_norm_square(outputs_var[4][:, 3]-labels_var[6])

    # total_loss
    total_loss = paddle.sqrt(bc_loss + output_var_0_eq_loss +
                             output_var_4_eq_loss + data_loss)
    paddle.optimizer.Adam(0.001).minimize(total_loss)

    if prim_enabled():
        prim2orig(main_program.block(0))

place = paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)
exe.run(startup_program)

feeds = dict()
for i in range(len(inputs)):
    feeds['input' + str(i)] = inputs[i]

fetches = [total_loss.name]
for var in outputs_var:
    fetches.append(var.name)

main_program = compile_and_convert_back_to_program(
    main_program,
    feed=feeds,
    fetch_list=fetches,
    use_prune=True,
    loss_name=total_loss.name)

# num_epoch in train
train_epoch = 2000

# Solver time: (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
num_time_step = 10
current_interior = np.zeros(
    (len(pde_disc.geometry.interior), 3)).astype(np.float32)
current_user = GetRealPhyInfo(start_time, need_physic=True)[:, 0:3]
for i in range(num_time_step):
    next_time = start_time + (i + 1) * time_step
    print("############# train next time=%f train task ############" %
          next_time)
    self_lables = algo.feed_data_interior_cur(labels, labels_attr,
                                              current_interior)
    self_lables = algo.feed_data_user_cur(self_lables, labels_attr,
                                          current_user)
    self_lables = algo.feed_data_user_next(
        self_lables, labels_attr, GetRealPhyInfo(
            next_time, need_physic=True))
    for j in range(len(self_lables)):
        feeds['label' + str(j)] = self_lables[j]

    for k in range(train_epoch):
        out = exe.run(main_program, feed=feeds, fetch_list=fetches)
        print("autograd epoch: " + str(k + 1), "    loss:", out[0])
    next_uvwp = out[1:]
    # # Save vtk
    # file_path = "train_flow_unsteady_re200/fac3d_train_rslt_" + str(next_time)
    # psci.visu.save_vtk(filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)

    # next_info -> current_info
    next_interior = np.array(next_uvwp[0])
    next_user = np.array(next_uvwp[-1])
    current_interior = next_interior[:, 0:3]
    current_user = next_user[:, 0:3]
