import sys
sys.path.append('/workspace/PaddleScience')

import numpy as np
import paddle
from paddle.fluid.incubate.ad_transform.primx import prim2orig, enable_prim, prim_enabled

import paddlescience as psci

paddle.seed(1)
np.random.seed(1)

paddle.enable_static()
# enable_prim()

def GenInitPhyInfo(xyz):
    uvwp = np.zeros((len(xyz), 3)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvwp[i][0] = 1.0
    return uvwp

def GetRealPhyInfo(time):
    use_real_data = False
    if use_real_data is True:
        xyzuvwp = np.load("csv/flow_re20_" + str(time) + "_xyzuvwp.npy") 
    else:
        xyzuvwp = np.ones((1000, 7)).astype(np.float32)
    return xyzuvwp


def init_pde_and_algo():
    circle_center = (0.0, 0.0)
    circle_radius = 0.5
    geo = psci.geometry.CylinderInCube(
        origin=(-8, -8, -0.5),
        extent=(25, 8, 0.5),
        circle_center=circle_center,
        circle_radius=circle_radius)

    geo.add_boundary(name="top", criteria=lambda x, y, z: z == 0.5)
    geo.add_boundary(name="down", criteria=lambda x, y, z: z == -0.5)
    geo.add_boundary(name="left", criteria=lambda x, y, z: x == -8)
    geo.add_boundary(name="right", criteria=lambda x, y, z: x == 25)
    geo.add_boundary(name="front", criteria=lambda x, y, z: y == -8)
    geo.add_boundary(name="back", criteria=lambda x, y, z: y == 8)
    geo.add_boundary(
        name="circle",
        criteria=lambda x, y, z: (x - circle_center[0])**2 + (y - circle_center[1])**2 == circle_radius**2
    )

    # N-S
    pde = psci.pde.NavierStokes(nu=0.05, rho=1.0, dim=3, time_dependent=True)

    # set bounday condition
    bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
    bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_top_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_down_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_right_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_front_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_front_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_front_w = psci.bc.Dirichlet('w', rhs=0.0)

    bc_back_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_back_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_back_w = psci.bc.Dirichlet('w', rhs=0.0)

    # TODO 3. circle boundry
    bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
    bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
    bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

    pde.add_geometry(geo)

    # add bounday and boundary condition
    pde.add_bc("top", bc_top_u, bc_top_v, bc_top_w)
    pde.add_bc("down", bc_down_u, bc_down_v, bc_down_w)
    pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
    pde.add_bc("right", bc_right_u, bc_right_v, bc_right_w)
    pde.add_bc("front", bc_front_u, bc_front_v, bc_front_w)
    pde.add_bc("back", bc_back_u, bc_back_v, bc_back_w)
    pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

    # Discretization
    pde_disc = psci.discretize(
        pde,
        time_method="implicit",
        time_step=0.5,
        space_npoints=60000,
        space_method="sampling")

    # Get real data
    real_xyzuvwp = GetRealPhyInfo(0.5)
    real_xyz = real_xyzuvwp[:, 0:3]
    real_uvwp = real_xyzuvwp[:, 3:7]

    # load real physic data in geo
    pde_disc.geometry.data = real_xyz

    # Network
    # TODO: remove num_ins and num_outs
    net = psci.network.FCNet(
        num_ins=3,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        activation='tanh')

    # Loss, TO rename
    # bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
    loss = psci.loss.L2()

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)
    return net, pde_disc, algo

def transform_program():
    net, pde_disc, algo = init_pde_and_algo()

    inputs_data, inputs_attr = algo.create_inputs(pde_disc)
    labels_data, labels_attr = algo.create_labels(pde_disc)

    input_name = ['input0', 'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8']
    output_name = ['elementwise_add_9', 'elementwise_add_19', 'elementwise_add_29', 'elementwise_add_39',
                   'elementwise_add_49', 'elementwise_add_59', 'elementwise_add_69', 'elementwise_add_79', 'elementwise_add_89']
    label_name = ['label0', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6']

    main_program = paddle.load('/workspace/PaddleScience/cylinder3d_unsteady_without_loss_main_program.pdmodel')
    startup_program = paddle.load('/workspace/PaddleScience/cylinder3d_unsteady_without_loss_startup_program.pdmodel')


    with paddle.static.program_guard(main_program, startup_program):

        inputs = []
        for name in input_name:
            inputs.append(main_program.block(0).var(name))
        outputs = []
        for name in output_name:
            outputs.append(main_program.block(0).var(name))
        labels = []
        for name in label_name:
            labels.append(main_program.block(0).var(name))

        # bc loss
        bc_loss = 0.0
        # top
        out_i = outputs[1]
        bc_loss += paddle.norm((out_i[:, 0]-1.0)*(out_i[:, 0]-1.0)*1.0, p=1)
        # import pdb; pdb.set_trace()
        bc_loss += paddle.norm((out_i[:, 1]-0.0)*(out_i[:, 1]-0.0)*1.0, p=1)
        bc_loss += paddle.norm((out_i[:, 2]-0.0)*(out_i[:, 2]-0.0)*1.0, p=1)
        # down, left, right, front, back, circle
        for i in range(2, 8):
            out_i = outputs[i]
            # u, v, w
            for j in range(3):
                bc_loss += paddle.norm((out_i[:, j]-0.0)*(out_i[:, j]-0.0)*1.0, p=1)

        # eq loss
        input_i = inputs[0] # (51982, 3)
        out_i = outputs[0] # (51982, 4)
        x = input_i[:, 0]
        y = input_i[:, 1]
        z = input_i[:, 2]
        u = out_i[:, 0]
        v = out_i[:, 1]
        w = out_i[:, 2]
        p = out_i[:, 3]
        u_n = labels[0]
        v_n = labels[1]
        w_n = labels[2]

        jac0, = paddle.static.gradients([u], [input_i]) # du/dx, du/dy, du/dz
        jac1, = paddle.static.gradients([v], [input_i]) # dv/dx, dv/dy, dv/dz
        jac2, = paddle.static.gradients([w], [input_i]) # dw/dx, dw/dy, dw/dz
        jac3, = paddle.static.gradients([p], [input_i]) # dp/dx, dp/dy, dp/dz

        hes0, = paddle.static.gradients([jac0[:, 0]], [input_i]) # du*du/dx*dx, du*du/dx*dy, du*du/dx*dz
        hes1, = paddle.static.gradients([jac0[:, 1]], [input_i]) # du*du/dy*dx, du*du/dy*dy, du*du/dy*dz
        hes2, = paddle.static.gradients([jac0[:, 2]], [input_i]) # du*du/dz*dx, du*du/dz*dy, du*du/dz*dz
        hes3, = paddle.static.gradients([jac1[:, 0]], [input_i]) # dv*dv/dx*dx, dv*dv/dx*dy, dv*dv/dx*dz
        hes4, = paddle.static.gradients([jac1[:, 1]], [input_i]) # dv*dv/dy*dx, dv*dv/dy*dy, dv*dv/dy*dz
        hes5, = paddle.static.gradients([jac1[:, 2]], [input_i]) # dv*dv/dz*dx, dv*dv/dz*dy, dv*dv/dz*dz
        hes6, = paddle.static.gradients([jac2[:, 0]], [input_i]) # dw*dw/dx*dx, dw*dw/dx*dy, dw*dw/dx*dz
        hes7, = paddle.static.gradients([jac2[:, 1]], [input_i]) # dw*dw/dy*dx, dw*dw/dy*dy, dw*dw/dy*dz
        hes8, = paddle.static.gradients([jac2[:, 2]], [input_i]) # dw*dw/dz*dx, dw*dw/dz*dy, dw*dw/dz*dz

        nu = 0.05
        rho = 1.0
        continuty = jac0[:, 0] + jac1[:, 1] + jac2[:, 2]
        # + 2.0*u(x, y, z) - 2.0*u_n(x, y, z)
        momentum_x = 2.0 * u - 2.0 * u_n + u * jac0[:, 0] + v * jac0[:, 1] + w * jac0[:, 2] - \
                    nu / rho * hes0[:, 0] - nu / rho * hes1[:, 1] - nu / rho * hes2[:, 2] + \
                    1.0 / rho * jac3[:, 0]
        momentum_y = 2.0 * v - 2.0 * v_n + u * jac1[:, 0] + v * jac1[:, 1] + w * jac1[:, 2] - \
                    nu / rho * hes3[:, 0] - nu / rho * hes4[:, 1] - nu / rho * hes5[:, 2] + \
                    1.0 / rho * jac3[:, 1]
        momentum_z = 2.0 * w - 2.0 * w_n + u * jac2[:, 0] + v * jac2[:, 1] + w * jac2[:, 2] - \
                    nu / rho * hes6[:, 0] - nu / rho * hes7[:, 1] - nu / rho * hes8[:, 2] + \
                    1.0 / rho * jac3[:, 2]
        eq_loss = paddle.norm(continuty, p=2)*paddle.norm(continuty, p=2) + \
                  paddle.norm(momentum_x, p=2)*paddle.norm(momentum_x, p=2) + \
                  paddle.norm(momentum_y, p=2)*paddle.norm(momentum_y, p=2) + \
                  paddle.norm(momentum_z, p=2)*paddle.norm(momentum_z, p=2)

        # data_loss
        input_i = inputs[8]
        out_i = outputs[8]
        label3 = labels[3]
        label4 = labels[4]
        label5 = labels[5]
        data_loss = paddle.norm(out_i[:, 0] - label3, p=2) + \
                    paddle.norm(out_i[:, 1] - label4, p=2) + \
                    paddle.norm(out_i[:, 2] - label5, p=2)

        # total_loss
        total_loss = paddle.sqrt(bc_loss) + paddle.sqrt(eq_loss) + paddle.sqrt(data_loss)

        paddle.fluid.optimizer.AdamOptimizer(0.001).minimize(total_loss)
        if prim_enabled():
            prim2orig(main_program.block(0))

    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    feeds = dict()
    for i in range(len(inputs_data)):
        feeds['input' + str(i)] = inputs_data[i]
    
    real_xyzuvwp = GetRealPhyInfo(0.5)
    real_xyz = real_xyzuvwp[:, 0:3]
    real_uvwp = real_xyzuvwp[:, 3:7]
    uvw = GenInitPhyInfo(pde_disc.geometry.interior)
    self_lables = algo.feed_labels_data_n(labels=labels_data, labels_attr=labels_attr, data_n=uvw)
    self_lables = algo.feed_labels_data(labels=self_lables, labels_attr=labels_attr, data=real_uvwp)
    for i in range(len(self_lables)):
        feeds['label' + str(i)] = self_lables[i]
    
    fetchs = [bc_loss.name, eq_loss.name, data_loss.name, total_loss.name]
    for i in range(10):
        out = exe.run(main_program,
                    feed=feeds, 
                    fetch_list=fetchs)
        print('epoch: {}, bc_loss: {}, eq_loss: {}, data_loss: {}, total_loss: {}'.format(str(i), out[-4], out[-3], out[-2], out[-1]))
if __name__ == '__main__':
    transform_program()