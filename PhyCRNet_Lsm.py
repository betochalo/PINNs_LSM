import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import torch.nn.functional as F
import os
from torch.nn.utils import weight_norm

torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1 / 12, 0, 0],
               [0, 0, -8 / 12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8 / 12, 0, 0],
               [0, 0, -1 / 12, 0, 0]]]]


# def init_hidden_tensor(prev_state):
#     return prev_state[0], prev_state[1]


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        c = 0.1
        module.weight.data.uniform_(-c * np.sqrt(1 / np.prod(module.weight.shape[:-1])),
                                    c * np.sqrt(1 / np.prod(module.weight.shape[:-1])))

    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM"""

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state

        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias=True, padding_mode='circular'
        )

        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias=False, padding_mode='circular'
        )

        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias=True, padding_mode='circular'
        )

        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias=False, padding_mode='circular'
        )

        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias=True, padding_mode='circular'
        )

        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias=False, padding_mode='circular'
        )

        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias=True, padding_mode='circular'
        )

        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias=False, padding_mode='circular'
        )

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return prev_state[0], prev_state[1]


class EncoderBlock(nn.Module):
    """encoder with CNN"""

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(EncoderBlock, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels,
                                          self.hidden_channels, self.input_kernel_size,
                                          self.input_stride,
                                          self.input_padding, bias=True, padding_mode='circular'))

        self.act = nn.ReLU()
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class PhyCRNetLSM(nn.Module):
    """ Physics-informed convolutional-recurrent neural networks """

    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding, dt,
                 num_layers, upscale_factor, step=1, ef_step=None):

        super(PhyCRNetLSM, self).__init__()
        self.init_state = None
        if ef_step is None:
            ef_step = [1]

        # Input channels of layer includes input_channels and hidden channels
        self.input_channels = [input_channels] + hidden_channels  # [1, 8, 32, 128, 128]
        self.hidden_channels = hidden_channels  # [8, 32, 128, 128]
        self.input_kernel_size = input_kernel_size  # [4, 4, 4, 3]
        self.input_stride = input_stride  # [2, 2, 2, 1]
        self.input_padding = input_padding  # [1, 1, 1, 1]
        self.step = step  # 314
        self.ef_step = ef_step  # [0, 1, 2, ..., 313]
        self._all_layers = []  # Lista Vacía
        self.dt = dt  # 0.5
        self.upscale_factor = upscale_factor  # 8

        # number of layers [3, 1]
        self.num_encoder = num_layers[0]  # 3
        self.num_convlstm = num_layers[1]  # 1

        # encoder - downsampling
        for n in range(self.num_encoder):  # n = 1, n = 2, n = 3
            name = 'encoder{}'.format(n)
            cell = EncoderBlock(
                input_channels=self.input_channels[n],
                hidden_channels=self.hidden_channels[n],
                input_kernel_size=self.input_kernel_size[n],
                input_stride=self.input_stride[n],
                input_padding=self.input_padding[n]
            )

            setattr(self, name, cell)
            self._all_layers.append(cell)

        # ConvLSTM
        for n in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(n)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[n],
                hidden_channels=self.hidden_channels[n],
                input_kernel_size=self.input_kernel_size[n],
                input_stride=self.input_stride[n],
                input_padding=self.input_padding[n]
            )

            setattr(self, name, cell)
            self._all_layers.append(cell)

        # Output layer
        self.output_layer = nn.Conv2d(1, 1, kernel_size=5, stride=1,
                                      padding=2, padding_mode='circular')

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        # Initializa weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, init_state, x):
        self.init_state = init_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            # Enconder
            for n in range(self.num_encoder):
                name = 'encoder{}'.format(n)
                x = getattr(self, name)(x)

            # convlstm
            for n in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(n)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.init_state[n - self.num_encoder])
                    internal_state.append((h, c))

                # One-step forward
                (h, c) = internal_state[n - self.num_encoder]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[n - self.num_encoder] = (x, new_c)

            # Ouput
            x = self.pixelshuffle(x)
            x = self.output_layer(x)

            # residual conncection

            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.ef_step:
                outputs.append(x)

        return outputs, second_last_state


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, x):
        padding_derivative = [self.padding, self.padding, self.padding, self.padding]
        input_padded = F.pad(x, padding_derivative, mode='reflect')
        derivative = self.filter(input_padded)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$ constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    """Loss generator for the physics loss"""

    def __init__(self, dt1=(10.0 / 200), dx1=(20.0 / 64)):
        """Construct the derivative, X = Width, Y = Height"""

        super(loss_generator, self).__init__()

        self.dx = Conv2dDerivative(
            DerFilter=partial_x,
            resol=(dx1 * 1),
            kernel_size=5,
            name='dx_operator'
        )

        self.dy = Conv2dDerivative(
            DerFilter=partial_y,
            resol=(dx1 * 1),
            kernel_size=5,
            name='dy_operator'
        )

        # Temporal derivative operator

        self.dt = Conv1dDerivative(
            DerFilter=[[[-1, 0, 1]]],
            resol=(dt1 * 2),
            kernel_size=3,
            name='partial_t'
        )

    def get_phy_loss(self, output, u, v, phi0):
        # spatial derivatives
        phi_x = self.dx(output[1:-1, 0:1, :, :])
        phi_y = self.dy(output[1:-1, 0:1, :, :])

        # temporal derivative - phi
        phi = output
        lent = phi.shape[0]
        lenx = phi.shape[3]
        leny = phi.shape[2]
        phi_conv1d = phi.permute(2, 3, 1, 0)
        phi_conv1d = phi_conv1d.reshape(lenx * leny, 1, lent)
        phi_t = self.dt(phi_conv1d)
        phi_t = phi_t.reshape(leny, lenx, 1, lent - 2)
        phi_t = phi_t.permute(3, 2, 0, 1)

        # phi = output[1:-1, 0:1, :, :]
        # sgn = torch.sgn(phi0)

        # assert phi_t.shape == phi_t.shape
        assert phi_x.shape == phi_t.shape
        assert phi_y.shape == phi_t.shape
        assert phi_y.shape == phi_x.shape

        norm_grad = torch.sqrt(phi_x**2 + phi_y**2)

        f_phi = phi_t + torch.mul(u, phi_x) + torch.mul(v, phi_y)
        # f_phit = phi_t + sgn * (1 - norm_grad)

        return f_phi, norm_grad #, f_phit


def compute_loss(output, loss_func, u, v, phi0):
    """Calculate the physics loss"""
    mse_loss = nn.MSELoss()
    f_phi, norm = loss_func.get_phy_loss(output, u, v, phi0)
    # Transformation
    # phi_int = (phi <= 0).float()

    loss = mse_loss(f_phi, torch.zeros_like(f_phi)) #+ mse_loss(norm, torch.ones_like(norm)) # + mse_loss(f_phit, torch.zeros_like(f_phit))#+ mse_loss(phi, sol)  # Impose

    return loss


def train(model, input, initial_state, n_iters, time_batch_size, learning_rate,
          dt, dx, save_path, pre_model_save_path, num_time_batch, u, v):
    state_detached = None
    train_loss_list = []
    second_last_state = []
    prev_output = []

    # batch_loss = 0.0
    best_loss = 8e-2

    # load previous model

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)
    # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, pre_model_save_path)  # model,
    # optimizer, scheduler, save_dir

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = loss_generator(dt, dx)

    for epoch in range(n_iters):
        optimizer.zero_grad()
        batch_loss = 0

        for time_batch_id in range(num_time_batch):
            if time_batch_id == 0:
                hidden_state = initial_state
                phi_0 = input
            else:
                hidden_state = state_detached
                phi_0 = prev_output[-2:-1].detach()

            # output is a list
            output, second_last_state = model(hidden_state, phi_0)

            output = torch.cat(tuple(output), dim=0)

            output = torch.cat((phi_0, output), dim=0)

            # get loss
            loss = compute_loss(output, loss_func, u, v, input)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            # Update the state and output for next batch4
            prev_output = output
            state_detached = []
            for n in range(len(second_last_state)):
                (h, c) = second_last_state[n]
                state_detached.append((h.detach(), c.detach()))

        optimizer.step()
        scheduler.step()

        # Print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0),
                                            batch_loss))

        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def post_process(output, true, axis_lim, phi_lim, num, save_path):
    # get the limit
    xmin, xmax, ymin, ymax = axis_lim
    phi_min, phi_max = phi_lim

    # grid
    x = np.linspace(xmin, xmax, 64 + 1)
    x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)

    phi_star = true[num, 0, :, :]
    phi_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0, 0].scatter(x_star, y_star, c=phi_pred, alpha=0.9, edgecolors='none',
                          cmap='RdYlBu', marker='s', s=4, vmin=phi_min, vmax=phi_max)

    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set.title('phi-RCNN')
    fig.colorbar(cf, ax=ax[0, 0])

    cf = ax[0, 1].scatter(x_star, y_star, c=phi_star, alpha=0.9, edgecolors='none',
                          cmap='RdYlBu', marker='s', s=4, vmin=phi_min, vmax=phi_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title('phi-Ref.')
    fig.colorbar(cf, ax=ax[0, 1])

    # plt.draw()
    plt.savefig(save_path + 'uv_comparison_' + str(num).zfill(3) + '.png')
    plt.close('all')
    return phi_star, phi_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    """Save model and optimizer"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(m, optimizer, scheduler, save_dir):
    """Load model and optimizer"""
    checkpoint = torch.load(save_dir)
    m.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model load!')
    return m, optimizer, scheduler


def summary_parameters(model):
    for n in model.parameters():
        print(n.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))


if __name__ == '__main__':

    data_dir = './Data/LSM_data/LSM_1256x1x64x64.mat'
    data_uv = './Data/LSM_data/UV_1256x2x64x64.mat'
    data = scipy.io.loadmat(data_dir)
    datauv = scipy.io.loadmat(data_uv)
    phi = data['phi']
    uv = datauv['uv']

    # initial condition
    phi0 = phi[0:1, ...]
    inp = torch.tensor(phi0, dtype=torch.float32)
    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8))
    # (h0, c0) = (torch.zeros(1, 64, 8, 8), torch.zeros(1, 64, 8, 8))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))

    # grid parameters
    time_steps = 10
    dt = 0.5
    dx = 1.0 / 64

    # Build the model
    time_batch_size = 9
    # U V
    u1 = uv[0:time_batch_size, 0:1, :, :]
    v1 = uv[0:time_batch_size, 1:2, :, :]
    # Condition interface
    sol = phi[1:time_steps, 0:1, :, :]
    interface = (sol <= 0).astype(float)
    sol_loss = torch.tensor(sol, dtype=torch.float32)
    inter_cond = torch.tensor(interface, dtype=torch.float32)
    # Convert tensors
    u = torch.tensor(u1, dtype=torch.float32)
    v = torch.tensor(v1, dtype=torch.float32)
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 2000
    lr_adam = 1e-4
    pre_model_save_path = './Weights/3_train/parameters.pt'
    model_save_path = './Weights/3_train/parameters3.pt'
    fig_save_path = './img/'

    # model = PhyCRNetLSM(
    #     input_channels=1,
    #     hidden_channels=[8, 32, 64, 64],
    #     input_kernel_size=[4, 4, 4, 3],
    #     input_stride=[2, 2, 2, 1],
    #     input_padding=[1, 1, 1, 1],
    #     dt=dt,
    #     num_layers=[3, 1],
    #     upscale_factor=8,
    #     step=steps,
    #     ef_step=effective_step)
    #
    # start = time.time()
    # train_loss = train(model, inp, initial_state, n_iters_adam, time_batch_size,
    #                    lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch, u, v)
    # end = time.time()
    #
    # np.save('./Weights/3_train/train_loss1', train_loss)
    # print('The training time is:', (end - start))

    ########### model inference ##################
    time_batch_size_load = 99
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))
    model = PhyCRNetLSM(
        input_channels=1,
        hidden_channels=[8, 32, 64, 64],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps_load,
        ef_step=effective_step)

    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path)
    output, _ = model(initial_state, inp)

    output = torch.cat(tuple(output), dim=0)
    output = torch.cat((inp, output), dim=0)

    truth = phi[0:100, 0, :, :]
    phi_pred = output[:, 0, :, :].detach().cpu().numpy()
    phi1 = phi_pred[20, :, :]
    phi2 = truth[20, :, :]
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    nodes = 64

    x = np.linspace(x_min, x_max, nodes)
    y = np.linspace(y_min, y_max, nodes)

    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(5, 5))
    plt.contourf(X, Y, phi1, levels=[-1e6, 0], colors='yellow')
    plt.contour(X, Y, phi1, levels=[0], colors='black')
    plt.contourf(X, Y, phi2, levels=[-1e6, 0], colors='red')
    plt.contour(X, Y, phi2, levels=[0], colors='black')
    plt.savefig('./Weights/2_train/img2')
    plt.show()

    # Post process
    # true = []
    # pred = []
    #
    # for i in range(0, 50):
    #     phi_star, phi_pred = post_process(output, truth, [0, 1, 0, 1],
    #                                       )

