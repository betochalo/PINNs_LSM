import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchsummary import summary


class NetLevelSetMethod2D(nn.Module):
    def __init__(self, lay):
        super(NetLevelSetMethod2D, self).__init__()
        self.net = nn.Sequential()
        for i in range(0, len(lay) - 1):
            self.net.add_module('Linear_layer_%d' % i, nn.Linear(lay[i], lay[i + 1]))
            if i < len(lay) - 2:
                self.net.add_module('Tanh_layer_%d' % i, nn.Tanh())

    def forward(self, x):
        return self.net(x)


class PhysicsInformedNN:
    def __init__(self, X_phi, X_f, vf, phi, layers):
        # Data
        self.x_phi = torch.tensor(X_phi[:, 0:1], requires_grad=True).float()
        self.y_phi = torch.tensor(X_phi[:, 1:2], requires_grad=True).float()
        self.t_phi = torch.tensor(X_phi[:, 2:3], requires_grad=True).float()
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float()
        self.y_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float()
        self.t_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float()
        self.u_phi = torch.tensor(vf[:, 0:1]).float()
        self.v_phi = torch.tensor(vf[:, 1:2]).float()
        self.phi = torch.tensor(phi).float()

        self.layers = layers

        self.dnn = NetLevelSetMethod2D(layers)

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.iter = 0

    def net_phi(self, x, y, t):
        phi = self.dnn(torch.cat([x, y, t], dim=1))
        return phi

    def net_f(self, x, y, t):
        phi = self.net_phi(x, y, t)

        phi_t = torch.autograd.grad(
            phi, t,
            grad_outputs=torch.ones_like(phi),
            retain_graph=True,
            create_graph=True
        )[0]

        phi_x = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            retain_graph=True,
            create_graph=True
        )[0]

        phi_y = torch.autograd.grad(
            phi, y,
            grad_outputs=torch.ones_like(phi),
            retain_graph=True,
            create_graph=True
        )[0]

        f = phi_t + self.u_phi * phi_x + self.v_phi * phi_y

        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        phi_pred = self.net_phi(self.x_phi, self.y_phi, self.t_phi)
        f_pred = self.net_f(self.x_f, self.y_f, self.t_f)
        loss_phi = torch.mean((self.phi - phi_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_phi + loss_f
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_phi: %.5e, Loss_f: %.5e' % (
                    self.iter, loss.item(), loss_phi.item(),
                    loss_f.item()
                )
            )
        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_func())


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        y = torch.tensor(X[:, 1:2], requires_grad=True).float()
        t = torch.tensor(X[:, 2:3], requires_grad=True).float()

        self.dnn.eval()
        phi = self.net_phi(x, y, t)
        f = self.net_f(x, y, t)
        phi = phi.detach().cpu().numpy()
        f = f.detach().cpu().numpy
        return phi, f


# layers = [3, 30, 30, 30, 1]
# model = NetLevelSetMethod2D(layers)
# print(model)
