import torch
from torch.optim.optimizer import Optimizer, required

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_weight(weight, do_power_iteration=True, n_power_iterations=5):
    weight_mat = weight

    height = weight_mat.size(0)
    weight_mat = weight_mat.view(height, -1)

    u = torch.randn(1, weight_mat.shape[0]).to(device)  # move u to GPU
    for i in range(n_power_iterations):
        v = torch.mm(u, weight_mat)  # [1, n]
        v = v / torch.norm(v)
        u = torch.mm(v, weight_mat.T)  # [1, m]
        u = u / torch.norm(u)

    return torch.mul(
        (torch.mul(torch.ones(1, weight_mat.shape[0]).to(device), torch.sum(torch.mm(torch.mm(u, weight_mat), v.T)))).T,
        torch.mm(u.T, v))


class SGDSNR(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, 
                    momentum=0, dampening=0,
                    weight_decay=0, nesterov=False, 
                    spectrum_regularization=0, 
                    differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        spectrum_regularization=spectrum_regularization)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDSNR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDSNR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('spectrum_regularization', 0)
            

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            spectrum_regularization = group['spectrum_regularization']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                if spectrum_regularization != 0:
                    #print("spectrum_regularization", spectrum_regularization, p.dim())
                    if p.dim() > 1:
                        d_p = d_p.add(torch.reshape(compute_weight(p.to(device)), p.shape),
                                        alpha=spectrum_regularization)
                    else:
                        pass
                else:
                    pass

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
