import torch
from torch.optim.optimizer import Optimizer, required

__all__ = ['LARS']


class LARS(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0, eeta=0.001, epsilon=1e-5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon <= 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)

        return loss
