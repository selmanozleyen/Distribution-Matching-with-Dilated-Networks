import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn
import numpy as np


class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.output_size = c_size//stride
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood = self.cood.unsqueeze(0)   # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1  # map to [-1, 1]
        X, Y = np.meshgrid(self.cood.cpu(), self.cood.cpu())
        points = torch.from_numpy(np.column_stack((Y.ravel(), X.ravel()))).cuda()
        if self.norm_cood:
            points = points / self.c_size * 2 - 1  # map to [-1, 1]
        x = points[:, 0].unsqueeze(1)  # [N, 1]
        y = points[:, 1].unsqueeze(1)
        x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood*self.cood  # [#gt, #cood]
        y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood*self.cood
        y_dis.unsqueeze_(2)
        x_dis.unsqueeze_(1)
        dis = y_dis + x_dis
        self.dis = dis.view((dis.size(0), -1))  # size of [#gt, #cood * #cood]
        self.softmax = torch.nn.Softmin(dim=0)

    def forward(self, normed_density, unnormed_density, points, gt_discrete):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0  # wasserstain distance
        for idx, im_points in enumerate(points):
            source_prob = self.softmax(unnormed_density[idx][0].view([-1]).detach())
            target_prob = self.softmax(gt_discrete[idx].view([-1])).to(self.device)
            # use sinkhorn to solve OT, compute optimal beta.
            P, log = sinkhorn(target_prob, source_prob, self.dis, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
            beta = log['beta']  # size is the same as source_prob: [#cood * #cood]
            ot_obj_values += torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
            # compute the gradient of OT loss to predicted density (unnormed_density).
            # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
            # source_density = unnormed_density[idx][0].view([-1]).detach()
            # source_count = source_density.sum()
            # im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta  # size of [#cood * #cood]
            # im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)  # size of 1
            # im_grad = im_grad_1 - im_grad_2
            im_grad = ((-source_prob) * (1 + source_prob)) * beta
            im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
            # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
            loss += torch.sum(unnormed_density[idx] * im_grad)
            wd += torch.sum(self.dis * P).item()
        return loss, wd, ot_obj_values


