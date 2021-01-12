# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from fastreid.modeling.losses.utils import euclidean_dist, concat_all_gather
from fastreid.modeling.losses.triplet_loss import hard_example_mining
from fastreid.utils import comm
import torch.nn.functional as F


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class MoCo(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=512, K=65536):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K

        self.m = 0.25
        self.s = 96

        self.anneal = 0.01
        self.num_id = 256 // 32

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_label", torch.zeros((1, K), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        # gather keys/targets before updating queue
        if comm.get_world_size() > 1:
            keys = concat_all_gather(keys)
            targets = concat_all_gather(targets)
        else:
            keys = keys.detach()
            targets = targets.detach()

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:, ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, feat_q, feat_k, targets):
        r"""
        Memory bank enqueue and compute metric loss
        Args:
            feat_q: model features
            feat_k: model_ema features
            targets: gt labels

        Returns:
        """
        # dequeue and enqueue
        # feat_k = nn.functional.normalize(feat_k, p=2, dim=1)
        self._dequeue_and_enqueue(feat_k, targets)
        # loss = self._smooth_ap(feat_q, targets)
        # loss = self._circle_loss(feat_q, targets)
        loss = self._triplet_loss(feat_q, targets)
        return loss

    def _triplet_loss(self, feat_q, targets):
        dist_mat = euclidean_dist(feat_q, self.queue.t())

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()

        sorted_mat_distance, positive_indices = torch.sort(dist_mat + (-9999999.) * (1 - is_pos), dim=1,
                                                           descending=True)
        dist_ap = sorted_mat_distance[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(dist_mat + 9999999. * is_pos, dim=1,
                                                           descending=False)
        dist_an = sorted_mat_distance[:, 0]

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

        return loss

    def _circle_loss(self, feat_q, targets):
        feat_q = nn.functional.normalize(feat_q, p=2, dim=1)

        sim_mat = torch.mm(feat_q, self.queue)

        N, M = sim_mat.size()

        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()
        same_indx = torch.eye(N, N, device=is_pos.device)
        remain_indx = torch.zeros(N, M-N, device=is_pos.device)
        same_indx = torch.cat((same_indx, remain_indx), dim=1)
        is_pos = is_pos - same_indx

        is_neg = targets.view(N, 1).expand(N, M).ne(self.queue_label.expand(N, M)).float()

        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss

    def _smooth_ap(self, embedding, targets):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        embedding = F.normalize(embedding, dim=1)

        # For distributed training, gather all features from different process.

        sim_dist = torch.matmul(embedding, self.queue)
        N, M = sim_dist.size()

        # Compute the mask which ignores the relevance score of the query to itself
        mask_indx = 1.0 - torch.eye(M, device=sim_dist.device)
        mask_indx = mask_indx.unsqueeze(dim=0).repeat(N, 1, 1)  # (N, M, M)

        # sim_dist -> N, 1, M -> N, M, N
        sim_dist_repeat = sim_dist.unsqueeze(dim=1).repeat(1, M, 1)  # (N, M, M)

        # Compute the difference matrix
        sim_diff = sim_dist_repeat - sim_dist_repeat.permute(0, 2, 1)  # (N, M, M)

        # Pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask_indx

        # Compute all the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1  # (N, N)

        pos_mask = targets.view(N, 1).expand(N, M).eq(self.queue_label.view(M, 1).expand(M, N).t()).float()  # (N, M)

        pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, M, 1)  # (N, M, M)

        # Compute positive rankings
        pos_sim_sg = sim_sg * pos_mask_repeat
        sim_pos_rk = torch.sum(pos_sim_sg, dim=-1) + 1  # (N, N)

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = 0
        group = N // self.num_id
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap += pos_divide / torch.sum(pos_mask[ind*group]) / N
        return 1 - ap
