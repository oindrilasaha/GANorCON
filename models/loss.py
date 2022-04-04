import torch.nn.functional as F
import time
import torch


def regression_loss(prediction_normalized, kp_normalized, alpha=10., **kwargs):
    kp = kp_normalized.to(prediction_normalized.device)
    B, nA, _ = prediction_normalized.shape
    return F.smooth_l1_loss(prediction_normalized * alpha, kp * alpha)


def selected_regression_loss(prediction_normalized, kp_normalized, visible, alpha=10., **kwargs):
    kp = kp_normalized.to(prediction_normalized.device)
    B, nA, _ = prediction_normalized.shape
    for i in range(B):
        vis = visible[i]
        invis = [not v for v in vis]
        kp[i][invis] = 0.
        prediction_normalized[i][invis] = 0.

    return F.smooth_l1_loss(prediction_normalized * alpha, kp * alpha)

def cross_entropy2d(input, target, weight=None, size_average=True):
    # import pdb;pdb.set_trace()
    # input = input.permute(0,2,1).reshape(-1,34,512,512)
    n, c, h, w = input.size()
    target = target.squeeze(1).long()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        print('whys the sizes different')
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss
