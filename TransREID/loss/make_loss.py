# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from CIL.loss.triplet_loss import TripletLoss


def make_loss():    # modified
    triplet = TripletLoss()

    def loss_func(score, feat, target):
        if isinstance(score, list):
            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
        else:
            ID_LOSS = F.cross_entropy(score, target)

        if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
        else:
                TRI_LOSS = triplet(feat, target)[0]

        return ID_LOSS, TRI_LOSS
    return loss_func