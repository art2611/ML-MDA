import torch.nn.functional as F
from .loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .loss.triplet_loss import TripletLoss
from .loss.center_loss import CenterLoss

def make_loss(num_classes):

    # feat_dim = 2048
    feat_dim = 512

    # center_criterion = CenterLoss(num_classes=num_classes,
    #                               feat_dim=feat_dim,
    #                               use_gpu=True)  # center losses

    triplet = TripletLoss()  # No margin
    print("using soft triplet losses for training")

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target): # Label smooth "on" + softmax triplet setting
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [
                triplet(feats, target)[0] for feats in feat[1:]
            ]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(
                feat[0], target)[0]
        else:
            TRI_LOSS = triplet(feat, target)[0]
        return ID_LOSS, TRI_LOSS


    return loss_func #, center_criterion