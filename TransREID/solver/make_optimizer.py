import torch

def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.008
        weight_decay = 1e-4
        if "bias" in key:
            lr = 0.008 * 2
            weight_decay = 1e-4

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)

    return optimizer