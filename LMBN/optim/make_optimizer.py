import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from .n_adam import NAdam
from .warmup_cosine_scheduler import WarmupCosineAnnealingLR


def make_optimizer(model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    # if args.model in ['PCB', 'PCB_v', 'PCB_conv']:
    #     ignored_params = []
    #     for i in range(6):
    #         name = 'classifier' + str(i)
    #         c = getattr(model, name)
    #         ignored_params = ignored_params + list(map(id, c.parameters()))
    #
    #     ignored_params = tuple(ignored_params)
    #
    #     base_params = filter(lambda p: id(
    #         p) not in ignored_params, model.model.parameters())
    #
    #     params = []
    #     for i in range(6):
    #         name = 'classifier' + str(i)
    #         c = getattr(model.model, name)
    #         params.append({'params': c.parameters(), 'lr': 0.0006})
    #     params = [{'params': base_params,
    #                'lr': 0.1 * 0.0006}] + params
    #
    #     optimizer_pcb = optim.Adam(params, weight_decay=5e-4)
    #
    #     return optimizer_pcb

    optimizer_function = optim.Adam
    kwargs = {
        'betas': (0.9, 0.999),
        'eps': 1.0e-08,
        'amsgrad': False
    } # amsgrad = false and not False in the config file

    # [{"lr": 0.0006, "weight_decay": 0.0005, "eps":1.0e-08, "betas":(0.9, 0.999), "amsgrad":False}]

    kwargs['lr'] = 0.0006
    kwargs['weight_decay'] = 0.0005

    # return optimizer_function(trainable, **kwargs)
    return optimizer_function(trainable, **{"lr": 0.0006, "weight_decay": 0.0005, "eps":1.0e-08, "betas":(0.9, 0.999), "amsgrad":False})


def make_scheduler(optimizer, last_epoch):

    # if args.warmup in ['linear', 'constant'] and args.load == '' and args.pre_train == '':
    milestones = "step_50_80_110".split('_')
    milestones.pop(0)
    milestones = list(map(lambda x: int(x), milestones))


    scheduler = WarmupCosineAnnealingLR(
        optimizer, multiplier=1, warmup_epoch=10, min_lr= 0.0006 / 1000, epochs=140, last_epoch=last_epoch)

    return scheduler

