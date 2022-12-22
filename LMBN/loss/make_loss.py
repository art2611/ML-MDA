import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from LMBN.loss.triplet import CrossEntropyLabelSmooth
from LMBN.loss.multi_similarity_loss import MultiSimilarityLoss


class LossFunction():
    def __init__(self, num_classes):
        super(LossFunction, self).__init__()

        self.nGPU = 1
        self.loss = []
        for loss in "0.5*CrossEntropy+0.5*MSLoss".split('+'): # Crossentropy MSLoss
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = CrossEntropyLabelSmooth(
                    num_classes=num_classes)
            elif loss_type == 'MSLoss':
                loss_function = MultiSimilarityLoss(margin=0.7)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        # print(f"len(self.loss) : {len(self.loss)} - make_loss (LMBN)")
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        self.log = torch.Tensor()

    def compute(self, outputs, labels):
        losses = []

        for i, l in enumerate(self.loss):
            if l['type'] in ['CrossEntropy']:

                if isinstance(outputs[0], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[0]]
                elif isinstance(outputs[0], torch.Tensor):
                    loss = [l['function'](outputs[0], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[0])))

                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

                # self.log[-1, i] += effective_loss.item()

            elif l['type'] in ['MSLoss']:
                if isinstance(outputs[-1], list):
                    loss = [l['function'](output, labels)
                            for output in outputs[-1]]
                elif isinstance(outputs[-1], torch.Tensor):
                    loss = [l['function'](outputs[-1], labels)]
                else:
                    raise TypeError(
                        'Unexpected type: {}'.format(type(outputs[-1])))
                loss = sum(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()

            else:
                pass

        loss_sum = sum(losses)

        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.6f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)

            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    # Following codes not being used

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'losses.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'losses.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)):
                    l.scheduler.step()


def make_loss(num_classes):
    return LossFunction(num_classes)