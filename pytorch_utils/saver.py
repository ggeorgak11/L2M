from __future__ import division
import os
import datetime

import torch

class CheckpointSaver(object):

    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._get_latest_checkpoint()

    # check if a checkpoint exists in the current directory
    def exists_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)

    # save checkpoint
    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size, dataset_perm, total_step_count):
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(os.path.join(self.save_dir,
                                                           timestamp.strftime('%Y_%m_%d-%H_%M_%S') + '.pt'))
        checkpoint = {'models':{}, 'optimizers':{}}
        for model in models:
            checkpoint['models'][model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint['optimizers'][optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['dataset_perm'] = dataset_perm
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
        print('Saving checkpoint file [' + checkpoint_filename + ']')
        torch.save(checkpoint, checkpoint_filename) 

    # load a checkpoint
    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)
        print("model:", models.keys())
        print("checkpoint models", checkpoint['models'].keys())
        for model in models:
            if model in checkpoint['models']:
                models[model].load_state_dict(checkpoint['models'][model])
            else:
                raise Exception("Missing model in checkpoint: {}".format(model))
        for optimizer in optimizers:
            if optimizer in checkpoint['optimizers']:
                optimizers[optimizer].load_state_dict(checkpoint['optimizers'][optimizer])
            else:
                raise Exception("Missing optimizer in checkpoint: {}".format(optimizer))
        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                'dataset_perm': checkpoint['dataset_perm'],
                'total_step_count': checkpoint['total_step_count']}

    # get filename of latest checkpoint if it exists
    def _get_latest_checkpoint(self):
        checkpoint_list = [] 
        for dirpath, _, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]

