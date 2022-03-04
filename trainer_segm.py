import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from pytorch_utils.base_trainer import BaseTrainer
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
from datasets.dataloader import HabitatDataImgSegm
from sklearn.metrics import confusion_matrix
import metrics


class TrainerSegm(BaseTrainer):
    """ Implements training for prediction models
    """
    def init_fn(self):
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.train_ds = HabitatDataImgSegm(self.options, config_file=self.options.config_train_file)
        self.test_ds = HabitatDataImgSegm(self.options, config_file=self.options.config_val_file)

        self.img_segmentor = get_img_segmentor_from_options(self.options)

        self.models_dict = {'img_segm_model':self.img_segmentor}

        print("Using ", torch.cuda.device_count(), "gpus")
        for k in self.models_dict:
            self.models_dict[k] = nn.DataParallel(self.models_dict[k])

        self.optimizers_dict = {}
        for model in self.models_dict:
            self.optimizers_dict[model] = \
                    torch.optim.Adam([{'params':self.models_dict[model].parameters(),
                                    'initial_lr':self.options.lr}],
                                    lr=self.options.lr,
                                    betas=(self.options.beta1, 0.999) )


    def train_step(self, input_batch, step_count):
        for model in self.models_dict:
            self.models_dict[model].train()
        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].zero_grad()

        pred_output = self.models_dict['img_segm_model'](input_batch)

        loss_output = self.models_dict['img_segm_model'].module.loss_cel(input_batch, pred_output)

        pred_segm_loss = loss_output['pred_segm_loss']
        pred_segm_loss.sum().backward(retain_graph=True)
        self.optimizers_dict['img_segm_model'].step()

        output = {}
        output['segm'] = {'pred_segm':pred_output['pred_segm'].detach()}
        output['metrics'] = {'pred_segm_err': loss_output['pred_segm_err']}
        output['losses'] = {'pred_segm_loss': pred_segm_loss}

        for k in output['metrics']:
            output['metrics'][k] = torch.mean(output['metrics'][k])
        for k in output['losses']:
            output['losses'][k] = torch.mean(output['losses'][k])

        return [output]


    def train_summaries(self, input_batch, save_images, model_output):
        self._save_summaries(input_batch, model_output, save_images, is_train=True)



    def test(self):
        for model in self.models_dict:
            self.models_dict[model].eval()

        test_data_loader = DataLoader(self.test_ds,
                                      batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        batch = None

        self.options.test_iters = len(test_data_loader) # the length of dataloader depends on the batch size
        labels = list(range(self.options.n_object_classes))
        overall_confusion_matrix = None
        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing',
                                           total=self.options.test_iters)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():

                pred_output = self.models_dict['img_segm_model'](batch)

                loss_output = self.models_dict['img_segm_model'].module.loss_cel(batch, pred_output)

                pred_segm = pred_output['pred_segm']

                # Decide label for each location based on predition probs
                pred_labels = torch.argmax(pred_segm.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

                gt_segm = batch['gt_segm'].cpu() #.numpy() # B x T x 1 x cH x cW

                current_confusion_matrix = confusion_matrix(y_true=gt_segm.flatten(), y_pred=pred_labels.flatten(), labels=labels)
                current_confusion_matrix = torch.tensor(current_confusion_matrix)

                if overall_confusion_matrix is None:
                    overall_confusion_matrix = current_confusion_matrix
                else:
                    overall_confusion_matrix += current_confusion_matrix

                # Stop testing if test iterations has been exceeded
                if tstep > self.options.test_iters:
                    break

        mAcc_obj = metrics.overall_pixel_accuracy(overall_confusion_matrix)
        class_mAcc_obj, _ = metrics.per_class_pixel_accuracy(overall_confusion_matrix)
        mIoU_obj, _ = metrics.jaccard_index(overall_confusion_matrix)
        mF1_obj, _ = metrics.F1_Score(overall_confusion_matrix)

        output = {}
        output['metrics'] = {'overall_pixel_accuracy':mAcc_obj,
                             'per_class_pixel_accuracy':class_mAcc_obj,
                             'mean_interesction_over_union':mIoU_obj,
                             'mean_f1_score':mF1_obj}
        output['losses'] = {'pred_segm_err': loss_output['pred_segm_err']}
        output['segm'] = {'pred_segm':pred_segm}

        for k in output['metrics']:
            output['metrics'][k] = torch.mean(output['metrics'][k])
        for k in output['losses']:
            output['losses'][k] = torch.mean(output['losses'][k])


        self._save_summaries(batch, output, save_images=True, is_train=False)



    def _save_summaries(self, batch, output, save_images, is_train=False):
        prefix = 'train/' if is_train else 'test/'

        for scalar_type in ['losses', 'metrics']:
            for k in output[scalar_type]:
                self.summary_writer.add_scalar(prefix + k, output[scalar_type][k], self.step_count)

        if is_train:
            self.summary_writer.add_scalar(prefix + "lr", self.get_lr(), self.step_count)


        if save_images:
            # input imgs
            self.summary_writer.add_video(prefix+"gifs/input_imgs", batch['images'], self.step_count, fps=0.25)

            # predicted segm sem
            color_pred_segm = viz_utils.colorize_grid(output['segm']['pred_segm'], color_mapping=27)
            self.summary_writer.add_video(prefix+"gifs/pred_segm", color_pred_segm, self.step_count, fps=0.25)

            # gt segm sem
            color_gt_segm = viz_utils.colorize_grid(batch['gt_segm'], color_mapping=27)
            self.summary_writer.add_video(prefix+"gifs/gt_segm", color_gt_segm, self.step_count, fps=0.25)
    