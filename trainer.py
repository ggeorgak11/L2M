import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_utils.base_trainer import BaseTrainer
from datasets.dataloader import HabitatDataOffline
from models.predictors import get_predictor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
from sklearn.metrics import confusion_matrix
import metrics


class Trainer(BaseTrainer):
    """ Implements training for prediction models
    """
    def init_fn(self):
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.train_ds = HabitatDataOffline(self.options, config_file=self.options.config_train_file, img_segm=self.options.with_img_segm)
        self.test_ds = HabitatDataOffline(self.options, config_file=self.options.config_val_file, img_segm=self.options.with_img_segm)

        self.predictor_model = get_predictor_from_options(self.options)

        # Init the weights from normal distr with mean=0, std=0.02
        if self.options.init_gaussian_weights:
            self.predictor_model.apply(self.weights_init)

        self.models_dict = {'predictor_model':self.predictor_model}

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


    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def train_step(self, input_batch, step_count):
        for model in self.models_dict:
            self.models_dict[model].train()
        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].zero_grad()

        ## Input batch should already contain pred_ego_crops_sseg (ground-projected semantic segmentation) ##

        ### Predict the semantic map crops
        pred_output = self.models_dict['predictor_model'](input_batch)

        loss_output = self.models_dict['predictor_model'].module.loss_cel(input_batch, pred_output)

        pred_map_loss_spatial = loss_output['pred_map_loss_spatial']
        pred_map_loss_objects = loss_output['pred_map_loss_objects']
        pred_map_loss = pred_map_loss_spatial + pred_map_loss_objects
        pred_map_loss.sum().backward(retain_graph=True)

        pred_maps_objects = pred_output['pred_maps_objects']
        pred_maps_spatial = pred_output['pred_maps_spatial']

        self.optimizers_dict['predictor_model'].step()

        output = {}
        output['maps'] = {'pred_maps_objects':pred_maps_objects.detach(),
                          'pred_maps_spatial':pred_maps_spatial.detach(),
                          }
        output['metrics'] = {'pred_map_err_objects': loss_output['pred_map_err_objects'],
                             'pred_map_err_spatial': loss_output['pred_map_err_spatial'],
                             }
        output['losses'] = {'pred_map_loss_objects': pred_map_loss_objects,
                            'pred_map_loss_spatial': pred_map_loss_spatial,
                            }

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
        object_labels = list(range(self.options.n_object_classes))
        spatial_labels = list(range(self.options.n_spatial_classes))
        overall_confusion_matrix_objects, overall_confusion_matrix_spatial = None, None
        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing',
                                           total=self.options.test_iters)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():

                pred_output = self.models_dict['predictor_model'](batch)

                loss_output = self.models_dict['predictor_model'].module.loss_cel(batch, pred_output)

                pred_maps_objects = pred_output['pred_maps_objects']
                pred_maps_spatial = pred_output['pred_maps_spatial']

                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW
                pred_labels_spatial = torch.argmax(pred_maps_spatial.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

                gt_crops_spatial = batch['gt_grid_crops_spatial'].cpu() #.numpy() # B x T x 1 x cH x cW
                gt_crops_objects = batch['gt_grid_crops_objects'].cpu() #.numpy() # B x T x 1 x cH x cW

                current_confusion_matrix_objects = confusion_matrix(y_true=gt_crops_objects.flatten(), y_pred=pred_labels_objects.flatten(), labels=object_labels)
                current_confusion_matrix_objects = torch.tensor(current_confusion_matrix_objects)
                current_confusion_matrix_spatial = confusion_matrix(y_true=gt_crops_spatial.flatten(), y_pred=pred_labels_spatial.flatten(), labels=spatial_labels)
                current_confusion_matrix_spatial = torch.tensor(current_confusion_matrix_spatial)

                if overall_confusion_matrix_objects is None:
                    overall_confusion_matrix_objects = current_confusion_matrix_objects
                    overall_confusion_matrix_spatial = current_confusion_matrix_spatial
                else:
                    overall_confusion_matrix_objects += current_confusion_matrix_objects
                    overall_confusion_matrix_spatial += current_confusion_matrix_spatial

                # Stop testing if test iterations has been exceeded
                if tstep > self.options.test_iters:
                    break

        mAcc_obj = metrics.overall_pixel_accuracy(overall_confusion_matrix_objects)
        class_mAcc_obj, _ = metrics.per_class_pixel_accuracy(overall_confusion_matrix_objects)
        mIoU_obj, _ = metrics.jaccard_index(overall_confusion_matrix_objects)
        mF1_obj, _ = metrics.F1_Score(overall_confusion_matrix_objects)

        mAcc_sp = metrics.overall_pixel_accuracy(overall_confusion_matrix_spatial)
        class_mAcc_sp, _ = metrics.per_class_pixel_accuracy(overall_confusion_matrix_spatial)
        mIoU_sp, _ = metrics.jaccard_index(overall_confusion_matrix_spatial)
        mF1_sp, _ = metrics.F1_Score(overall_confusion_matrix_spatial)

        output = {}
        output['metrics'] = {'overall_pixel_accuracy_objects':mAcc_obj,
                             'per_class_pixel_accuracy_objects':class_mAcc_obj,
                             'mean_interesction_over_union_objects':mIoU_obj,
                             'mean_f1_score_objects':mF1_obj,
                             'overall_pixel_accuracy_spatial':mAcc_sp,
                             'per_class_pixel_accuracy_spatial':class_mAcc_sp,
                             'mean_interesction_over_union_spatial':mIoU_sp,
                             'mean_f1_score_spatial':mF1_sp}
        output['losses'] = {'pred_map_err_objects': loss_output['pred_map_err_objects'],
                             'pred_map_err_spatial': loss_output['pred_map_err_spatial']}
        output['maps'] = {'pred_maps_objects':pred_maps_objects,
                          'pred_maps_spatial':pred_maps_spatial}

        for k in output['metrics']:
            output['metrics'][k] = torch.mean(output['metrics'][k])
        for k in output['losses']:
            output['losses'][k] = torch.mean(output['losses'][k])

        self._save_summaries(batch, output, save_images=True, is_train=False)



    def _save_summaries(self, batch, output, save_images, is_train=False):
        prefix = 'train/' if is_train else 'test/'

        if save_images:
            # input crops
            color_step_geo_crops = viz_utils.colorize_grid(batch['step_ego_grid_crops_spatial'], color_mapping=3)
            self.summary_writer.add_video(prefix+"gifs/input_crops", color_step_geo_crops, self.step_count, fps=0.25)
            # predicted crops
            color_pred_crops_spatial = viz_utils.colorize_grid(output['maps']['pred_maps_spatial'], color_mapping=3)
            self.summary_writer.add_video(prefix+"gifs/pred_crops_spatial", color_pred_crops_spatial, self.step_count, fps=0.25)
            color_pred_crops_objects = viz_utils.colorize_grid(output['maps']['pred_maps_objects'], color_mapping=18)
            self.summary_writer.add_video(prefix+"gifs/pred_crops_objects", color_pred_crops_objects, self.step_count, fps=0.25)
            # gt crops
            color_gt_crops_spatial = viz_utils.colorize_grid(batch['gt_grid_crops_spatial'], color_mapping=3)
            self.summary_writer.add_video(prefix+"gifs/gt_crops_spatial", color_gt_crops_spatial, self.step_count, fps=0.25)
            color_gt_crops_objects = viz_utils.colorize_grid(batch['gt_grid_crops_objects'], color_mapping=18)
            self.summary_writer.add_video(prefix+"gifs/gt_crops_objects", color_gt_crops_objects, self.step_count, fps=0.25)

        for scalar_type in ['losses', 'metrics']:
            for k in output[scalar_type]:
                self.summary_writer.add_scalar(prefix + k, output[scalar_type][k], self.step_count)

        if is_train:
            self.summary_writer.add_scalar(prefix + "lr", self.get_lr(), self.step_count)
