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
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from sklearn.metrics import confusion_matrix
import metrics
import os


class TrainerFinetune(BaseTrainer):
    """ Implements training for prediction models
    """
    def init_fn(self):
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.train_ds = HabitatDataOffline(self.options, config_file=self.options.config_train_file, 
                                                img_segm=self.options.with_img_segm, finetune=self.options.finetune)
        self.test_ds = HabitatDataOffline(self.options, config_file=self.options.config_val_file, 
                                                img_segm=self.options.with_img_segm, finetune=self.options.finetune)

        self.model_id = self.options.model_number
        ensemble_exp = os.listdir(self.options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        self.models_dict = {'predictor_model': get_predictor_from_options(self.options)}
        self.models_dict = {k:v.to(self.device) for k,v in self.models_dict.items()}

        # Needed only for models trained with multi-gpu setting
        self.models_dict['predictor_model'] = nn.DataParallel(self.models_dict['predictor_model'])

        checkpoint_dir = self.options.ensemble_dir + "/" + ensemble_exp[self.model_id-1]
        latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
        print("Model", self.model_id, "loading checkpoint", latest_checkpoint)
        self.models_dict = tutils.load_model(models=self.models_dict, checkpoint_file=latest_checkpoint)


        self.optimizers_dict = {}
        for model in self.models_dict:
            self.optimizers_dict[model] = \
                    torch.optim.Adam([{'params':self.models_dict[model].parameters(),
                                    'initial_lr':self.options.lr}],
                                    lr=self.options.lr,
                                    betas=(self.options.beta1, 0.999) )
        
        # Load the pre-trained img segmentation model
        if self.options.with_img_segm:
            self.img_segmentor = get_img_segmentor_from_options(self.options)
            self.img_segmentor = self.img_segmentor.to(self.device)
            
            # Needed only for models trained with multi-gpu setting
            self.img_segmentor = nn.DataParallel(self.img_segmentor)

            latest_checkpoint = tutils.get_latest_model(save_dir=self.options.img_segm_model_dir)
            print("Loading image segmentation checkpoint", latest_checkpoint)
            
            checkpoint = torch.load(latest_checkpoint)
            self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
            self.img_segmentor.eval()

            # used for ground-projection # default HFOV is 90 degrees
            self.hfov = 90.0 * np.pi / 180.
            self.cell_size = self.options.cell_size
            self.object_labels = self.options.n_object_classes
            self.crop_size = (self.options.crop_size, self.options.crop_size)
            self.img_segm_size = (self.options.img_segm_size,self.options.img_segm_size)

            # Build 3D transformation matrices
            self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1])), device='cuda')
            self.xs = self.xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
            self.ys = self.ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
            K = np.array([
                [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
                [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]])
            self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
            # create the points2D containing all image coordinates
            x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1])), device='cuda')
            xy_img = torch.cat((x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
            points2D_step = xy_img.reshape(2, -1)
            self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2

            

    def train_step(self, input_batch, step_count):
        for model in self.models_dict:
            self.models_dict[model].train()
        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].zero_grad()

        # Get the ground-projected image segmentation
        ### For finetuning we do not store the pred_ego_crops_sseg before-hand so we need to run the pretrained img_segm ###
        if self.options.with_img_segm:
            input_batch['pred_ego_crops_sseg'] = utils.run_img_segm(model=self.img_segmentor, 
                                                              input_batch=input_batch, 
                                                              object_labels=self.object_labels, 
                                                              crop_size=self.crop_size,
                                                              cell_size=self.cell_size,
                                                              xs=self.xs,
                                                              ys=self.ys,
                                                              inv_K=self.inv_K,
                                                              points2D_step=self.points2D_step)

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

                if self.options.with_img_segm:
                    batch['pred_ego_crops_sseg'] = utils.run_img_segm(model=self.img_segmentor,
                                                                      input_batch=batch,
                                                                      object_labels=self.object_labels,
                                                                      crop_size=self.crop_size,
                                                                      cell_size=self.cell_size,
                                                                      xs=self.xs,
                                                                      ys=self.ys,
                                                                      inv_K=self.inv_K,
                                                                      points2D_step=self.points2D_step)
                                                                      
                pred_output = self.models_dict['predictor_model'](batch)

                loss_output = self.models_dict['predictor_model'].module.loss_cel(batch, pred_output)

                pred_maps_objects = pred_output['pred_maps_objects']
                pred_maps_spatial = pred_output['pred_maps_spatial']

                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW
                pred_labels_spatial = torch.argmax(pred_maps_spatial.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

                gt_crops_spatial = batch['gt_grid_crops_spatial'].cpu() # B x T x 1 x cH x cW
                gt_crops_objects = batch['gt_grid_crops_objects'].cpu() # B x T x 1 x cH x cW

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
