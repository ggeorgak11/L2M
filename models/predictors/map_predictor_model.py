
import torch
import torch.nn as nn
import torch.nn.functional as F


class MapPredictorHier(nn.Module):
    def __init__(self, segmentation_model, map_loss_scale, with_img_segm):
        super(MapPredictorHier, self).__init__()
        self._segmentation_model = segmentation_model
        self._map_loss_scale = map_loss_scale
        self.with_img_segm = with_img_segm
        
        self.cel_loss_spatial = nn.CrossEntropyLoss()
        self.cel_loss_objects = nn.CrossEntropyLoss()
        

    def forward(self, batch, is_train=True):

        step_ego_crops = batch['step_ego_grid_crops_spatial']
        B, T, _, cH, cW = step_ego_crops.shape # batch, sequence length, _, crop height, crop width

        if self.with_img_segm:
            pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops, batch['pred_ego_crops_sseg'])
        else:
            pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops)

        # number of classes for each case
        spatial_C = pred_maps_raw_spatial.shape[1]
        objects_C = pred_maps_raw_objects.shape[1]

        # Get a prob distribution over the labels
        pred_maps_raw_spatial = pred_maps_raw_spatial.view(B,T,spatial_C,cH,cW)
        pred_maps_spatial = F.softmax(pred_maps_raw_spatial, dim=2)
        pred_maps_raw_objects = pred_maps_raw_objects.view(B,T,objects_C,cH,cW)
        pred_maps_objects = F.softmax(pred_maps_raw_objects, dim=2)

        output = {'pred_maps_raw_spatial':pred_maps_raw_spatial,
                  'pred_maps_raw_objects':pred_maps_raw_objects,
                  'pred_maps_spatial':pred_maps_spatial,
                  'pred_maps_objects':pred_maps_objects}
        return output

    
    def loss_cel(self, batch, pred_outputs):
        pred_maps_raw_spatial = pred_outputs['pred_maps_raw_spatial']
        pred_maps_raw_objects = pred_outputs['pred_maps_raw_objects']
        B, T, spatial_C, cH, cW = pred_maps_raw_spatial.shape
        objects_C = pred_maps_raw_objects.shape[2]

        gt_crops_spatial, gt_crops_objects = batch['gt_grid_crops_spatial'], batch['gt_grid_crops_objects']
        pred_map_loss_spatial = self.cel_loss_spatial(input=pred_maps_raw_spatial.view(B*T,spatial_C,cH,cW), target=gt_crops_spatial.view(B*T,cH,cW))
        pred_map_loss_objects = self.cel_loss_objects(input=pred_maps_raw_objects.view(B*T,objects_C,cH,cW), target=gt_crops_objects.view(B*T,cH,cW))
        
        pred_map_err_spatial = pred_map_loss_spatial.clone().detach()
        pred_map_err_objects = pred_map_loss_objects.clone().detach()

        output={}
        output['pred_map_err_spatial'] = pred_map_err_spatial
        output['pred_map_err_objects'] = pred_map_err_objects
        output['pred_map_loss_spatial'] = self._map_loss_scale * pred_map_loss_spatial
        output['pred_map_loss_objects'] = self._map_loss_scale * pred_map_loss_objects
        return output