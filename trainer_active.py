import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models.predictors import get_predictor_from_options
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
from datasets.dataloader import HabitatDataScene
from models.semantic_grid import SemanticGrid
import torchgeometry as tgm
import test_utils as tutils
from planning.ddppo_policy import DdppoPolicy


class ActiveTrainer(object):
    """ Implements training for prediction models
    """
    def __init__(self, options, scene_id):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scene_id = scene_id

        self.train_ds = HabitatDataScene(self.options, config_file=self.options.config_train_file, scene_id=self.scene_id)

        self.episodes_save_dir = self.options.active_ep_save_dir + self.options.split + "/" + self.scene_id + "/"
        if not os.path.exists(self.episodes_save_dir):
            os.makedirs(self.episodes_save_dir)

        # init ensemble
        self.optimizers_dict = {}
        ensemble_exp = os.listdir(self.options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}
            self.models_dict[n] = {k:v.to(self.device) for k,v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = self.options.ensemble_dir + "/" + ensemble_exp[n]
            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()


            self.optimizers_dict[n]={}
            self.optimizers_dict[n]['predictor_model'] = \
                    torch.optim.Adam([{'params':self.models_dict[n]['predictor_model'].parameters(),
                                    'initial_lr':self.options.lr}],
                                    lr=self.options.lr,
                                    betas=(self.options.beta1, 0.999) )

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


        self.step_count = 0
        self.gauss_filter = tgm.image.GaussianBlur((11, 11), (10.5, 10.5))

        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        model_path = self.options.root_path + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy.to(self.device)


        # Build 3D transformation matrices
        # Need to do them independent of dataloader because the img_segm projection has different dimensions
        self.img_segm_size = (128,128)
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self.ys = self.ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


    def train_active(self):

        with torch.no_grad():

            for idx in range(len(self.train_ds)):

                if idx >= self.options.max_num_episodes:
                    break

                episode = self.train_ds.scene_data['episodes'][idx]

                self.step_count+=1 # episode counter

                scene = self.train_ds.sim.semantic_annotations()
                # convert the labels to the reduced set of categories
                instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
                instance_id_to_label_id_3 = instance_id_to_label_id.copy()
                instance_id_to_label_id_27 = instance_id_to_label_id.copy()
                for inst_id in instance_id_to_label_id.keys():
                    curr_lbl = instance_id_to_label_id[inst_id]
                    instance_id_to_label_id_3[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
                    instance_id_to_label_id_27[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]


                self.train_ds.sim.reset()
                self.train_ds.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                sim_obs = self.train_ds.sim.get_sensor_observations()
                observations = self.train_ds.sim._sensor_suite.get_observations(sim_obs)                

                sg = SemanticGrid(1, self.train_ds.grid_dim, self.train_ds.crop_size[0], self.train_ds.cell_size,
                                            spatial_labels=self.train_ds.spatial_labels, object_labels=self.train_ds.object_labels)

                # Initialize the local policy hidden state
                self.l_policy.reset()

                abs_poses = [] 
                agent_height = []
                t = 0
                ltg_counter=0
                ltg = torch.zeros((1, 1, 2), dtype=torch.int64, device=self.device)

                ego_grid_crops_spatial = torch.zeros((self.options.episode_len, self.train_ds.spatial_labels,
                                                        self.train_ds.crop_size[0], self.train_ds.crop_size[1]), dtype=torch.float32)
                step_ego_grid_crops_spatial = torch.zeros((self.options.episode_len, self.train_ds.spatial_labels,
                                                        self.train_ds.crop_size[0], self.train_ds.crop_size[1]), dtype=torch.float32)
                imgs = torch.zeros((self.options.episode_len, 3, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device='cuda')
                depth_imgs = torch.zeros((self.options.episode_len, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device='cuda')
                ssegs_27 = np.zeros((self.options.episode_len, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=np.float32)


                for t in range(self.options.episode_len):

                    img = observations['rgb'][:,:,:3]
                    depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)
                    depth = F.interpolate(depth_obsv.clone(), size=self.train_ds.img_size, mode='nearest')
                    depth = depth.squeeze(0).permute(1,2,0)

                    if self.train_ds.cfg_norm_depth:
                        depth = utils.unnormalize_depth(depth, min=self.train_ds.min_depth, max=self.train_ds.max_depth)
                    
                    semantic = observations['semantic']
                    semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0).float(), size=self.train_ds.img_size, mode='nearest').int()
                    semantic = semantic.squeeze(0).squeeze(0)

                    # 3d info
                    local3D_step = utils.depth_to_3D(depth, self.train_ds.img_size, self.train_ds.xs, self.train_ds.ys, self.train_ds.inv_K)

                    ssegData = np.expand_dims(semantic.cpu().numpy(), 0).astype(float) # 1 x H x W
                    ssegData_3 = np.vectorize(instance_id_to_label_id_3.get)(ssegData.copy()) # convert instance ids to category ids
                    ssegData_27 = np.vectorize(instance_id_to_label_id_27.get)(ssegData.copy()) # convert instance ids to category ids

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.train_ds.sim.get_agent_state())

                    abs_poses.append(agent_pose)
                    agent_height.append(y_height)

                    # get the relative pose with respect to the first pose in the sequence
                    rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
                    _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                    pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0], self.train_ds.grid_dim[0], self.train_ds.cell_size, device=self.device) # B x T x 3

                    # do ground-projection, update the map
                    if self.options.occ_from_depth:
                        ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.train_ds.grid_dim, cell_size=self.train_ds.cell_size, 
                                                                                    device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)
                    else:
                        ego_grid_sseg_3 = map_utils.ground_projection([self.train_ds.points2D_step], [local3D_step], ssegs_3.unsqueeze(0).unsqueeze(0), sseg_labels=self.train_ds.spatial_labels,
                                                                                            grid_dim=self.train_ds.grid_dim, cell_size=self.train_ds.cell_size)

                    ego_grid_crops_3 = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.train_ds.crop_size)

                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg_3, pose=_rel_pose, abs_pose=torch.tensor(abs_poses))
                    # step_geo_grid contains the map snapshot every time a new observation is added
                    step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
                    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                    step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_pose, abs_pose=torch.tensor(abs_poses))
                    # Crop the grid around the agent at each timestep
                    step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.train_ds.crop_size)


                    # store info for every step of the episode
                    ego_grid_crops_spatial[t,:,:,:] = ego_grid_crops_3
                    step_ego_grid_crops_spatial[t,:,:,:] = step_ego_grid_crops
                    imgData = utils.preprocess_img(img, cropSize=self.train_ds.img_segm_size, pixFormat='NCHW', normalize=True)
                    imgs[t,:,:,:] = imgData
                    depth_resize = depth.clone().permute(2,0,1).unsqueeze(0)
                    depth_img = F.interpolate(depth_resize, size=self.train_ds.img_segm_size, mode='nearest')
                    depth_imgs[t,:,:,:] = depth_img
                    ssegData_resize = cv2.resize(ssegData_27.reshape(self.train_ds.img_size[0], self.train_ds.img_size[1]), 
                                                                                            self.train_ds.img_segm_size, interpolation=cv2.INTER_NEAREST)
                    ssegs_27[t,:,:,:] = ssegData_resize.reshape(1, self.train_ds.img_segm_size[0], self.train_ds.img_segm_size[1])

                    # Run the image segmentation, map prediction and uncertainty
                    if self.options.with_img_segm:
                        img_segm_input = {'images': imgData.unsqueeze(0).unsqueeze(0),
                                          'depth_imgs': depth_img.unsqueeze(0)}

                        pred_ego_crops_sseg = utils.run_img_segm(model=self.img_segmentor,
                                                                     input_batch=img_segm_input,
                                                                     object_labels=self.train_ds.object_labels,
                                                                     crop_size=self.train_ds.crop_size,
                                                                     cell_size=self.train_ds.cell_size,
                                                                     xs=self.xs,
                                                                     ys=self.ys,
                                                                     inv_K=self.train_ds.inv_K,
                                                                     points2D_step=self.points2D_step)

                        mean_ensemble_uncertainty = self.run_map_predictor(step_ego_grid_crops, self.options.uncertainty_type, pred_ego_crops_sseg)
                    else:
                        mean_ensemble_uncertainty = self.run_map_predictor(step_ego_grid_crops)

                    # add crop uncertainty to uncertainty map
                    sg.register_uncertainty(uncertainty_crop=mean_ensemble_uncertainty, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))

                    ltg_dist = torch.linalg.norm(ltg.clone().float()-pose_coords.float())*torch.tensor(self.options.cell_size,device=self.device) # distance to current long-term goal

                    # Estimate long term goal
                    if ((ltg_counter % self.options.steps_after_plan == 0) or  # either every k steps
                       (ltg_dist < 0.2)): # or we reached ltg

                        # Use cost map to decide next long term direction
                        ltg = self.get_long_term_goal(sg, cost_map=sg.uncertainty_map)
                        ltg_counter = 0 # reset the ltg counter
                    ltg_counter += 1

                    ##### Action decision process #####

                    ### Use local policy to select action ###
                    action_id = self.run_local_policy(depth=depth, goal=ltg.clone(),
                                                                    pose_coords=pose_coords.clone(), rel_agent_o=rel[2], step=t)
                    # if stop is selected from local policy then randomly select an action
                    if action_id==0:
                       action_id = random.randint(1,3)

                    # explicitly clear observation otherwise they will be kept in memory the whole time
                    observations = None

                    # Apply next action
                    observations = self.train_ds.sim.step(action_id)


                # save the episode -- follow the store_episodes_parallel.py convention

                gt_grid_crops_spatial = map_utils.get_gt_crops(torch.tensor(abs_poses), self.train_ds.pcloud, self.train_ds.label_seq_spatial, agent_height,
                                                                    self.train_ds.grid_dim, self.train_ds.crop_size, self.train_ds.cell_size)
                gt_grid_crops_objects = map_utils.get_gt_crops(torch.tensor(abs_poses), self.train_ds.pcloud, self.train_ds.label_seq_objects, agent_height,
                                                                    self.train_ds.grid_dim, self.train_ds.crop_size, self.train_ds.cell_size)

                filepath = self.episodes_save_dir+'ep_'+str(idx)+'_'+str(idx)+"_"+self.scene_id
                np.savez_compressed(filepath+'.npz',
                                    abs_pose=abs_poses,
                                    ego_grid_crops_spatial=ego_grid_crops_spatial,
                                    step_ego_grid_crops_spatial=step_ego_grid_crops_spatial,
                                    gt_grid_crops_spatial=gt_grid_crops_spatial,
                                    gt_grid_crops_objects=gt_grid_crops_objects,
                                    images=imgs.cpu().numpy(),
                                    ssegs=ssegs_27,
                                    depth_imgs=depth_imgs.cpu().numpy())


    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        # Replicates the output of a IntegratedPointGoalGPSAndCompassSensor
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho*self.train_ds.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.train_ds.img_size[0], self.train_ds.img_size[1], 1)
        depth_values = (depth - self.train_ds.min_depth)/(self.train_ds.max_depth-self.train_ds.min_depth)
        return self.l_policy.plan(depth_values, point_goal_with_gps_compass, step)


    def get_long_term_goal(self, sg, cost_map):
        ### Choose long term goal
        goal = torch.zeros((sg.per_class_uncertainty_map.shape[0], 1, 2), dtype=torch.int64, device=self.device)
        explored_grid = map_utils.get_explored_grid(sg.proj_grid) # B x 1 x H x W
        current_UNexplored_map = 1-explored_grid

        # find the ltg from the unobserved areas of the map
        unexplored_cost_map = cost_map * current_UNexplored_map

        unexplored_cost_map = unexplored_cost_map.squeeze(1)
        for b in range(unexplored_cost_map.shape[0]):
            map_ = unexplored_cost_map[b,:,:]
            inds = utils.unravel_index(map_.argmax(), map_.shape)
            goal[b,0,0] = inds[1]
            goal[b,0,1] = inds[0]

        return goal


    def run_map_predictor(self, step_ego_grid_crops, uncertainty_type, pred_ego_crops_sseg=None):

        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops.unsqueeze(0)}
        if self.options.with_img_segm:
            input_batch['pred_ego_crops_sseg'] = pred_ego_crops_sseg
        input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

        model_pred_output = {} # keep track of each individual model predictions to apply the loss later
        ensemble_pred_maps = [] 
        for n in range(self.options.ensemble_size):
            model_pred_output[n] = self.models_dict[n]['predictor_model'](input_batch)
            ensemble_pred_maps.append(model_pred_output[n]['pred_maps_objects'].clone())
        ensemble_pred_maps = torch.stack(ensemble_pred_maps) # N x B x T x C x cH x cW

        B, T, C, cH, cW = model_pred_output[0]['pred_maps_objects'].shape

        if uncertainty_type=="entropy": # predictive entropy
            # Estimate average predictions from the ensemble
            mean_ensemble_prediction = torch.mean(ensemble_pred_maps, dim=0) # B x T x C x cH x cW
            mean_ensemble_uncertainty = utils.get_entropy(pred=mean_ensemble_prediction) # B x T x 1 x cH x cW

        elif uncertainty_type=="epistemic":
            ### Estimate the variance of each class for each location # 1 x B x T x object_classes x crop_dim x crop_dim
            ensemble_var = torch.zeros((1, B, T, C, cH, cW), dtype=torch.float32, device=self.device)
            for i in range(ensemble_pred_maps.shape[3]): # num of classes
                ensemble_class = ensemble_pred_maps[:,:,:,i,:,:]
                ensemble_class_var = torch.var(ensemble_class, dim=0, keepdim=True)
                ensemble_var[:,:,:,i,:,:] = ensemble_class_var
            # Estimate mean class variance for each location
            mean_ensemble_uncertainty = torch.mean(ensemble_var, dim=3, keepdim=True).squeeze(0) # B x T x 1 x cH x cW

        elif uncertainty_type=="bald": # maximise the mutual information between predictions and model posterior
            pred_entropy = torch.zeros((self.options.ensemble_size, B, T, 1, cH, cW), dtype=torch.float32, device=self.device)
            for n in range(ensemble_pred_maps.shape[0]):
                pred_entropy[n,:,:,:,:,:] = utils.get_entropy(pred=ensemble_pred_maps[n,:,:,:,:,:])
            mean_pred_entropy = torch.mean(pred_entropy, dim=0)

            mean_ensemble_prediction = torch.mean(ensemble_pred_maps, dim=0) # B x T x C x cH x cW
            predictive_entropy = utils.get_entropy(pred=mean_ensemble_prediction) # B x T x 1 x cH x cW
            mean_ensemble_uncertainty = predictive_entropy - mean_pred_entropy # B x T x 1 x cH x cW

        else:
            raise Exception('Unknown uncertainty type')

        # Apply gaussian blur on the mean uncertainty
        mean_ensemble_uncertainty = self.gauss_filter(mean_ensemble_uncertainty.view(B*T, 1, cH, cW)).view(B,T,1,cH,cW)

        return mean_ensemble_uncertainty
