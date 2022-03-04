import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataScene, HabitatDataOffline
from models.predictors import get_predictor_from_options
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import metrics
import json
import cv2
import random
import math
from planning.ddppo_policy import DdppoPolicy


class NavTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        # point to our generated test episodes
        self.options.episodes_root = "habitat-api/data/datasets/objectnav/mp3d/"+self.options.test_set+"/"

        self.scene_id = scene_id
        self.test_ds = HabitatDataScene(self.options, config_file=self.options.config_test_file, scene_id=self.scene_id)


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

        # Define what threshold to use for semantic prediction
        self.sem_thresh = self.options.sem_thresh

        if self.options.test_set=="v3" or self.options.test_set=="v5":
            self.target_sem_lbls = {"chair":1, "sofa":5, "bed":6, "cushion":4, "counter":13, "table":3}
            # put a limit on the number of episodes to run for each object
            self.episode_counter = {"chair":0, "sofa":0, "bed":0, "cushion":0, "counter":0, "table":0}
            self.ep_limit = 25
        elif self.options.test_set=="v6":
            self.target_sem_lbls = {"plant":7, "toilet":9, "tv_monitor":10}
        elif self.options.test_set=="v7":
            self.target_sem_lbls = {"cabinet":19, "fireplace":23}

        self.metrics = ['distance_to_goal', 'success', 'spl', 'softspl']

        # initialize metrics
        self.results = {}
        for target in self.target_sem_lbls:
            self.results[target] = {}
            for met in self.metrics:
                self.results[target][met] = []
        self.results['all'] = {}
        for met in self.metrics:
            self.results['all']['mean_'+met] = []
        self.results['all']['mean_distance_to_goal_failed'] = []

        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        model_path = self.options.root_path + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy = self.l_policy.to(self.device)


        # Build 3D transformation matrices
        # Need to do them independent of dataloader because the img_segm projection has different dimensions
        self.img_segm_size = (self.options.img_segm_size,self.options.img_segm_size)
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self.ys = self.ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


    def test_navigation(self):

        with torch.no_grad():

            list_dist_to_goal, list_success, list_spl, list_soft_spl = [],[],[],[]

            for idx in range(len(self.test_ds)):

                episode = self.test_ds.scene_data['episodes'][idx]

                len_shortest_path = len(episode['shortest_paths'][0])
                objectgoal = episode['object_category']

                if objectgoal not in self.target_sem_lbls.keys():
                    continue
                else:
                    sem_lbl = self.target_sem_lbls[objectgoal]

                if self.options.test_set=="v3":
                    if self.episode_counter[objectgoal] >= self.ep_limit:
                        continue
                    self.episode_counter[objectgoal] += 1

                # Collect all predefined viewpoint goals for a specific object category (to use in metrics)
                goals = self.test_ds.scene_data['goals_by_category'][self.scene_id+'.glb_'+objectgoal]
                episode_goal_positions = [ viewpoint['agent_state']['position']
                                           for goal in goals
                                           for viewpoint in goal['view_points'] ]


                print("Ep:", idx, objectgoal, "Sem lbl:", sem_lbl, "Len:", len_shortest_path)
                self.step_count+=1 # episode counter for tensorboard


                if not self.options.occ_from_depth or self.options.use_semantic_sensor:
                    scene = self.test_ds.sim.semantic_annotations()
                    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
                    # convert the labels to the reduced set of categories
                    instance_id_to_label_id_spatial = instance_id_to_label_id.copy()
                    instance_id_to_label_id_objects = instance_id_to_label_id.copy()
                    for inst_id in instance_id_to_label_id.keys():
                        curr_lbl = instance_id_to_label_id[inst_id]
                        instance_id_to_label_id_spatial[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
                        instance_id_to_label_id_objects[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]


                self.test_ds.sim.reset()
                self.test_ds.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                sim_obs = self.test_ds.sim.get_sensor_observations()
                observations = self.test_ds.sim._sensor_suite.get_observations(sim_obs)

                # For each episode we need a new instance of a fresh global grid
                sg = SemanticGrid(1, self.test_ds.grid_dim, self.test_ds.crop_size[0], self.test_ds.cell_size,
                                    spatial_labels=self.test_ds.spatial_labels, object_labels=self.test_ds.object_labels)

                # Initialize the local policy hidden state
                self.l_policy.reset()

                abs_poses = []
                agent_height = []
                t = 0
                ltg_counter=0
                ltg = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
                agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
                previous_pos = self.test_ds.sim.get_agent_state().position

                while t < self.options.max_steps:

                    img = observations['rgb'][:,:,:3]
                    depth = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
                    semantic = observations['semantic']

                    if self.test_ds.cfg_norm_depth:
                        depth = utils.unnormalize_depth(depth, min=self.test_ds.min_depth, max=self.test_ds.max_depth)

                    # 3d info
                    local3D_step = utils.depth_to_3D(depth, self.test_ds.img_size, self.test_ds.xs, self.test_ds.ys, self.test_ds.inv_K)

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.test_ds.sim.get_agent_state())

                    abs_poses.append(agent_pose)
                    agent_height.append(y_height)

                    # get the relative pose with respect to the first pose in the sequence
                    rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
                    _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                    _rel_pose = _rel_pose.to(self.device)

                    pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3

                    # do ground-projection, update the map
                    if self.options.occ_from_depth:
                        ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size, 
                                                                                    device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)
                    else:
                        # in this case we use the semantic sensor to obtain the ground-projection during the episode
                        ssegData = np.expand_dims(semantic.cpu().numpy(), 0).astype(float) # 1 x H x W
                        ssegData_spatial = np.vectorize(instance_id_to_label_id_spatial.get)(ssegData.copy()) # convert instance ids to category ids
                        ssegs_spatial = torch.from_numpy(ssegData_spatial).unsqueeze(0).float()
                        ego_grid_sseg_3 = map_utils.ground_projection([self.test_ds.points2D_step], [local3D_step], ssegs_spatial, sseg_labels=self.test_ds.spatial_labels,
                                                                                            grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size)

                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg_3, pose=_rel_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
                    # step_geo_grid contains the map snapshot every time a new observation is added
                    step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
                    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                    step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
                    # Crop the grid around the agent at each timestep
                    step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.test_ds.crop_size)

                    # Run the image segmentation, map prediction and uncertainty
                    if self.options.with_img_segm:
                        imgData = utils.preprocess_img(img, cropSize=self.img_segm_size, pixFormat='NCHW', normalize=True)
                        depth_resize = depth.clone().permute(2,0,1).unsqueeze(0)
                        depth_img = F.interpolate(depth_resize, size=self.img_segm_size, mode='nearest')

                        img_segm_input = {'images': imgData.unsqueeze(0).unsqueeze(0),
                                          'depth_imgs': depth_img.unsqueeze(0)}

                        if self.options.use_semantic_sensor:
                            # get img segmentation from the simulator's semantic sensor
                            ssegData = np.expand_dims(semantic.cpu().numpy(), 0).astype(float) # 1 x H x W
                            ssegData_objects = np.vectorize(instance_id_to_label_id_objects.get)(ssegData.copy()) # convert instance ids to category ids
                            ssegs_objects = torch.from_numpy(ssegData_objects).unsqueeze(0).unsqueeze(0).float()
                            img_labels = ssegs_objects.clone()
                        else:
                            img_labels = None
                            
                        pred_ego_crops_sseg = utils.run_img_segm(model=self.img_segmentor,
                                                                     input_batch=img_segm_input,
                                                                     object_labels=self.test_ds.object_labels,
                                                                     crop_size=self.test_ds.crop_size,
                                                                     cell_size=self.test_ds.cell_size,
                                                                     xs=self.xs,
                                                                     ys=self.ys,
                                                                     inv_K=self.test_ds.inv_K,
                                                                     points2D_step=self.points2D_step,
                                                                     img_labels=img_labels)

                        mean_ensemble_prediction, mean_ensemble_spatial, per_class_uncertainty = self.run_map_predictor(step_ego_grid_crops, pred_ego_crops_sseg)
                    else:
                        mean_ensemble_prediction, mean_ensemble_spatial, per_class_uncertainty = self.run_map_predictor(step_ego_grid_crops)


                    # add crop uncertainty to uncertainty map
                    sg.register_per_class_uncertainty(per_class_uncertainty_crop=per_class_uncertainty, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))
                    # add semantic prediction to semantic map
                    sg.register_sem_pred(prediction_crop=mean_ensemble_prediction, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))


                    ltg_dist = torch.linalg.norm(ltg.clone().float()-pose_coords.float())*self.options.cell_size # distance to current long-term goal

                    # Estimate long term goal
                    if ((ltg_counter % self.options.steps_after_plan == 0) or  # either every k steps
                       (ltg_dist < 0.2)): # or we reached ltg

                        # Given hyperparams, estimate the cost map
                        cost_map = tutils.get_cost_map(sg, sem_lbl, self.options.a_1, self.options.a_2)
                        # Use cost map to decide next long term direction
                        ltg = self.get_long_term_goal(sg, cost_map)
                        ltg_counter = 0 # reset the ltg counter
                    ltg_counter += 1


                    # Option to save visualizations of steps
                    if self.options.save_nav_images:
                        save_img_dir_ = self.options.save_img_dir_+'ep_'+str(idx)+'_'+objectgoal+'/'
                        if not os.path.exists(save_img_dir_):
                            os.makedirs(save_img_dir_)
                        # saves egocentric rgb, depth observations
                        viz_utils.display_sample(img.cpu().numpy(), np.squeeze(depth.cpu().numpy()), 
                                                            semantic.cpu().numpy(), savepath=save_img_dir_+"path_"+str(t)+'.png')
                        # saves semantic grid (geocentric), object heatmap, and uncertainty
                        viz_utils.save_visual_steps(self.test_ds, sg, sem_lbl, abs_poses[t], ltg.clone().cpu().numpy(), 
                                                            pose_coords.clone().cpu().numpy(), agent_height, save_img_dir_, t)
                        # saves predicted ares (egocentric)
                        viz_utils.save_map_pred_steps(step_ego_grid_crops, mean_ensemble_spatial, 
                                                            mean_ensemble_prediction, pred_ego_crops_sseg, save_img_dir_, t)
                        # saves image semantic segmentation
                        viz_utils.write_tensor_imgSegm(img=pred_img_segm, savepath=save_img_dir_, name='pred_img_segm', t=t)


                    ##### Action decision process #####

                    # At every step find the distance to the closest predicted occurence of the target
                    _, obj_dist, _ = tutils.get_closest_target_location(sg, pose_coords.clone(), sem_lbl, self.options.cell_size, self.sem_thresh)
                    
                    # Check stopping criteria
                    if tutils.decide_stop(obj_dist, self.options.stop_dist) or t==self.options.max_steps-1:
                        t+=1
                        break

                    action_id = self.run_local_policy(depth=depth, goal=ltg.clone(),
                                                                pose_coords=pose_coords.clone(), rel_agent_o=rel[2], step=t)
                    # if stop is selected from local policy then randomly select an action
                    if action_id==0:
                        action_id = random.randint(1,3)

                    # explicitly clear observation otherwise they will be kept in memory the whole time
                    observations = None

                    # Apply next action
                    observations = self.test_ds.sim.step(action_id)

                    # estimate distance covered by agent
                    current_pos = self.test_ds.sim.get_agent_state().position
                    agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                    previous_pos = current_pos

                    t+=1


                ## Episode ended ##
                nav_metrics = tutils.get_metrics(sim=self.test_ds.sim,
                                            episode_goal_positions=episode_goal_positions,
                                            success_distance=self.test_ds.success_distance,
                                            start_end_episode_distance=episode['info']['geodesic_distance'],
                                            agent_episode_distance=agent_episode_distance,
                                            stop_signal=True)
                for met in self.metrics: # metrics per object
                    self.results[objectgoal][met].append(nav_metrics[met])

                list_dist_to_goal.append(nav_metrics['distance_to_goal'])
                list_success.append(nav_metrics['success'])
                list_spl.append(nav_metrics['spl'])
                list_soft_spl.append(nav_metrics['softspl'])


                output = {}
                output['metrics'] = {'mean_dist_to_goal': np.mean(np.asarray(list_dist_to_goal.copy())),
                                     'mean_success': np.mean(np.asarray(list_success.copy())),
                                     'mean_spl': np.mean(np.asarray(list_spl.copy())),
                                     'mean_soft_spl': np.mean(np.asarray(list_soft_spl.copy()))}
                self.save_test_summaries(output)


            ## Scene ended ##
            # write results to json
            for target in self.target_sem_lbls:
                # keep distance_to_goal only in failed episodes
                failed_ind = np.asarray(self.results[target]['success'], dtype=np.int64)==0
                dist_to_goal_failed = np.asarray(self.results[target]['distance_to_goal'])[failed_ind]
                self.results[target]['mean_distance_to_goal_failed'] = np.mean(dist_to_goal_failed)

                for met in self.metrics:
                    self.results[target]["mean_"+met] = np.mean(np.asarray(self.results[target][met])) # per target per metric mean

                    if not math.isnan(self.results[target]['mean_'+met]): # collect mean values from each target
                        self.results['all']['mean_'+met].append(self.results[target]['mean_'+met])

                if not math.isnan(self.results[target]['mean_distance_to_goal_failed']):
                    self.results['all']['mean_distance_to_goal_failed'].append(self.results[target]['mean_distance_to_goal_failed'])
            # get mean over all targets for each metric
            for met in self.metrics:
                self.results['all']['mean_'+met] = np.mean( np.asarray(self.results['all']['mean_'+met]) )
            self.results['all']['mean_distance_to_goal_failed'] = np.mean( np.asarray(self.results['all']['mean_distance_to_goal_failed']) )

            print(self.results)
            with open(self.options.log_dir+'/results_'+self.scene_id+'.json', 'w') as outfile:
                json.dump(self.results, outfile, indent=4)



    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho*self.test_ds.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        depth_values = (depth - self.test_ds.min_depth)/(self.test_ds.max_depth-self.test_ds.min_depth)
        return self.l_policy.plan(depth_values, point_goal_with_gps_compass, step)


    def run_map_predictor(self, step_ego_grid_crops, pred_ego_crops_sseg=None):

        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops.unsqueeze(0)}
        if self.options.with_img_segm:
            input_batch['pred_ego_crops_sseg'] = pred_ego_crops_sseg
        input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

        model_pred_output = {} # keep track of each individual model predictions to apply the loss later
        ensemble_pred_maps = []
        ensemble_spatial_maps = []        
        for n in range(self.options.ensemble_size):
            model_pred_output[n] = self.models_dict[n]['predictor_model'](input_batch)
            ensemble_pred_maps.append(model_pred_output[n]['pred_maps_objects'].clone())
            ensemble_spatial_maps.append(model_pred_output[n]['pred_maps_spatial'].clone())
        ensemble_pred_maps = torch.stack(ensemble_pred_maps) # N x B x T x C x cH x cW
        ensemble_spatial_maps = torch.stack(ensemble_spatial_maps)

        ### Estimate average predictions from the ensemble
        mean_ensemble_prediction = torch.mean(ensemble_pred_maps, dim=0) # B x T x C x cH x cW
        mean_ensemble_spatial = torch.mean(ensemble_spatial_maps, dim=0) # B x T x C x cH x cW

        B, T, C, cH, cW = model_pred_output[0]['pred_maps_objects'].shape
        ### Estimate the variance of each class for each location # 1 x B x T x object_classes x crop_dim x crop_dim
        ensemble_var = torch.zeros((1, B, T, C, cH, cW), dtype=torch.float32).to(self.device)
        for i in range(ensemble_pred_maps.shape[3]): # num of classes
            ensemble_class = ensemble_pred_maps[:,:,:,i,:,:]
            ensemble_class_var = torch.var(ensemble_class, dim=0, keepdim=True)
            ensemble_var[:,:,:,i,:,:] = ensemble_class_var
        per_class_uncertainty = ensemble_var.squeeze(0) # B x T x C x cH x cW

        return mean_ensemble_prediction, mean_ensemble_spatial, per_class_uncertainty


    def get_long_term_goal(self, sg, cost_map):
        ### Choose long term goal
        goal = torch.zeros((sg.per_class_uncertainty_map.shape[0], 1, 2), dtype=torch.int64, device=self.device)
        explored_grid = map_utils.get_explored_grid(sg.proj_grid)
        current_UNexplored_map = 1-explored_grid
        unexplored_cost_map = cost_map * current_UNexplored_map
        unexplored_cost_map = unexplored_cost_map.squeeze(1)
        for b in range(unexplored_cost_map.shape[0]):
            map_ = unexplored_cost_map[b,:,:]
            inds = utils.unravel_index(map_.argmax(), map_.shape)
            goal[b,0,0] = inds[1]
            goal[b,0,1] = inds[0]
        return goal


    def save_test_summaries(self, output):
        prefix = 'test/' + self.scene_id + '/'
        for k in output['metrics']:
            self.summary_writer.add_scalar(prefix + k, output['metrics'][k], self.step_count)



class SemMapTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_ds = HabitatDataOffline(options, config_file=options.config_val_file, img_segm=self.options.with_img_segm)

        ensemble_exp = os.listdir(self.options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        N = len(ensemble_exp) # number of models in the ensemble
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

        self.spatial_classes = {0:"void", 1:"occupied", 2:"free"}
        self.object_classes = {0:"void", 17:"floor", 15:'wall', 3:"table", 4:"cushion", 13:"counter", 1:"chair", 5:"sofa", 6:"bed"}

        # initialize res dicts
        self.results_spatial = {}
        self.results_objects = {}
        for object_ in list(self.object_classes.values()):
            self.results_objects[object_] = {}
        self.results_objects['objects_all'] = {}
        for spatial in list(self.spatial_classes.values()):
            self.results_spatial[spatial] = {}
        self.results_spatial['spatial_all'] = {}



    def test_semantic_map(self):
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

                gt_crops_spatial = batch['gt_grid_crops_spatial'].cpu() # B x T x 1 x cH x cW
                gt_crops_objects = batch['gt_grid_crops_objects'].cpu() # B x T x 1 x cH x cW

                ensemble_object_maps, ensemble_spatial_maps = [], []
                N = len(self.models_dict) # numbers of models in the ensemble
                for n in range(self.options.ensemble_size):
                    pred_output = self.models_dict[n]['predictor_model'](batch)
                    ensemble_object_maps.append(pred_output['pred_maps_objects'].clone())
                    ensemble_spatial_maps.append(pred_output['pred_maps_spatial'].clone())
                ensemble_object_maps = torch.stack(ensemble_object_maps) # N x B x T x C x cH x cW
                ensemble_spatial_maps = torch.stack(ensemble_spatial_maps)

                # Getting the mean predictions from the ensemble
                pred_maps_objects = torch.mean(ensemble_object_maps, dim=0) # B x T x C x cH x cW
                pred_maps_spatial = torch.mean(ensemble_spatial_maps, dim=0)

                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW
                pred_labels_spatial = torch.argmax(pred_maps_spatial.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

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


        mAcc_sp = metrics.overall_pixel_accuracy(overall_confusion_matrix_spatial)
        class_mAcc_sp, per_class_Acc = metrics.per_class_pixel_accuracy(overall_confusion_matrix_spatial)
        mIoU_sp, per_class_IoU = metrics.jaccard_index(overall_confusion_matrix_spatial)
        mF1_sp, per_class_F1 = metrics.F1_Score(overall_confusion_matrix_spatial)

        print("Spatial prediction results:")
        for i in range(len(spatial_labels)):
            self.results_spatial[self.spatial_classes[i]] = {'Acc':per_class_Acc[i].item(),
                                                             'IoU':per_class_IoU[i].item(),
                                                             'F1':per_class_F1[i].item()}
            print("Class:", self.spatial_classes[i], "Acc:", per_class_Acc[i], "IoU:", per_class_IoU[i], "F1:", per_class_F1[i])
        print("mAcc:", mAcc_sp, "mIoU:", mIoU_sp, "mF1:", mF1_sp)
        self.results_spatial['spatial_all']['mAcc'] = mAcc_sp.item()
        self.results_spatial['spatial_all']['mIoU'] = mIoU_sp.item()
        self.results_spatial['spatial_all']['mF1'] = mF1_sp.item()

        mAcc_obj = metrics.overall_pixel_accuracy(overall_confusion_matrix_objects)
        class_mAcc_obj, per_class_Acc = metrics.per_class_pixel_accuracy(overall_confusion_matrix_objects)
        mIoU_obj, per_class_IoU = metrics.jaccard_index(overall_confusion_matrix_objects)
        mF1_obj, per_class_F1 = metrics.F1_Score(overall_confusion_matrix_objects)

        print("\nSemantic prediction results:")
        classes = list(self.object_classes.keys())
        classes.sort()
        per_class_Acc = per_class_Acc[classes]
        per_class_IoU = per_class_IoU[classes]
        per_class_F1 = per_class_F1[classes]
        for i in range(len(classes)):
            lbl = classes[i]
            self.results_objects[self.object_classes[lbl]] = {'Acc':per_class_Acc[i].item(),
                                                              'IoU':per_class_IoU[i].item(),
                                                              'F1':per_class_F1[i].item()}
            print("Class:", self.object_classes[lbl], "Acc:", per_class_Acc[i], "IoU:", per_class_IoU[i], "F1:", per_class_F1[i])
        mean_per_class_Acc = torch.mean(per_class_Acc)
        mean_per_class_IoU = torch.mean(per_class_IoU)
        mean_per_class_F1 = torch.mean(per_class_F1)
        print("mAcc:", mean_per_class_Acc, "mIoU:", mean_per_class_IoU, "mF1:", mean_per_class_F1)
        self.results_objects['objects_all']['mAcc'] = mean_per_class_Acc.item()
        self.results_objects['objects_all']['mIoU'] = mean_per_class_IoU.item()
        self.results_objects['objects_all']['mF1'] = mean_per_class_F1.item()

        res = {**self.results_spatial, **self.results_objects}
        with open(self.options.log_dir+'/sem_map_results.json', 'w') as outfile:
            json.dump(res, outfile, indent=4)

        # save the confusion matrices
        filepath = self.options.log_dir+'/confusion_matrices.npz'
        np.savez_compressed(filepath,
                            overall_confusion_matrix_spatial=overall_confusion_matrix_spatial,
                            overall_confusion_matrix_objects=overall_confusion_matrix_objects)

        print(overall_confusion_matrix_spatial)
        print()
        print(overall_confusion_matrix_objects)
