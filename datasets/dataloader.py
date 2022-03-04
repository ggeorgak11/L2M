
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
import habitat.utils.visualizations.maps as map_util
import datasets.util.utils as utils
import os
import gzip
import json


class HabitatDataOffline(Dataset):

    def __init__(self, options, config_file, img_segm=False, finetune=False):
        config = get_config(config_file)
        self.config = config
        
        self.img_segm = img_segm
        self.finetune = finetune # whether we are running a finetuning active job

        self.episodes_file_list = []
        self.episodes_file_list += self.collect_stored_episodes(options, split=config.DATASET.SPLIT)
        
        if options.dataset_percentage < 1: # Randomly choose the subset of the dataset to be used
            random.shuffle(self.episodes_file_list)
            self.episodes_file_list = self.episodes_file_list[ :int(len(self.episodes_file_list)*options.dataset_percentage) ]
        self.number_of_episodes = len(self.episodes_file_list)

        self.object_labels = options.n_object_classes

        if self.img_segm:
            self.episodes_imgSegm_dir = options.stored_imgSegm_episodes_dir
            self.episodes_dir = options.stored_episodes_dir


    def collect_stored_episodes(self, options, split):
        episodes_dir = options.stored_episodes_dir + split + "/"
        episodes_file_list = []
        _scenes_dir = os.listdir(episodes_dir)
        scenes_dir = [ x for x in _scenes_dir if os.path.isdir(episodes_dir+x) ]
        for scene in scenes_dir:
            for fil in os.listdir(episodes_dir+scene+"/"):
                episodes_file_list.append(episodes_dir+scene+"/"+fil)
        return episodes_file_list


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        # Load from the pre-stored objnav training episodes
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose']
        ego_grid_crops_spatial = torch.from_numpy(ep['ego_grid_crops_spatial'])
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'])
        gt_grid_crops_objects = torch.from_numpy(ep['gt_grid_crops_objects'])

        ### Transform abs_pose to rel_pose
        rel_pose = []
        for i in range(abs_pose.shape[0]):
            rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i,:], pos1=abs_pose[0,:]))

        item = {}
        item['pose'] = torch.from_numpy(np.asarray(rel_pose)).float()
        item['abs_pose'] = torch.from_numpy(abs_pose).float()
        item['ego_grid_crops_spatial'] = ego_grid_crops_spatial # already torch.float32
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_spatial
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial # Long tensor, int64
        item['gt_grid_crops_objects'] = gt_grid_crops_objects # Long tensor, int64


        if self.img_segm:

            if self.finetune:
                item['images'] = torch.from_numpy(ep['images']) # T x 3 x H x W # images are already pre-processed
                item['gt_segm'] = torch.from_numpy(ep['ssegs']).type(torch.int64) # T x 1 x H x W
                item['depth_imgs'] = torch.from_numpy(ep['depth_imgs']) # T x 1 x H x W
            else:
                ep_file_imgSegm = ep_file.replace(self.episodes_dir, self.episodes_imgSegm_dir)
                ep_imgSegm = np.load(ep_file_imgSegm)
                pred_ego_crops_sseg = torch.from_numpy(ep_imgSegm['pred_ego_crops_sseg'])
                item['pred_ego_crops_sseg'] = pred_ego_crops_sseg

        return item


# Dataloader only for training the img segmentation (i.e. loading only relevant data) that inherits from HabitatDataOffline
class HabitatDataImgSegm(HabitatDataOffline):

    def __init__(self, options, config_file, store=False):
        super().__init__(options, config_file, img_segm=False)
        self.store = store


    def __getitem__(self, idx):
        # Load from the pre-stored objnav training episodes
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        item={}
        item['images'] = torch.from_numpy(ep['images']) # T x 3 x H x W # images are already pre-processed
        item['gt_segm'] = torch.from_numpy(ep['ssegs']).type(torch.int64) # T x 1 x H x W
        item['depth_imgs'] = torch.from_numpy(ep['depth_imgs'])

        if self.store:
            item['filename'] = ep_file

        return item


## Loads the simulator and episodes separately to enable per_scene collection of data
class HabitatDataScene(Dataset):

    def __init__(self, options, config_file, scene_id, existing_episode_list=[]):
        self.scene_id = scene_id

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        ## Load episodes of scene_id
        ep_file_path = options.root_path + options.episodes_root + cfg.DATASET.SPLIT + "/content/" + self.scene_id + ".json.gz"
        with gzip.open(ep_file_path, "rt") as fp:
            self.scene_data = json.load(fp)
        self.number_of_episodes = len(self.scene_data["episodes"])

        self.success_distance = cfg.TASK.SUCCESS.SUCCESS_DISTANCE

        ## Dataloader params
        self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.spatial_labels = options.n_spatial_classes
        self.object_labels = options.n_object_classes
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.crop_size = (options.crop_size, options.crop_size)
        self.img_size = (options.img_size, options.img_size)
        self.img_segm_size = (options.img_segm_size, options.img_segm_size)
        self.normalize = True
        self.pixFormat = 'NCHW'
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

        self.episode_len = options.episode_len
        self.truncate_ep = options.truncate_ep

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]

        self.occ_from_depth = options.occ_from_depth
        self.occupancy_height_thresh = options.occupancy_height_thresh

        # Build 3D transformation matrices
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_size[0]), np.linspace(1,-1,self.img_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        K = np.array([
            [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
        # create the points2D containing all image coordinates
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_size[0]-1, self.img_size[0]), np.linspace(0, self.img_size[1]-1, self.img_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_size[0],self.img_size[1]), y.reshape(1,self.img_size[0],self.img_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        episode = self.scene_data['episodes'][idx]

        len_shortest_path = len(episode['shortest_paths'][0])
        objectgoal = episode['object_category']

        if len_shortest_path > 50: # skip that episode to avoid memory issues
            return None
        if len_shortest_path < self.episode_len+1:
            return None

        if idx in self.existing_episode_list:
            print("Episode", idx, 'already exists!')
            return None

        scene = self.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        # convert the labels to the reduced set of categories
        instance_id_to_label_id_3 = instance_id_to_label_id.copy()
        instance_id_to_label_id_objects = instance_id_to_label_id.copy()
        for inst_id in instance_id_to_label_id.keys():
            curr_lbl = instance_id_to_label_id[inst_id]
            instance_id_to_label_id_3[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
            instance_id_to_label_id_objects[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]

        # if truncated, run episode only up to the chosen step start_ind+episode_len
        if self.truncate_ep:
            start_ind = random.randint(0, len_shortest_path-self.episode_len-1)
            episode_extend = start_ind+self.episode_len
        else:
            episode_extend = len_shortest_path

        # imgs, depth, and ssegs stored here are (128,128) rather than the simulator's self.img_size:(256,256)
        # because they are going to be used during image segmentation training
        imgs = torch.zeros((episode_extend, 3, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)
        depth_imgs = torch.zeros((episode_extend, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)
        ssegs_objects = torch.zeros((episode_extend, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)

        ssegs_3 = torch.zeros((episode_extend, 1, self.img_size[1], self.img_size[0]), dtype=torch.float32, device=self.device)

        points2D, local3D, abs_poses, rel_poses, action_seq, agent_height = [], [], [], [], [], []

        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
        sim_obs = self.sim.get_sensor_observations()
        observations = self.sim._sensor_suite.get_observations(sim_obs)


        for i in range(episode_extend):
            img = observations['rgb'][:,:,:3]
            depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)

            depth = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth = depth.squeeze(0).permute(1,2,0)

            if self.cfg_norm_depth:
                depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)            

            semantic = observations['semantic']
            semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode='nearest').int()
            semantic = semantic.squeeze(0).squeeze(0)

            # visual and 3d info
            imgData = utils.preprocess_img(img, cropSize=self.img_segm_size, pixFormat=self.pixFormat, normalize=self.normalize)
            local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)

            ssegData = np.expand_dims(semantic.cpu().numpy(), 0).astype(float) # 1 x H x W
            ssegData_3 = np.vectorize(instance_id_to_label_id_3.get)(ssegData.copy()) # convert instance ids to category ids
            ssegData_objects = np.vectorize(instance_id_to_label_id_objects.get)(ssegData.copy()) # convert instance ids to category ids

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())

            imgs[i,:,:,:] = imgData
            depth_resize = F.interpolate(depth_obsv.clone(), size=self.img_segm_size, mode='nearest')
            depth_imgs[i,:,:,:] = depth_resize.squeeze(0)
            ssegs_3[i,:,:,:] = torch.from_numpy(ssegData_3).float()
            ssegData_resize = F.interpolate(torch.from_numpy(ssegData_objects).unsqueeze(0).float(), size=self.img_segm_size, mode='nearest')
            ssegs_objects[i,:,:,:] = ssegData_resize.squeeze()

            abs_poses.append(agent_pose)
            agent_height.append(y_height)
            points2D.append(self.points2D_step)
            local3D.append(local3D_step)

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
            rel_poses.append(rel)

            # explicitly clear observation otherwise they will be kept in memory the whole time
            observations = None

            action_id = episode['shortest_paths'][0][i]
            if action_id==None:
                break
            observations = self.sim.step(action_id)


        pose = torch.from_numpy(np.asarray(rel_poses)).float()
        abs_pose = torch.from_numpy(np.asarray(abs_poses)).float()

        # Create the ground-projected grids
        if self.occ_from_depth:
            ego_grid_sseg_3 = map_utils.est_occ_from_depth(local3D, grid_dim=self.grid_dim, cell_size=self.cell_size, 
                                                    device=self.device, occupancy_height_thresh=self.occupancy_height_thresh)
        else:
            ego_grid_sseg_3 = map_utils.ground_projection(self.points2D, local3D, ssegs_3, sseg_labels=self.spatial_labels, grid_dim=self.grid_dim, cell_size=self.cell_size)

        ego_grid_crops_3 = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.crop_size)
        step_ego_grid_3 = map_utils.get_acc_proj_grid(ego_grid_sseg_3, pose, abs_pose, self.crop_size, self.cell_size)
        step_ego_grid_crops_3 = map_utils.crop_grid(grid=step_ego_grid_3, crop_size=self.crop_size)
        # Get cropped gt
        gt_grid_crops_spatial = map_utils.get_gt_crops(abs_pose, self.pcloud, self.label_seq_spatial, agent_height,
                                                            self.grid_dim, self.crop_size, self.cell_size)
        gt_grid_crops_objects = map_utils.get_gt_crops(abs_pose, self.pcloud, self.label_seq_objects, agent_height,
                                                            self.grid_dim, self.crop_size, self.cell_size)

        item = {}
        item['images'] = imgs
        item['depth_imgs'] = depth_imgs
        item['ssegs'] = ssegs_objects
        item['episode_id'] = idx
        item['scene_id'] = self.scene_id
        item['abs_pose'] = abs_pose
        item['ego_grid_crops_spatial'] = ego_grid_crops_3
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_3
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial
        item['gt_grid_crops_objects'] = gt_grid_crops_objects
        return item
