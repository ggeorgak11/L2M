
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError
import numpy as np
from datasets.dataloader import HabitatDataScene
import datasets.util.utils as utils
import os
import argparse
import torch
import random
import json


class Params(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val', 'test'])

        self.parser.add_argument('--grid_dim', type=int, dest='grid_dim', default=384)
        self.parser.add_argument('--crop_size', type=int, dest='crop_size', default=64)
        self.parser.add_argument('--cell_size', type=float, dest='cell_size', default=0.1)
        self.parser.add_argument('--turn_angle', type=int, dest='turn_angle', default=30)
        self.parser.add_argument('--forward_step_size', type=float, dest='forward_step_size', default=0.25)
        self.parser.add_argument('--n_object_classes', type=int, dest='n_object_classes', default=27)
        self.parser.add_argument('--n_spatial_classes', type=int, dest='n_spatial_classes', default=3)

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)

        self.parser.add_argument('--max_num_episodes', dest='max_num_episodes', type=int, default=2500)

        self.parser.add_argument('--episode_len', type=int, dest='episode_len', default=10)
        self.parser.add_argument('--truncate_ep', dest='truncate_ep', default=True,
                                  help='truncate episode run in dataloader in order to do only the necessary steps')
        self.parser.add_argument('--occ_from_depth', dest='occ_from_depth', default=True, action='store_true',
                                help='if enabled, uses only depth to get the ground-projected egocentric grid')

        self.parser.add_argument('--scenes_list', nargs='+')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="~/")
        
        self.parser.add_argument('--episodes_path', type=str, dest='episodes_path', default="habitat-api/data/datasets/objectnav/mp3d/")
        self.parser.add_argument('--ep_set', type=str, dest='ep_set', default='v1', choices=['v1','v3','v5'])
        self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="")
        
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')
        self.parser.add_argument('--episodes_save_dir', type=str, dest='episodes_save_dir', default="mp3d_objnav_episodes_tmp/")

        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)


def store_episodes(options, config_file, scene_id):

    episode_save_dir = options.root_path + options.scenes_dir  + options.episodes_save_dir + options.split + "/" + scene_id + "/"
    if not os.path.exists(episode_save_dir):
        os.makedirs(episode_save_dir)

    existing_episode_list = os.listdir(episode_save_dir) # keep track of previously saved episodes

    options.episodes_root = options.episodes_path + options.ep_set + '/'

    data = HabitatDataScene(options, config_file, scene_id=scene_id, existing_episode_list=existing_episode_list)

    print(len(data))

    ep_count = len(existing_episode_list)
    for i in range(len(data)):
        ex = data[i]

        if ep_count >= options.max_num_episodes:
            break

        if ex is None:
            continue

        ep_count+=1

        scene_id = ex['scene_id']
        episode_id = ex['episode_id']
        abs_pose = ex['abs_pose']
        ego_grid_crops_spatial = ex['ego_grid_crops_spatial'].cpu()
        step_ego_grid_crops_spatial = ex['step_ego_grid_crops_spatial'].cpu()
        gt_grid_crops_spatial = ex['gt_grid_crops_spatial'].cpu()
        gt_grid_crops_objects = ex['gt_grid_crops_objects'].cpu()

        images = ex['images'].cpu()
        ssegs = ex['ssegs'].cpu()
        depth_imgs = ex['depth_imgs'].cpu()


        if options.truncate_ep: # assumes that the maps were created only up to the desired step
            abs_pose = abs_pose[-options.episode_len:,:]
            ego_grid_crops_spatial = ego_grid_crops_spatial[-options.episode_len:,:,:,:]
            step_ego_grid_crops_spatial = step_ego_grid_crops_spatial[-options.episode_len:,:,:,:]
            gt_grid_crops_spatial = gt_grid_crops_spatial[-options.episode_len:,:,:,:]
            gt_grid_crops_objects = gt_grid_crops_objects[-options.episode_len:,:,:,:]
            images = images[-options.episode_len:,:,:,:]
            ssegs = ssegs[-options.episode_len:,:,:,:]
            depth_imgs = depth_imgs[-options.episode_len:,:,:,:]
        else: # assumes episode was run until its end
            total_episode_len = ego_grid_crops_spatial.shape[0]
            ind = random.randint(0, total_episode_len-options.episode_len-1)
            abs_pose = abs_pose[ind:ind+options.episode_len,:]
            ego_grid_crops_spatial = ego_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            step_ego_grid_crops_spatial = step_ego_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            gt_grid_crops_spatial = gt_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            gt_grid_crops_objects = gt_grid_crops_objects[ind:ind+options.episode_len,:,:,:]
            images = images[ind:ind+options.episode_len,:,:,:]
            ssegs = ssegs[ind:ind+options.episode_len,:,:,:]
            depth_imgs = depth_imgs[ind:ind+options.episode_len,:,:,:]
        
        print('Saving episode', ep_count, 'of id', episode_id, 'scene', scene_id)

        filepath = episode_save_dir+'ep_'+str(ep_count)+'_'+str(episode_id)+"_"+scene_id
        np.savez_compressed(filepath+'.npz',
                            abs_pose=abs_pose,
                            ego_grid_crops_spatial=ego_grid_crops_spatial,
                            step_ego_grid_crops_spatial=step_ego_grid_crops_spatial,
                            gt_grid_crops_spatial=gt_grid_crops_spatial,
                            gt_grid_crops_objects=gt_grid_crops_objects,
                            images=images,
                            ssegs=ssegs,
                            depth_imgs=depth_imgs
                            )


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = Params().parser.parse_args()

    print("options:")
    for k in options.__dict__.keys():
        print(k, options.__dict__[k])


    save_path = options.root_path + options.scenes_dir + options.episodes_save_dir + options.split + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'options.json'), "w") as f:
        json.dump(vars(options), f, indent=4)

    if options.split=="val":
        config_file = "configs/my_objectnav_mp3d_val.yaml"
    elif options.split=="train":
        config_file = "configs/my_objectnav_mp3d_train.yaml"
    else:
        config_file = "configs/my_objectnav_mp3d_test.yaml"

    scene_ids = options.scenes_list

    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    config_files = [config_file] * n
    args = [*zip(options_list, config_files, scene_ids)]

    with Pool(processes=options.gpu_capacity) as pool:

        pool.starmap(store_episodes, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
