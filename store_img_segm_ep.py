
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError
import numpy as np
from datasets.dataloader import HabitatDataImgSegm
import datasets.util.utils as utils
import os
import argparse
import torch
import torch.nn as nn
from models.img_segmentation import get_img_segmentor_from_options
import test_utils as tutils

# Run the trained img segmentation and store the ground-projected semantic predictions
# The predictions are used in the training of the ensemble map predictors
# Loads the episodes from "mp3d_objnav_episodes_final" 

class Params(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val', 'test'])
        
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)
        self.parser.add_argument('--crop_size', type=int, dest='crop_size', default=64)
        self.parser.add_argument('--cell_size', type=float, dest='cell_size', default=0.1)
        self.parser.add_argument('--n_object_classes', type=int, dest='n_object_classes', default=27)

        self.parser.add_argument('--dataset_percentage', type=float, dest='dataset_percentage', default=1)

        self.parser.add_argument('--scenes_list', nargs='+')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="~/")
        self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="habitat-api/data/datasets/objectnav/mp3d/v1/")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')
        self.parser.add_argument('--episodes_save_dir', type=str, dest='episodes_save_dir', default="mp3d_objnav_episodes_final_imgSegmOut/")
        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', default='mp3d_objnav_episodes_tmp/')

        self.parser.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default=None,
                                    help='job path that contains the pre-trained img segmentation model')

        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=1)
        self.parser.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale') # needed in model def


def store_episodes(options, config_file, model, current_scene, device):

    data = HabitatDataImgSegm(options, config_file=config_file, store=True)

    # Keep only the episodes corresponding to current_scene from the dataloader
    file_list = data.episodes_file_list
    scene_file_list = [x.split('/')[-1].split('.')[0].split('_')[-1] for x in file_list]
    ind = [i for i, e in enumerate(scene_file_list) if e == current_scene]
    new_list = np.asarray(file_list)[ np.asarray(ind) ]
    data.episodes_file_list = new_list.tolist()
    data.number_of_episodes = len(new_list)
    

    # Build 3D transformation matrices from img segm ground projection
    xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1,options.img_segm_size[0]), np.linspace(1,-1,options.img_segm_size[1])), device=device)
    xs = xs.reshape(1,options.img_segm_size[0],options.img_segm_size[1])
    ys = ys.reshape(1,options.img_segm_size[0],options.img_segm_size[1])
    K = np.array([
        [1 / np.tan(data.hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(data.hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])
    inv_K = torch.tensor(np.linalg.inv(K), device=device)
    # create the points2D containing all image coordinates
    x, y = torch.tensor(np.meshgrid(np.linspace(0, options.img_segm_size[0]-1, options.img_segm_size[0]), 
                                                np.linspace(0, options.img_segm_size[1]-1, options.img_segm_size[1])), device=device)
    xy_img = torch.cat((x.reshape(1,options.img_segm_size[0],options.img_segm_size[1]), y.reshape(1,options.img_segm_size[0],options.img_segm_size[1])), dim=0)
    points2D_step = xy_img.reshape(2, -1)
    points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2

    print(len(data))

    for i in range(len(data)):
        ex = data[i]

        scene_id = ex['filename'].split('/')[-1].split('.')[0].split('_')[-1]
        episode_save_dir = options.root_path + options.scenes_dir  + options.episodes_save_dir + options.split + "/" + scene_id + "/"
        filepath = episode_save_dir + ex['filename'].split('/')[-1]

        if scene_id!=current_scene:
            continue
        if os.path.exists(filepath):
            continue
        
        batch = {'images':ex['images'].to(device).unsqueeze(0),
                 'gt_segm':ex['gt_segm'].to(device).unsqueeze(0),
                 'depth_imgs':ex['depth_imgs'].to(device).unsqueeze(0)}

        pred_ego_crops_sseg = utils.run_img_segm(model=model, 
                                                input_batch=batch, 
                                                object_labels=options.n_object_classes, 
                                                crop_size=options.crop_size, 
                                                cell_size=options.cell_size,
                                                xs=xs,
                                                ys=ys,
                                                inv_K=inv_K,
                                                points2D_step=points2D_step)

        pred_ego_crops_sseg = pred_ego_crops_sseg.squeeze(0).cpu() # T x C x cH x cW

        if not os.path.exists(episode_save_dir):
            os.makedirs(episode_save_dir)

        np.savez_compressed(filepath, pred_ego_crops_sseg=pred_ego_crops_sseg)



if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    options = Params().parser.parse_args()

    print("options:")
    for k in options.__dict__.keys():
        print(k, options.__dict__[k])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if options.split=="val":
        config_file = "configs/my_objectnav_mp3d_val.yaml"
    elif options.split=="train":
        config_file = "configs/my_objectnav_mp3d_train.yaml"
    else:
        config_file = "configs/my_objectnav_mp3d_test.yaml"

    # Load the pre-trained img segmentation model
    img_segmentor = get_img_segmentor_from_options(options)
    img_segmentor = img_segmentor.to(device)
    
    # Needed only for models trained with multi-gpu setting
    img_segmentor = nn.DataParallel(img_segmentor)

    latest_checkpoint = tutils.get_latest_model(save_dir=options.img_segm_model_dir)
    print("Loading image segmentation checkpoint", latest_checkpoint)
    
    checkpoint = torch.load(latest_checkpoint)
    img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
    img_segmentor.eval()

    options.crop_size = (options.crop_size, options.crop_size)
    options.img_segm_size = (options.img_segm_size, options.img_segm_size)

    scene_ids = options.scenes_list

    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    config_files = [config_file] * n
    img_segmentor_list = [img_segmentor] * n
    device_list = [device] * n

    args = [*zip(options_list, config_files, img_segmentor_list, scene_ids, device_list)]

    with Pool(processes=options.gpu_capacity) as pool:

        pool.starmap(store_episodes, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")