from pytorch_utils.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0,
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='~/semantic_grid/logs', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=1000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_nav_batch_size', type=int, default=1, help='Batch size during navigation test')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')

        # Dataset related options
        train.add_argument('--data_type', dest='data_type', type=str, default='train',
                            choices=['train', 'val'],
                            help='Choose which dataset to run on, valid only with --use_store')
        train.add_argument('--dataset_percentage', dest='dataset_percentage', type=float, default=1.0,
                            help='percentage of dataset to be used during training for ensemble learning')

        train.add_argument('--summary_steps', type=int, default=1000,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=5000,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=30000,
                           help='Chekpoint saving frequency')

        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')


        train.add_argument('--is_train', dest='is_train', action='store_true',
                            help='Define whether training or testing mode')

        train.add_argument('--config_train_file', type=str, dest='config_train_file',
                            default='configs/my_objectnav_mp3d_train.yaml',
                            help='path to habitat dataset train config file')

        self.parser.add_argument('--config_test_file', type=str, dest='config_test_file',
                                default='configs/my_objectnav_mp3d_test.yaml',
                                help='path to test config file -- to be used with our episodes')

        self.parser.add_argument('--config_val_file', type=str, dest='config_val_file',
                                default='configs/my_objectnav_mp3d_val.yaml',
                                help='path to habitat dataset val config file')

        self.parser.add_argument('--ensemble_dir', type=str, dest='ensemble_dir', default=None,
                                help='Path containing the experiments comprising the ensemble')

        self.parser.add_argument('--n_spatial_classes', type=int, default=3, dest='n_spatial_classes',
                                help='number of categories for spatial prediction')
        self.parser.add_argument('--n_object_classes', type=int, default=27, dest='n_object_classes',
                                choices=[18,27], help='number of categories for object prediction')
        self.parser.add_argument('--grid_dim', type=int, default=384, dest='grid_dim',
                                    help='Semantic grid size (grid_dim, grid_dim)')
        self.parser.add_argument('--cell_size', type=float, default=0.1, dest="cell_size",
                                    help='Physical dimensions (meters) of each cell in the grid')
        self.parser.add_argument('--crop_size', type=int, default=64, dest='crop_size',
                                    help='Size of crop around the agent')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)


        train.add_argument('--map_loss_scale', type=float, default=1.0, dest='map_loss_scale')
        train.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale')


        train.add_argument('--init_gaussian_weights', dest='init_gaussian_weights', action='store_true',
                            help='initializes the model weights from gaussian distribution')


        train.set_defaults(shuffle_train=True, shuffle_test=True)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=20000)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=0.0002)
        optimizer_options.add_argument('--beta1', type=float, default=0.5)

        model_options = self.parser.add_argument_group('Model')

        model_options.add_argument('--with_img_segm', dest='with_img_segm', default=False, action='store_true',
                                    help='uses the img segmentation pre-trained model during training or testing')
        model_options.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default=None,
                                    help='job path that contains the pre-trained img segmentation model')


        self.parser.add_argument('--sem_map_test', dest='sem_map_test', default=False, action='store_true')

        ## Hyperparameters for planning in test navigation


        self.parser.add_argument('--max_steps', type=int, dest='max_steps', default=500,
                                  help='Maximum steps for each test episode')

        self.parser.add_argument('--steps_after_plan', type=int, dest='steps_after_plan', default=20,
                                 help='how many times to use the local policy before selecting long-term-goal and replanning')

        self.parser.add_argument('--stop_dist', type=float, dest='stop_dist', default=0.5,
                                 help='decision to stop distance')

        self.parser.add_argument('--turn_angle', dest='turn_angle', type=int, default=10,
                                help='angle to rotate left or right in degrees for habitat simulator')
        self.parser.add_argument('--forward_step_size', dest='forward_step_size', type=float, default=0.25,
                                help='distance to move forward in meters for habitat simulator')

        self.parser.add_argument('--save_nav_images', dest='save_nav_images', action='store_true',
                                 help='Keep track and store maps during navigation testing')

        self.parser.add_argument('--a_1', type=float, dest='a_1', default=0.1,
                                 help='hyperparameter for choosing long-term-goal')
        self.parser.add_argument('--a_2', type=float, dest='a_2', default=1.0,
                                 help='hyperparameter for choosing long-term-goal')

        # options relating to active training (using scenes dataloader)
        self.parser.add_argument('--ensemble_size', type=int, dest='ensemble_size', default=4)

        self.parser.add_argument('--active_training', dest='active_training', default=False, action='store_true')
        self.parser.add_argument('--img_segm_training', dest='img_segm_training', default=False, action='store_true')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="~/")
        self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="habitat-api/data/datasets/objectnav/mp3d/v1/")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')

        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', default='mp3d_objnav_episodes_final/')
        self.parser.add_argument('--stored_imgSegm_episodes_dir', type=str, dest='stored_imgSegm_episodes_dir', default='mp3d_objnav_episodes_final_imgSegmOut/')

        self.parser.add_argument('--active_ep_save_dir', type=str, dest='active_ep_save_dir', default='mp3d_objnav_episodes_active/',
                                 help='used only during active training to store the episodes')
        self.parser.add_argument('--max_num_episodes', type=int, dest='max_num_episodes', default=1000,
                                help='how many episodes to collect per scene when running the active training')
        

        self.parser.add_argument('--model_number', type=int, dest='model_number', default=1,
                                help='only used when finetuning the model in the active training case - defines which model in ensemble to use')
        self.parser.add_argument('--finetune', dest='finetune', default=False, action='store_true',
                                help='Enable finetuning of an ensemble model')

        self.parser.add_argument('--uncertainty_type', type=str, dest='uncertainty_type', default='epistemic',
                                choices=['epistemic', 'entropy', 'bald'], help='how to estimate uncertainty in active training')

        self.parser.add_argument('--episode_len', type=int, dest='episode_len', default=10)
        self.parser.add_argument('--truncate_ep', dest='truncate_ep', default=False,
                                  help='truncate episode run in dataloader in order to do only the necessary steps, used in store_episodes_parallel')

        self.parser.add_argument('--occ_from_depth', dest='occ_from_depth', default=True, action='store_true',
                                help='if enabled, uses only depth to get the ground-projected egocentric grid')

        self.parser.add_argument('--local_policy_model', type=str, dest='local_policy_model', default='4plus',
                                choices=['2plus', '4plus'])

        self.parser.add_argument('--scenes_list', nargs='+')
        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)

        self.parser.add_argument('--test_set', type=str, dest='test_set', default='v3', choices=['v3','v5','v6','v7'],
                                help='which set of test episodes to use, each has different objects')


        self.parser.add_argument('--use_semantic_sensor', dest='use_semantic_sensor', default=False, action='store_true',
                                help='use simulators sensor instead of pretrained semantic segmentation net')

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                help='used when estimating occupancy from depth')


        self.parser.add_argument('--sem_thresh', dest='sem_thresh', type=float, default=0.75,
                                help='used to identify possible targets in the semantic map')