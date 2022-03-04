
""" Entry point for training
"""

from train_options import TrainOptions
from trainer import Trainer
from trainer_finetune import TrainerFinetune
from trainer_active import ActiveTrainer
from trainer_segm import TrainerSegm
from tester import NavTester, SemMapTester
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError


def active_training(options, scene_id):
    trainer = ActiveTrainer(options, scene_id)
    trainer.train_active()

def nav_testing(options, scene_id):
    tester = NavTester(options, scene_id)
    tester.test_navigation()


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = TrainOptions().parse_args()

    if options.is_train:

        if options.active_training:
            scene_ids = options.scenes_list

            # Create iterables for map function
            n = len(scene_ids)
            options_list = [options] * n
            args = [*zip(options_list, scene_ids)]

            # isolate OpenGL context in each simulator instance
            with Pool(processes=options.gpu_capacity) as pool:
                pool.starmap(active_training, args)

        elif options.img_segm_training:
            trainer = TrainerSegm(options)
            trainer.train()
        elif options.finetune:
            trainer = TrainerFinetune(options)
            trainer.train()
        else:
            trainer = Trainer(options)
            trainer.train()

    else:
        if options.sem_map_test:
            tester = SemMapTester(options)
            tester.test_semantic_map()
        else:
            scene_ids = options.scenes_list

            # Create iterables for map function
            n = len(scene_ids)
            options_list = [options] * n
            args = [*zip(options_list, scene_ids)]

            # isolate OpenGL context in each simulator instance
            with Pool(processes=options.gpu_capacity) as pool:
                pool.starmap(nav_testing, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
