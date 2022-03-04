
from datasets.dataloader import HabitatDataOffline

def get_dataset_from_options(options, is_train=True):
    if is_train:
        return HabitatDataOffline(options, config_file=options.config_train_file, img_segm=options.with_img_segm)
    else:
        return HabitatDataOffline(options, config_file=options.config_val_file, img_segm=options.with_img_segm)

