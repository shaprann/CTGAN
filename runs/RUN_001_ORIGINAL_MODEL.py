from ..spatiotemporal.SEN12MSCRTS import DatasetManager
from ..spatiotemporal.mods import ZeroPixelsS2, CategoricalCloudMaps, CloudfreeArea
from ..datasets.s2_dataset import CTGAN_S2_Dataset
from ..train import Trainer
from types import SimpleNamespace
import argparse
import os

EXPERIMENT_NAME = os.path.basename(__file__)[:-3]
print(f"RUNNING EXPERIMENT {EXPERIMENT_NAME}")

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0, help="gpu id")
parser_opt = parser.parse_args()

path_to_logs = "/LOCAL2/shvl/logs/CTGAN"
path_to_checkpoints = "/LOCAL2/shvl/checkpoints/CTGAN"
path_to_predictions = "/LOCAL2/shvl/predictions/CTGAN"
root_dir = '/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS'
cloud_maps_dir = "/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS_cloud_maps"

dataset_manager = DatasetManager(
    root_dir=root_dir,
    cloud_maps_dir=cloud_maps_dir,
    cloud_probability_threshold=0.35
)
dataset_manager.load_from_file()
ZeroPixelsS2(dataset_manager).apply_modification(verbose=True)
CategoricalCloudMaps(dataset_manager).apply_modification(verbose=True)
CloudfreeArea(dataset_manager).apply_modification(verbose=True)

opt = SimpleNamespace(
    experiment_name=EXPERIMENT_NAME,
    n_epochs=100,
    gan_mode='lsgan',
    lr=5e-4,
    workers=32,
    batch_size=4,
    val_batch_size=32,
    lambda_L1=100.0,
    lambda_aux=50.0,
    image_size=256,
    aux_loss=True,
    label_noise=True,
    gpu_id=parser_opt.cuda,
    manual_seed=99,
    save_step=5000,
    image_step=40,
    val_step=20,
    path_to_logs=path_to_logs,
    path_to_checkpoints=path_to_checkpoints,
    path_to_predictions=path_to_predictions,
    load_checkpoint=None,
)

dataset = CTGAN_S2_Dataset(dataset_manager).subset(s1_resampled=True, inplace=True)
trainer = Trainer(opt=opt, dataset=dataset)
trainer.train()
