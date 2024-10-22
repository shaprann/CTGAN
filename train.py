from .spatiotemporal.SEN12MSCRTS import DatasetManager
from .spatiotemporal.mods import ZeroPixelsS2, CategoricalCloudMaps, CloudfreeArea
from .datasets.s2_dataset import CTGAN_S2_Dataset
from .model.CTGAN import CTGAN_Generator, CTGAN_Discriminator
from .utils import fixed_seed, set_requires_grad, LSGANLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data._utils.collate import default_collate

from tqdm import tqdm
from os.path import join
from skimage.io import imsave
import pandas as pd
import numpy as np
import warnings
import argparse
import os

warnings.filterwarnings("ignore")


class Trainer:

    DTYPE = torch.float32

    def __init__(self, opt, dataset: CTGAN_S2_Dataset):

        self.opt = opt
        self.experiment_name = opt.experiment_name
        self.device = torch.device(f'cuda:{self.opt.gpu_id}')
        fixed_seed(self.opt.manual_seed)

        self.train_dataset = dataset.subset(split="train", inplace=False)
        self.val_dataset = dataset.subset(split="val", inplace=False)
        del dataset

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.workers,
            persistent_workers=False,
            prefetch_factor=1,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.opt.val_batch_size,
            num_workers=1,
            persistent_workers=False,
            prefetch_factor=1,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
        self.val_generator = self.infinite(self.val_loader)  # get new batch using next(self.val_generator)

        self.GEN = CTGAN_Generator(image_size=self.opt.image_size).to(device=self.device, dtype=self.DTYPE)
        self.DIS = CTGAN_Discriminator().to(device=self.device, dtype=self.DTYPE)
        self.optim_GEN = torch.optim.AdamW(self.GEN.parameters(), lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
        self.optim_DIS = torch.optim.AdamW(self.DIS.parameters(), lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)

        self.noise = self.opt.label_noise
        self.criterionGAN = LSGANLoss().to(dtype=self.DTYPE, device=self.device)
        self.criterionL1 = torch.nn.L1Loss().to(dtype=self.DTYPE, device=self.device)
        self.criterionMSE = nn.MSELoss().to(dtype=self.DTYPE, device=self.device)

        self.scheduler_GEN = CosineAnnealingLR(self.optim_GEN, T_max=opt.n_epochs, eta_min=1e-6)
        self.scheduler_DIS = CosineAnnealingLR(self.optim_DIS, T_max=opt.n_epochs, eta_min=1e-6)

        self._init_folders()
        self.image_batch = self._init_example_images()
        self.writer = self._init_writer()
        self._maybe_write_images_at_init()

        self.step = 0
        self.psnr_max = 0.
        self.ssim_max = 0.
        self.save_now_with_postfix = None

        if self.opt.load_checkpoint:
            self.load_from_checkpoint()

    @staticmethod
    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def batched_tensor_to_image(tensor):

        def factor_int(n):
            """Source: https://stackoverflow.com/a/57503963"""
            a = int(np.floor(np.sqrt(n)))
            b = int(np.ceil(n / float(a)))
            return a, b

        grid_shape = factor_int(tensor.size(0))

        if not grid_shape[0] * grid_shape[1] == tensor.size(0):
            tensor = torch.cat([
                tensor,
                0.5 * torch.ones(
                    grid_shape[0] * grid_shape[1] - tensor.size(0),
                    *tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device
                )
            ])

        tensor = tensor.moveaxis(1, -1)
        tensor = tensor.unflatten(0, grid_shape)
        tensor = tensor.moveaxis(0, 2)
        tensor = tensor.flatten(0, 1)
        tensor = tensor.flatten(1, 2)
        return tensor.numpy()

    def _init_folders(self):

        self.path_to_logs = join(self.opt.path_to_logs, f'{self.experiment_name}_CTGAN')
        os.makedirs(self.path_to_logs, exist_ok=True)

        if self.opt.save_step is not None:
            self.path_to_checkpoints = join(self.opt.path_to_checkpoints, f'{self.experiment_name}_CTGAN')
            os.makedirs(self.path_to_checkpoints, exist_ok=True)

        if self.opt.image_step is not None:
            self.path_to_predictions = join(self.opt.path_to_predictions, f'{self.experiment_name}_CTGAN')
            os.makedirs(self.path_to_predictions, exist_ok=True)

    def _init_example_images(self):
        all_indices = self.val_dataset.data.index
        good_indices = all_indices[all_indices.droplevel(["patch", "timestep"]).isin(self.val_dataset.VEGETATION_TILES)]
        image_indices = np.random.RandomState(42).choice(a=good_indices, size=36)
        image_batch = default_collate([self.val_dataset[i] for i in image_indices])
        return image_batch

    def _init_writer(self):
        writer = SummaryWriter(self.path_to_logs)
        options_markdown = pd.DataFrame.from_dict(vars(self.opt), orient='index', columns=["Options"]).to_markdown()
        writer.add_text("Options", options_markdown, global_step=0)
        writer.add_text("Dataset Modifications", str(self.train_dataset.dataset_manager._modifications), global_step=0)
        return writer

    def _maybe_write_images_at_init(self):
        if self.opt.image_step is not None:
            print(f"Saving input and target images to {self.path_to_predictions}")
            self.write_images(self.image_batch["S2_t0"], postfix="TARGET")
            self.write_images(self.image_batch["S2_t-1"], postfix="INPUT_t-1")
            self.write_images(self.image_batch["S2_t-2"], postfix="INPUT_t-2")
            self.write_images(self.image_batch["S2_t-3"], postfix="INPUT_t-3")

    def train(self):

        for epoch in range(self.opt.n_epochs):

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} of {self.opt.n_epochs} - ", unit=" step"):

                loss_values = self.make_step(batch)

                self.write(loss_values)
                self.maybe_validate()
                self.maybe_save()
                self.maybe_write_images()

            self.scheduler_DIS.step()
            self.scheduler_GEN.step()

    def make_step(self, batch):

        self.GEN.train()
        self.DIS.train()
        loss_values = {}

        batch = {
            name: tensor.to(device=self.device, dtype=self.DTYPE, non_blocking=True)
            for name, tensor in batch.items()
        }
        inputs = [batch["S2_t-1"], batch["S2_t-2"], batch["S2_t-3"]]
        cloud_masks = [batch["S2CLOUDMASK_t-1"], batch["S2CLOUDMASK_t-2"], batch["S2CLOUDMASK_t-3"]]
        target = batch["S2_t0"]
        with torch.no_grad():
            discriminator_mask = self.DIS.get_receptive_field(batch["S2CLOUDMASK_t0"])

        """Forward Pass Generator"""

        prediction, cloud_mask_preds, aux_preds = self.GEN(inputs)

        """Update Discriminator"""

        set_requires_grad(self.DIS, True)
        self.optim_DIS.zero_grad()

        # Prediction
        inputs_with_prediction = torch.cat([*inputs, prediction], 1)
        inputs_with_prediction_DETACHED = inputs_with_prediction.detach()
        DIS_score_for_prediction = self.DIS(inputs_with_prediction_DETACHED)
        DIS_score_for_prediction = DIS_score_for_prediction[discriminator_mask]
        loss_values["DIS/loss_for_prediction"] = self.criterionGAN(DIS_score_for_prediction, False, self.noise)

        # Target
        inputs_with_target = torch.cat([*inputs, target], 1)
        DIS_score_for_target = self.DIS(inputs_with_target)
        DIS_score_for_target = DIS_score_for_target[discriminator_mask]
        loss_values["DIS/loss_for_target"] = self.criterionGAN(DIS_score_for_target, True, self.noise)

        # Combine loss and calculate gradients
        loss_values["DIS/LOSS"] = (loss_values["DIS/loss_for_prediction"] + loss_values["DIS/loss_for_target"]) * 0.5
        loss_values["DIS/LOSS"].backward()
        self.optim_DIS.step()

        """Update Generator"""

        self.optim_GEN.zero_grad()
        set_requires_grad(self.DIS, False)

        # First, G(A) should fake the discriminator
        DIS_score_for_prediction_ = self.DIS(inputs_with_prediction)
        loss_values["GEN/loss_against_dis"] = self.criterionGAN(DIS_score_for_prediction_, True, self.noise)

        # Second, G(A) = B
        loss_values["GEN/loss_L1"] = self.criterionL1(prediction, target) * self.opt.lambda_L1

        """Calculate loss for cloud predictions"""

        loss_values["GEN/loss_clouds"] = 0.0
        for mask_true, mask_pred in zip(cloud_masks, cloud_mask_preds):
            loss_values["GEN/loss_clouds"] += self.criterionMSE(mask_true, mask_pred)

        """Calculate auxiliary loss"""

        loss_values["GEN/loss_aux"] = 0.0
        if self.opt.aux_loss:
            for true_image, aux_prediction in zip(inputs, aux_preds):
                loss_values["GEN/loss_aux"] += self.criterionL1(true_image, aux_prediction)
            loss_values["GEN/loss_aux"] *= self.opt.lambda_aux

        """Calculate final loss"""

        loss_values["GEN/LOSS"] = (
                  loss_values["GEN/loss_against_dis"]
                + loss_values["GEN/loss_L1"]
                + loss_values["GEN/loss_clouds"]
                + loss_values["GEN/loss_aux"]
        )

        loss_values["GEN/LOSS"].backward()
        self.optim_GEN.step()

        self.step += 1

        return {name: value.detach() for name, value in loss_values.items()}

    def write(self, loss_values):
        for name, value in loss_values.items():
            self.writer.add_scalar(name, value, self.step)
        self.writer.flush()

    def maybe_save(self):

        if self.opt.save_step is not None:

            if self.step % self.opt.save_step == 0:
                print(f"\n Savind model {self.experiment_name} at step {self.step}\n")
                self.save(postfix=f"_step_{self.step}")
            elif self.save_now_with_postfix is not None:
                print(f"\n Savind model {self.experiment_name} at step {self.step} as {self.save_now_with_postfix}\n")
                self.save(postfix=self.save_now_with_postfix)

    def save(self, postfix):

        filename = f"CTGAN_{postfix}.checkpoint"
        target_filepath = join(self.path_to_checkpoints, filename)

        torch.save(
            {
                'step': self.step,
                'GEN': self.GEN.state_dict(),
                'DIS': self.DIS.state_dict(),
                'optim_GEN': self.optim_GEN.state_dict(),
                'optim_DIS': self.optim_DIS.state_dict(),
            },
            target_filepath
        )

    def load_from_checkpoint(self):

        checkpoint_filepath = self.opt.load_checkpoint

        print(f"\nLoading checkpoint:  {checkpoint_filepath}\n")

        checkpoint = torch.load(checkpoint_filepath, weights_only=True)
        self.step = checkpoint["step"]
        self.GEN.load_state_dict(checkpoint["GEN"])
        self.DIS.load_state_dict(checkpoint["DIS"])
        self.optim_GEN.load_state_dict(checkpoint["optim_GEN"])
        self.optim_DIS.load_state_dict(checkpoint["optim_DIS"])

    def maybe_validate(self):

        if self.step % self.opt.val_step == 0:
            batch = next(self.val_generator)
            loss_values = self.evaluate(batch)
            self.write(loss_values)

    def evaluate(self, batch):

        self.GEN.train()   # TODO: fix this!!! actually should be eval()
        self.DIS.train()
        loss_values = {}

        with torch.no_grad():

            batch = {
                name: tensor.to(device=self.device, dtype=self.DTYPE, non_blocking=True)
                for name, tensor in batch.items()
            }
            inputs = [batch["S2_t-1"], batch["S2_t-2"], batch["S2_t-3"]]
            cloud_masks = [batch["S2CLOUDMASK_t-1"], batch["S2CLOUDMASK_t-2"], batch["S2CLOUDMASK_t-3"]]
            target = batch["S2_t0"]
            discriminator_mask = self.DIS.get_receptive_field(batch["S2CLOUDMASK_t0"])

            """Evaluate Generator"""

            prediction, cloud_mask_preds, aux_preds = self.GEN(inputs)
            inputs_with_prediction = torch.cat([*inputs, prediction], 1)

            # First, GEN should deceive the discriminator
            DIS_score_for_prediction = self.DIS(inputs_with_prediction)
            loss_values["GEN/loss_against_dis"] = self.criterionGAN(DIS_score_for_prediction, True, noise=False)

            # Second, GEN should predict target image
            loss_values["GEN/loss_L1"] = self.criterionL1(prediction, target) * self.opt.lambda_L1

            """Evaluate cloud predictions"""

            loss_values["GEN/loss_clouds"] = 0.0
            for mask_true, mask_pred in zip(cloud_masks, cloud_mask_preds):
                loss_values["GEN/loss_clouds"] += self.criterionMSE(mask_true, mask_pred)

            """Evaluate auxiliary loss"""

            loss_values["GEN/loss_aux"] = 0.0
            if self.opt.aux_loss:
                for true_image, aux_prediction in zip(inputs, aux_preds):
                    loss_values["GEN/loss_aux"] += self.criterionL1(true_image, aux_prediction)
                loss_values["GEN/loss_aux"] *= self.opt.lambda_aux

            """Evaluate Discriminator"""

            # First, DIS should detect that prediction is fake
            loss_values["DIS/loss_for_prediction"] = self.criterionGAN(DIS_score_for_prediction, False, noise=False)

            # Second, GEN should detect that target image is real
            inputs_with_target = torch.cat([*inputs, target], 1)
            DIS_score_for_target = self.DIS(inputs_with_target)
            DIS_score_for_target = DIS_score_for_target[discriminator_mask]
            loss_values["DIS/loss_for_target"] = self.criterionGAN(DIS_score_for_target, True, noise=False)

            # Combine loss and calculate gradients
            loss_values["DIS/LOSS"] = (loss_values["DIS/loss_for_prediction"] + loss_values["DIS/loss_for_target"]) * 0.5

            """Calculate final loss"""

            loss_values["GEN/LOSS"] = (
                    loss_values["GEN/loss_against_dis"]
                    + loss_values["GEN/loss_L1"]
                    + loss_values["GEN/loss_clouds"]
                    + loss_values["GEN/loss_aux"]
            )

            return {"VAL_" + name: value.detach() for name, value in loss_values.items()}

    def maybe_write_images(self):

        if self.opt.image_step is not None:

            if self.step % self.opt.image_step == 0:

                for mode in ["train", "eval"]:

                    predictions_dict = self.run_inference(self.image_batch, mode=mode)
                    postfix = f"MODE_{mode.upper()}_prediction_step_{self.step}"
                    self.write_images(predictions_dict["prediction"], postfix)

    def write_images(self, batched_tensor, postfix):

        batched_tensor = batched_tensor.cpu().detach()
        image = self.batched_tensor_to_image(batched_tensor)
        image = image[:, :, :3]
        image = (image.clip(0.0, 1.0) * 255).astype(np.uint8)
        image_name = f"{self.experiment_name}_{postfix}.png"
        imsave(join(self.path_to_predictions, image_name), image)

    def run_inference(self, batch, mode="eval"):

        if mode == "eval":
            self.GEN.eval()
        else:
            self.GEN.train()  # TODO: change this! This should be eval()

        with torch.no_grad():

            inputs = [
                batch["S2_t-1"].to(device=self.device, dtype=self.DTYPE, non_blocking=True),
                batch["S2_t-2"].to(device=self.device, dtype=self.DTYPE, non_blocking=True),
                batch["S2_t-3"].to(device=self.device, dtype=self.DTYPE, non_blocking=True)
            ]

            prediction, cloud_mask_preds, aux_preds = self.GEN(inputs)

            return {
                "prediction": prediction,
                "CLOUDMASK_PRED_t-1": cloud_mask_preds[0],
                "CLOUDMASK_PRED_t-2": cloud_mask_preds[1],
                "CLOUDMASK_PRED_t-3": cloud_mask_preds[2],
                "AUX_PRED_t-1": aux_preds[0],
                "AUX_PRED_t-2": aux_preds[1],
                "AUX_PRED_t-3": aux_preds[2],
            }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    """Path"""
    parser.add_argument("--path_to_logs", type=str, required=True, help="Path to save logs")                   #
    parser.add_argument("--path_to_checkpoints", type=str, default="", help="Path to save logs")  #
    parser.add_argument("--path_to_predictions", type=str, default="", help="Path to save logs")  #
    parser.add_argument("--experiment_name", type=str, required=True, help="Prefix for the tensorboard writer")
    parser.add_argument("--load_checkpoint", type=str, default='', help="path to the model")

    """Parameters"""
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--val_step", type=int, default=50, help="Validate after this number of batches")                #
    parser.add_argument("--save_step", type=int, default=None, help="Save after this number of batches")
    parser.add_argument("--image_step", type=int, default=None, help="Save image after this number of batches")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")   
    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")       #
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")                                     #
    parser.add_argument("--val_batch_size", type=int, default=16, help="size of the batches")  #
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_aux', type=float, default=50.0, help='weight for aux loss')
    parser.add_argument("--image_size", type=int, default=256, help="crop size")
    parser.add_argument("--aux_loss", action='store_true', help="whether use auxiliary loss(1/0)")
    parser.add_argument("--label_noise", action='store_true', help="whether to add noise on the label of gan training")

    """base_options"""
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--manual_seed", type=int, default=2022, help="random_seed you want")

    opt = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    print(opt)

    # Initialize dataset
    root_dir = '/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS'
    cloud_maps_dir = "/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS_cloud_maps"
    dataset_manager = DatasetManager(
        root_dir=root_dir,
        cloud_maps_dir=cloud_maps_dir
    )
    dataset_manager.load_from_file()
    ZeroPixelsS2(dataset_manager).apply_modification(verbose=True)
    CategoricalCloudMaps(dataset_manager).apply_modification(verbose=True)
    CloudfreeArea(dataset_manager).apply_modification(verbose=True)

    dataset = CTGAN_S2_Dataset(dataset_manager).subset(s1_resampled=True, inplace=True)

    trainer = Trainer(opt=opt, dataset=dataset)
    trainer.train()
