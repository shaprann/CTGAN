from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict
import os
from os import listdir
from os.path import join, isfile, basename
import tifffile as tiff
import numpy as np
import torch


class Sen12_TS(Dataset):

    # correction_factors = 1 / np.array([4, 2.6, 6.5, 1])[:, np.newaxis, np.newaxis]

    def __init__(self, opt, mode="train"):

        dataset_hierarchy = ["ROI", "patch", "timestep"]
        dataset_content = [
            "S2_filename",
            "S2_abspath",
            "S2_cloudmask_filename",
            "S2_cloudmask_abspath",
            "S1_filename",
            "S1_abspath",
            "S2_cloud_percentage",
            "S2_date",
            "S1_date"
        ]
        on_disc_hierarchy = ["ROI", "modality", "timestep", "patch"]
        filename_contains = ["modality_lowercase", "patch", "timestep", "date"]

        self.root = opt.root
        self.mode = mode
        self.images = {}

        self.cloudfree_images_list = pd.read_csv('sen12_good_data.csv')
        self.cloudfree_images_list = self.cloudfree_images_list.set_index(["ROI", "tile", "patch", "timestep"])
        self.cloudfree_images_list = self.cloudfree_images_list.sort_index()

        self.image_paths: pd.DataFrame = self.find_available_images()

        self.image_triplets = []
        self.targets = []
        self.create_image_triplets()

    def find_available_images(self):

        for path, directories, filenames in os.walk(self.root):

            for filename in filenames:

                # skip everything except for .tif images
                if not filename.endswith(".tif"):
                    continue

                # select only sentinel 2 filenames
                if not filename.startswith("s2"):
                    continue

                self.add_image_by_filename(filename)

    def add_image_by_filename(self, image_filename):

        modality, roi, tile, timestep, date, patch = self.parse_filename(image_filename)
        filepath = join(self.root, roi, str(tile), modality.capitalize(), str(timestep), image_filename)
        if not isfile(filepath):
            print("Failed: ", filepath)
            return

        roi_tile = join(roi, str(tile))
        if roi_tile not in self.images:
            self.images[roi_tile] = {}
        if patch not in self.images[roi_tile]:
            self.images[roi_tile][patch] = {}
        if timestep not in self.images[roi_tile][patch]:
            self.images[roi_tile][patch][timestep] = {}

        self.images[roi_tile][patch][timestep]["filename"] = image_filename
        self.images[roi_tile][patch][timestep]["filepath"] = filepath
        self.images[roi_tile][patch][timestep]["date"] = date

    @staticmethod
    def parse_filename(filename):
        filename = filename[:-4]  # remove .tif extension
        modality, roi, tile, _, timestep, date, _, patch = filename.split("_")
        return modality, roi, int(tile), int(timestep), date, int(patch)

    @staticmethod
    def adjust_sen12ms_cr_ts_single(image):

        SEN12MS_CR_TS_cloudfree_mean = np.array([1426.3226, 1192.9644, 1092.9033, 1024.9343, 1213.6475,
                                                 1859.4617, 2151.784, 2070.3547, 2333.43, 606.3736,
                                                 13.210275, 1818.0392, 1200.058])
        SEN12MS_CR_TS_cloudfree_std = np.array([218.31575, 310.5481, 374.8059, 522.1565, 491.71747,
                                                652.8539, 789.4757, 792.5027, 873.326, 250.2111,
                                                9.534574, 809.68256, 670.8796])
        Sen12_MTC_mean = np.array([768.19086, 735.3265, 478.61, 2564.3748])
        Sen12_MTC_std = np.array([587.6501, 370.52225, 301.14856, 849.8937])

        corrected_sen_12_ts_cloudfree = image.copy()

        # blue
        corrected_sen_12_ts_cloudfree[1] = (
                (
                        (
                                corrected_sen_12_ts_cloudfree[1]
                                - SEN12MS_CR_TS_cloudfree_mean[1]
                        )
                        * Sen12_MTC_std[2]
                        / SEN12MS_CR_TS_cloudfree_std[1]
                )
                + Sen12_MTC_mean[2]
        )

        # green
        corrected_sen_12_ts_cloudfree[2] = (
                (
                        (
                                corrected_sen_12_ts_cloudfree[2]
                                - SEN12MS_CR_TS_cloudfree_mean[2]
                        )
                        * Sen12_MTC_std[1]
                        / SEN12MS_CR_TS_cloudfree_std[2]
                )
                + Sen12_MTC_mean[1]
        )

        # red
        corrected_sen_12_ts_cloudfree[3] = (
                (
                        (
                                corrected_sen_12_ts_cloudfree[3]
                                - SEN12MS_CR_TS_cloudfree_mean[3]
                        )
                        * Sen12_MTC_std[0]
                        / SEN12MS_CR_TS_cloudfree_std[3]
                )
                + Sen12_MTC_mean[0]
        )

        # nir
        corrected_sen_12_ts_cloudfree[7] = (
                (
                        (
                                corrected_sen_12_ts_cloudfree[7]
                                - SEN12MS_CR_TS_cloudfree_mean[7]
                        )
                        * Sen12_MTC_std[3]
                        / SEN12MS_CR_TS_cloudfree_std[7]
                )
                + Sen12_MTC_mean[3]
        )

        return corrected_sen_12_ts_cloudfree

    def create_image_triplets(self):

        for roi_tile in self.images:
            for patch in self.images[roi_tile]:
                timesteps = sorted(self.images[roi_tile][patch])
                for timestep in timesteps:
                    # if we can extract three images in a row, add triplet to list
                    triplet = []
                    try:
                        if (roi_tile.split(os.sep)[0], int(roi_tile.split(os.sep)[1]), patch, timestep+4) in self.cloudfree_images_list.index:
                            for i in range(3):
                                triplet.append(self.images[roi_tile][patch][timestep + i]["filepath"])
                            target = self.images[roi_tile][patch][timestep + 4]["filepath"]
                            self.image_triplets.append(triplet)
                            self.targets.append(target)
                    except KeyError:
                        pass

    def __getitem__(self, index):

        triplet = self.image_triplets[index]
        target = self.targets[index]
        modality, roi, tile, _, _, patch = self.parse_filename(basename(triplet[0]))
        timesteps = [self.parse_filename(basename(filepath))[3] for filepath in triplet]

        images = [self.image_read(filepath) for filepath in triplet]
        target_image = self.image_read(target)
        prediction_filename = "_".join([
            modality,
            roi,
            str(tile),
            "ImgNo",
            str(timesteps[0]),
            str(timesteps[1]),
            str(timesteps[2]),
            "patch",
            str(patch)
        ]) + ".tif"

        return images, target_image, prediction_filename

    def __len__(self):
        return len(self.image_triplets)

    def image_read(self, image_path):
        img = tiff.imread(image_path)  # reads 13 bands
        img = (img / 1.0).transpose((2, 0, 1))
        img = self.adjust_sen12ms_cr_ts_single(img)
        img = img[[3, 2, 1, 7]]  # select bands [red, green, blue, NIR]

        if self.mode == 'train':
            pass

        image = torch.from_numpy((img.copy())).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image