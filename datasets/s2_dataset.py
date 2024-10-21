from .. import spatiotemporal
from spatiotemporal.torch_datasets.base_dataset import BaseDataset
from spatiotemporal.mods import ZeroPixelsS2, CategoricalCloudMaps, CloudfreeArea
import numpy as np
import torch


class CTGAN_S2_Dataset(BaseDataset):

    requirements = (
        ZeroPixelsS2.__name__,
        CategoricalCloudMaps.__name__,
        CloudfreeArea.__name__
    )
    BANDS = [3, 2, 1, 7]
    DTYPE = torch.float32
    NP_DTYPE = np.float32

    def __init__(
            self,
            dataset_manager,
            min_target_area=0.95,
            min_inputs_area=0.5,
            clip_inputs=False
    ):

        super().__init__(dataset_manager)
        self.min_target_area = min_target_area
        self.min_inputs_area = min_inputs_area
        self.cloud_probability_threshold = self.dataset_manager.cloud_probability_threshold
        self.clip_inputs = clip_inputs

        combined_dataset = self + self.shift(-1) + self.shift(-2) + self.shift(-3)
        self.data = combined_dataset.data
        del combined_dataset

        # remove NaN rows after applying shifts
        self.dropna()

        # remove target images that are too cloudy
        target_is_cloudfree = self.data.loc[:, (0, "CLOUDFREEAREA")] > self.min_target_area
        self.data = self.data[target_is_cloudfree]

        # remove rows where even the least cloudy image is too cloudy
        input_is_somewhat_cloudfree = (self.data.loc[:, ([-1, -2, -3], "CLOUDFREEAREA")] > self.min_inputs_area).any(
            axis=1)
        self.data = self.data[input_is_somewhat_cloudfree]

    def initialize_data(self):
        return self.manager.data[["S2", "S2CLOUDMAP", "CLOUDFREEAREA"]]

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        result = {}

        for index, filepath in sample[:, "S2"].items():
            image = self.utils.read_tif_fast(filepath)
            image = self.utils.rescale_s2(image, clip=self.clip_inputs)
            image = image[self.BANDS]
            result[f"S2_t{index}"] = image.astype(self.NP_DTYPE)

        for index, filepath in sample[:, "S2CLOUDMAP"].items():
            image = self.utils.read_tif_fast(filepath)
            image = (image < self.cloud_probability_threshold * 100).astype(self.NP_DTYPE)
            result[f"S2CLOUDMAP_t{index}"] = image[np.newaxis, ...]

        return result
