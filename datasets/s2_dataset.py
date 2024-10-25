from ..spatiotemporal.torch_datasets.base_dataset import BaseDataset
from ..spatiotemporal.mods import ZeroPixelsS2, CategoricalCloudMaps, CloudfreeArea
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

    # This is only used to get well-looking images
    VEGETATION_TILES = (
        ("ROIs2017", 117),
        ("ROIs1868", 17)
    )

    def __init__(
            self,
            dataset_manager,
            min_target_area=0.95,
            min_inputs_area=0.5,
            rescale=True,
            filter_bands=True,
            clip_inputs=False,
            include_index=False,
            include_cloudmaps=True,
            include_cloud_cover=False
    ):
        self.include_cloudmaps = include_cloudmaps
        super().__init__(dataset_manager)

        self.min_target_area = min_target_area
        self.min_inputs_area = min_inputs_area
        self.cloud_probability_threshold = self.dataset_manager.cloud_probability_threshold
        self.rescale = rescale
        self.filter_bands = filter_bands
        self.clip_inputs = clip_inputs
        self.include_index = include_index
        self.include_cloud_cover = include_cloud_cover

        self.build_dataset()
        self.filter_data()

    def initialize_data(self):

        if self.include_cloudmaps:
            return self.manager.data[["S2", "S2CLOUDMAP", "CLOUDFREEAREA"]]
        else:
            return self.manager.data[["S2", "CLOUDFREEAREA"]]

    def build_dataset(self):

        combined_dataset = self + self.shift(-1) + self.shift(-2) + self.shift(-3)
        self.data = combined_dataset.data

    def filter_data(self):

        # remove NaN rows after applying shifts
        self.dropna()

        # remove target images that are too cloudy
        if self.min_target_area is not None:
            target_is_cloudfree = self.data.loc[:, (0, "CLOUDFREEAREA")] > self.min_target_area
            self.data = self.data[target_is_cloudfree]

        # remove rows where even the least cloudy image is too cloudy
        if self.min_inputs_area is not None:
            input_not_too_cloudy = (self.data.loc[:, ([-1, -2, -3], "CLOUDFREEAREA")] > self.min_inputs_area).any(axis=1)
            self.data = self.data[input_not_too_cloudy]

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        result = {}

        for index, filepath in sample[:, "S2"].items():
            image = self.utils.read_tif_fast(filepath)
            if self.rescale:
                image = self.utils.rescale_s2(image, clip=self.clip_inputs)
            if self.filter_bands:
                image = image[self.BANDS]
            result[f"S2_t{index}"] = image.astype(self.NP_DTYPE)

        if self.include_cloudmaps:
            for index, filepath in sample[:, "S2CLOUDMAP"].items():
                image = self.utils.read_tif_fast(filepath)
                image = (image < self.cloud_probability_threshold * 100).astype(self.NP_DTYPE)
                result[f"S2CLOUDMASK_t{index}"] = image[np.newaxis, ...]

        if self.include_cloud_cover:
            for index, cloud_free_area in sample[:, "CLOUDFREEAREA"].items():
                result[f"CLOUDFREEAREA_t{index}"] = cloud_free_area

        if self.include_index:
            result["index"] = idx

        return result
