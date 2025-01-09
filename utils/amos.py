import os
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AmosDataset(Dataset):
    """
    Amos dataset class for PyTorch.
    """

    def __init__(
        self,
        path,
        split: Literal["train", "val", "test"],
        transform=None,
        index_range=None,
        slice_range=None,
        only_labeled=False,
        # TODO file name mask
    ):
        folder_map = {
            "train": "Tr",
            "val": "Va",
            "test": "Ts",
        }

        print(f"Loading Amos {split} data")
        self.images_folder = path + f"/images{folder_map[split]}/"
        self.labels_folder = path + f"/labels{folder_map[split]}/"

        # TODO file name mask

        images_df = pd.DataFrame(os.listdir(self.images_folder), columns=["image"])
        labels_df = pd.DataFrame(os.listdir(self.labels_folder), columns=["label"])

        # filter image not in range
        if index_range:
            assert len(index_range) == 2, "index_range must be a list of two integers"
            index_range = range(index_range[0], index_range[1] + 1)
            index_mask = images_df["image"].apply(
                lambda x: self._filter_filename(x, index_range, filter_type="index")
            )
        else:
            index_mask = [True] * len(images_df)
        if slice_range:
            assert len(slice_range) == 2, "slice_range must be a list of two integers"
            slice_range = range(slice_range[0], slice_range[1] + 1)
            slice_mask = images_df["image"].apply(
                lambda x: self._filter_filename(x, slice_range, filter_type="slice")
            )
        else:
            slice_mask = [True] * len(images_df)

        combined_mask = np.logical_and(index_mask, slice_mask)
        images_df = images_df[combined_mask]
        labels_df = labels_df[combined_mask]

        # filter if not at least one pixel is labeled
        if only_labeled:
            print(f"Filtering only labeled images ...")
            label_mask = labels_df["label"].apply(
                lambda label: np.array(Image.open(self.labels_folder + label)).sum() > 0
            )
            images_df = images_df[label_mask]
            labels_df = labels_df[label_mask]

        assert len(images_df) == len(
            labels_df
        ), "Number of images and labels do not match"

        self.dataset = pd.merge(images_df, labels_df, left_on="image", right_on="label")
        self.transform = transform

        print(f"Loaded {len(self.dataset)} {split} images")
        print(
            f"Transforms: {transform}, index_range: {index_range}, slice_range: {slice_range}, only_labeled: {only_labeled}"
        )

    def _filter_filename(self, filename, range, filter_type="index"):
        """
        Filters filenames and keeps only those with indices in the given range.
        Assumes the filename format is: "amos_XXXX_sYYY.png" (XXXX is the index)
        """
        # Extract the index part
        try:
            if filter_type == "index":
                index = int(filename.split("_")[1])  # Extract the XXXX part
            elif filter_type == "slice":
                index = int(filename.split("s")[1])  # Extract the YYY part
            else:
                raise ValueError("filter_type must be either 'index' or 'slice'")
            return index in range
        except (IndexError, ValueError):
            return False  # Skip files that don't match the format

    def __getitem__(self, index):
        """
        Returns the image and label at the given index.
        """
        img = Image.open(self.images_folder + self.dataset["image"][index])
        label = Image.open(self.labels_folder + self.dataset["label"][index])

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.dataset)
