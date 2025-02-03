import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class AmosDataset(Dataset):
    """
    Amos dataset class for PyTorch.
    """

    def __init__(
        self,
        img_dir,
        seg_dir,
        split: Literal["train", "val", "test"],
        transform=None,
        index_range=None,
        slice_range=None,
        only_labeled=False,
        # TODO file name mask
        img_name_filter=None,
        # TODO load_images_as_np_arrays or tensor with arbitrary shape
        load_as_tensor=False,  # if True, load img and seg as tensor. expect .pt, using torch.load
    ):
        self.images_folder = os.path.join(img_dir, split)
        self.labels_folder = os.path.join(seg_dir, split)

        print(f"Loading Amos {split} data")

        # TODO file name mask

        # load images, do recursive search for all images in multiple folders
        images_list = []
        labels_list = []
        for root, _, files in os.walk(self.images_folder, followlinks=True):
            print("Including images from", os.path.relpath(root, self.images_folder))
            for file in files:
                images_list.append(
                    os.path.join(os.path.relpath(root, self.images_folder), file)
                )
        for root, _, files in os.walk(self.labels_folder, followlinks=True):
            print("Including labels from", os.path.relpath(root, self.labels_folder))
            for file in files:
                labels_list.append(
                    os.path.join(os.path.relpath(root, self.labels_folder), file)
                )

        # Assume that the images and labels are named the same
        # items_list = list(set(images_list).intersection(set(labels_list)))
        images_df = pd.DataFrame(images_list, columns=["image"])
        labels_df = pd.DataFrame(labels_list, columns=["label"])
        print(f"{len(images_df)} images and {len(labels_df)} labels")
        images_df = images_df[images_df["image"].isin(labels_df["label"])]
        labels_df = labels_df[labels_df["label"].isin(images_df["image"])]
        print(f"{len(images_df)} images and {len(labels_df)} labels after intersection")

        # filter image not in range
        if index_range:
            assert len(index_range) == 2, "index_range must be a list of two integers"
            index_range = range(index_range[0], index_range[1] + 1)
            index_mask = images_df["image"].apply(
                lambda x: self._filter_filename(x, index_range, filter_type="index")
            )
        else:
            index_mask = [True] * len(images_df)
        # filter slice not in range
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

        # filter if not at least one pixel is labeled. NOT RECOMMENDED
        if only_labeled:
            print(f"Filtering only labeled images ...")
            label_mask = labels_df["label"].apply(
                lambda label: np.array(Image.open(self.labels_folder + label)).sum() > 0
            )
            images_df = images_df[label_mask]
            labels_df = labels_df[label_mask]

        print(f"{len(images_df)} images and {len(labels_df)} labels after masking")

        assert len(images_df) == len(
            labels_df
        ), "Number of images and labels do not match"

        self.dataset = pd.merge(images_df, labels_df, left_on="image", right_on="label")
        self.load_as_tensor = load_as_tensor
        self.transform = transform

        if img_name_filter is not None:
            self.dataset = self.dataset[
                self.dataset["image"].isin(img_name_filter)
            ].reset_index(drop=True)

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

        # if load np
        # F.interpolate(
        #                         torch.tensor(np.load(image)).unsqueeze(0).float(),
        #                         size=(config.image_size, config.image_size),
        #                     ).squeeze()

        img_path = os.path.join(self.images_folder, self.dataset["image"][index])
        seg_path = os.path.join(self.labels_folder, self.dataset["label"][index])

        if self.load_as_tensor:
            img = torch.load(img_path)
            label = torch.load(seg_path)
        else:
            img = Image.open(img_path)
            label = Image.open(seg_path)

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return {"images": img, "images_target": label}

    def __len__(self):
        return len(self.dataset)
