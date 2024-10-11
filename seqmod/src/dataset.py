"""Dataset.py module containing the custom dataset classes used in the wind speed prediction and storm image generation."""

import os
import glob

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from torchvision import transforms, io
import torchvision.transforms as transforms

from src.data_source import (
    get_subfolder_names_from_root,
    get_wind_speed_data,
    create_sequence_data,
)


class StormWindSpeedSequencedDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        storms_to_exclude: list = [],
        storm_name: str | None = None,
        numeric_cols: list | None = None,
        time_column: str = "relative_time",
        dt: int | None = 1800,
        input_sequence_length: int = 4,
        target_length: int = 1,
        data_split: str | None = "train",
        data_splits: tuple[float, float, float] = (0.8, 0.2, 0.2),
        transform: transforms.Compose | None = None,
        random_seed: int = 42,
    ):
        """Initialisation."""
        # inputs
        self.data_root_dir = data_root_dir
        self.storms_to_exclude = storms_to_exclude
        self.storm_name = storm_name
        # get list of storm name(s) to use.
        self.storm_names = self.get_storm_names()
        self.numeric_cols = numeric_cols
        self.time_column = time_column
        self.dt = dt
        self.data_split = data_split
        self.data_splits = data_splits
        self.transform = transform
        # sequence info
        self.input_sequence_length = input_sequence_length
        self.target_length = target_length
        # seed
        self.random_seed = random_seed

        self.df = get_wind_speed_data(
            data_dir=self.data_root_dir,
            storm_names=self.storm_names,
            numeric_cols=self.numeric_cols,
            time_column=self.time_column,
            dt=self.dt,
        )

        # create sequenced wind speed dataset
        self.df_sequence = create_sequence_data(
            df=self.df,
            storm_names=self.storm_names,
            input_length=self.input_sequence_length,
            target_length=self.target_length,
        )

        self.data = self.split_data()

    def get_storm_names(self):
        """"""
        if self.storm_name is not None:
            storm_names = [self.storm_name]
        else:
            storm_names = get_subfolder_names_from_root(root_dir=self.data_root_dir)
            if self.storms_to_exclude is not None:
                storm_names = [
                    s for s in storm_names if s not in self.storms_to_exclude
                ]

        return storm_names

    def split_data(self):
        """"""
        if self.data_split is None:
            # ability to return all of the data if needed i.e inference.
            return self.df_sequence
        else:
            train_val, df_test = train_test_split(
                self.df_sequence,
                train_size=self.data_splits[0],
                random_state=self.random_seed,
                shuffle=False,
            )

            df_train, df_val = train_test_split(
                train_val,
                test_size=self.data_splits[1],
                random_state=self.random_seed,
                shuffle=False,
            )
            if self.data_split == "train":
                return df_train
            elif self.data_split == "val":
                return df_val
            elif self.data_split == "test":
                return df_test
            else:
                raise ValueError("Need to input either 'train', 'val' or None.")

    def __len__(self):
        """"""
        return len(self.data)

    def __getitem__(self, idx):
        """"""
        item = self.data.iloc[idx].copy()

        # suppliamentary data
        storm = item[:1].values[0]
        ocean = item[1:2].values[0]

        # get speed and time features
        time_arr = np.array(
            item[2 : (self.input_sequence_length + self.target_length) + 2],
            dtype=np.int16,
        ).reshape(
            -1, 1
        )  # time
        speed_arr = np.array(
            item[(self.input_sequence_length + self.target_length) + 2 :],
            dtype=np.float32,
        ).reshape(
            -1, 1
        )  # wind speed

        # make into a torch tensor and separate out input and target (after transform)
        # _data = torch.from_numpy(np.array([time_arr, speed_arr]))
        _data = torch.from_numpy(
            np.concatenate([time_arr, speed_arr], axis=1).reshape(
                self.input_sequence_length + self.target_length, -1
            )
        )
        if self.transform:
            _data = self.transform(_data)

        input, target = _data[: -self.target_length, :], _data[-self.target_length :, :]
        # input dimensions (sequence_length, num_features/input_size)

        return (storm, ocean), input, target


class StormImageSequencedDataset(Dataset):
    def __init__(
        self,
        image_root_dir: str,
        image_file_type: str = "jpg",
        storm_name: str | None = None,
        input_sequence_length: int = 10,
        target_sequence_length: int = 1,
        data_split: str = "train",
        split_pct: tuple[float, float] = (0.8, 0.2),  # train, val
        transform: transforms.Compose | None = None,
        random_seed: int = 42,
    ):
        """Initialisation."""

        # images info
        self.image_root_dir = image_root_dir
        self.image_file_type = image_file_type
        self.storm_name = storm_name
        # sequence info
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        # data info
        self.data_split = data_split
        self.split_pct = split_pct
        self.random_seed = random_seed
        self.transform = transform

        # get desired storms
        self.storm_names = self.get_storm_names()

        # collate all storm images by storm
        self.df_storm_images = self.get_storm_image_data(img_file_type="jpg")

        # create sequenced data of all storms (no cross-overs)
        self.df_sequence = self.create_storm_sequence_data()

        # get data for chosen split
        self.data = self.split_data()

    def get_storm_names(self):
        """"""
        if self.storm_name is not None:
            storm_names = [self.storm_name]
        else:
            storm_names = get_subfolder_names_from_root(root_dir=self.data_root_dir)
            if self.storms_to_exclude is not None:
                storm_names = [
                    s for s in storm_names if s not in self.storms_to_exclude
                ]

        return storm_names

    def split_data(self):
        """"""
        df_train, df_val = train_test_split(
            self.df_sequence,
            train_size=self.split_pct[0],
            random_state=self.random_seed,
            shuffle=True,
        )
        if self.data_split == "train":
            return df_train
        elif self.data_split == "val":
            return df_val
        elif self.data_split is None:
            # ability to return all of the data if needed i.e inference.
            return self.df_sequence
        else:
            raise ValueError("Need to input either 'train', 'val' or None.")

    def get_storm_image_data(self, img_file_type: str = "jpg"):
        """"""
        storm_images_dict = (
            {}
        )  # store storm images (values) under each storm name (key)
        for storm in self.storm_names:
            images = []
            storm_img_path = os.path.join(
                self.image_root_dir, storm, f"*.{img_file_type}"
            )
            for img in sorted(glob.glob(storm_img_path)):
                # store each image for storm to add to dict
                images.append(img)

            storm_images_dict[storm] = images

        # unpack the storm image dictionary
        df_imgs = []
        for k, v in storm_images_dict.items():
            df_imgs.append(pd.DataFrame(data=v, columns=[k]))

        df_storm_images = pd.concat(objs=[df for df in df_imgs], axis=1)

        return df_storm_images

    def create_storm_sequence_data(self):
        """"""
        # create the sequence df
        storm_seq = []
        for _storm_name in self.df_storm_images.columns:
            # subset to only storm and make an array of the image file paths
            df_img_paths = self.df_storm_images[_storm_name]
            storm_img_arr = np.array(df_img_paths.dropna().values)
            for i in range(0, len(storm_img_arr) - self.input_sequence_length):
                # sliding window of size = input_sequence length+1
                # window contains [input sequence (in order), target sequence]
                storm_seq.append(
                    storm_img_arr[
                        i : i
                        + (self.input_sequence_length + self.target_sequence_length)
                    ]
                )

        # create sliding window labels for the columns
        columns = [
            f"t{i}"
            for i in range(
                -1 * (self.input_sequence_length - 1), 1 + self.target_sequence_length
            )
        ]

        df_sequence = pd.DataFrame(data=storm_seq, columns=columns)

        return df_sequence

    def __len__(self):
        """"""
        return len(self.data)

    def __getitem__(self, idx):
        """"""
        # TODO: make sure this can handle >1 sequence length
        img_tensors = [
            io.read_image(img_path) for img_path in self.data.iloc[idx].copy()
        ]

        # normalise the pixels (/max(pixel))
        img_tensors = torch.stack(img_tensors) / 255

        if self.transform:
            img_tensors = self.transform(img_tensors)

        # make sure pixels are in [0, 1]
        assert (img_tensors.max() <= 1) & (img_tensors.min() >= 0)

        image_seq, target_seq = (
            img_tensors[: self.input_sequence_length],
            img_tensors[self.input_sequence_length :],
        )

        return image_seq, target_seq
