#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from abc import ABC, abstractmethod
from typing import *
from dataclasses import dataclass
import irec.constants as const
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import numpy as np
from pathlib import Path
from shutil import unpack_archive, move
from urllib.request import urlretrieve
import seaborn as sns

from irec.data.split_data import split_stratified
from irec.utils.pandas_utils import has_columns
from irec.utils.utils import set_color, create_directory
import logging
import pandas as pd
from irec.constants import *

logger = logging.getLogger(__name__)


class Dataset:
    """Base class to manage datasets.

    This class is used by inherited classes for each specific dataset, providing basic functionality to
    - download a dataset from a URL
    - split the dataset
    - save and load datasets
    - preprocess a dataset

    The default download path of ~/rsys_datasets can be changed by setting the DATASETS_PATH environment variable,
    and each dataset will be downloaded to a subdirectory within this path.
    """

    @classmethod
    def __init_subclass__(
        cls,
        name: str,
        directory_name: str,
        url: str,
        url_archive_format: Optional[str],
        expected_files: List[str],
        description: str,
        source: str,
        data_subdirectory_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Used to set class variables during the class definition of derived classes and generate customised docs.
        NOTE: this is not compatible with python's ABC abstract base class, so this class derives from object."""
        cls.name = name
        cls.directory_name = directory_name
        cls.url = url
        cls.url_archive_format = url_archive_format
        cls.expected_files = expected_files
        cls.description = description
        cls.source = source
        cls.data_subdirectory_name = data_subdirectory_name

        if url_archive_format is None and len(expected_files) != 1:
            raise ValueError(
                "url_archive_format is None, which requires a single expected_file, found: {expected_files!r}"
            )

        # auto generate documentation
        if cls.__doc__ is not None:
            raise ValueError(
                "DatasetLoader docs are automatically generated and should be empty"
            )
        cls.__doc__ = f"{cls.description}\n\nFurther details at: {cls.source}"

        super().__init_subclass__(**kwargs)  # type: ignore

    def __init__(self) -> None:
        # basic check since this is effectively an abstract base class, and derived classes should have set name
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} can't be instantiated directly, please use a derived class"
            )
        self._create_data_directory()

        # initialized when load method is called
        self.dataset = None
        self.users_num = None
        self.items_num = None
        self.interactions_num = None
        self.density = None
        self.k_core = None

    @staticmethod
    def _all_datasets_directory() -> str:
        """Return the path of the base directory which contains subdirectories for each dataset."""
        return os.path.expanduser(os.path.join("~", const.DATASETS_PATH))

    @property
    def base_directory(self) -> str:
        """str: The full path of the directory containing this dataset."""
        return os.path.join(self._all_datasets_directory(), self.directory_name)

    @property
    def data_directory(self) -> str:
        """str: The full path of the directory containing the data content files for this dataset."""
        if self.data_subdirectory_name is None:
            return self.base_directory
        else:
            return os.path.join(self.base_directory, self.data_subdirectory_name)

    def _create_data_directory(self) -> None:
        data_dir = self.data_directory
        create_directory(data_dir)

    def _resolve_path(self, filename: str) -> str:
        """Convert dataset relative file names to their full path on filesystem"""
        return os.path.join(self.data_directory, filename)

    def _missing_files(self) -> List[str]:
        """Returns a list of files that are missing"""
        return [
            file
            for file in self.expected_files
            if not os.path.isfile(self._resolve_path(file))
        ]

    def _is_downloaded(self) -> bool:
        """Returns true if the expected files for the dataset are present"""
        return len(self._missing_files()) == 0

    def _delete_existing_files(self) -> None:
        """Delete the files for this dataset if they already exist"""
        for file in self.expected_files:
            try:
                os.remove(self._resolve_path(file))
                logger.info(
                    set_color(
                        f"{self.name} Raw files for dataset: {self.name} removed.",
                        "pink",
                    )
                )
            except OSError:
                pass

    def download(self, ignore_cache: Optional[bool] = False) -> None:
        """
        Download the dataset (if not already downloaded)

        Args:
            ignore_cache (bool, optional): Ignore a cached dataset and force a re-download.

        Raises:
            FileNotFoundError: If the dataset is not successfully downloaded.
        """

        if ignore_cache:
            self._delete_existing_files()  # remove any existing dataset files to ensure we re-download

        if ignore_cache or not self._is_downloaded():
            logger.info(
                set_color(
                    f"{self.name} dataset downloading to {self.base_directory} from {self.url}",
                    "yellow",
                )
            )
            temporary_filename, _ = urlretrieve(self.url)
            if self.url_archive_format is None:
                # not an archive, so the downloaded file is our data and just needs to be put into place
                self._create_data_directory()
                move(temporary_filename, self._resolve_path(self.expected_files[0]))
            else:
                # an archive to unpack.  The folder is created by unpack_archive - therefore the
                # directory_name for this dataset must match the directory name inside the archive file
                unpack_archive(
                    temporary_filename,
                    self._all_datasets_directory(),
                    self.url_archive_format,
                )
            # verify the download
            missing_files = self._missing_files()
            if missing_files:
                missing = ", ".join(missing_files)
                raise FileNotFoundError(
                    f"{self.name} dataset failed to download file(s): {missing} to {self.data_directory}"
                )
        else:
            logger.info(
                set_color(f"{self.name} dataset is already downloaded", "yellow")
            )

    @abstractmethod
    def _preprocess_dataset(self, k_core) -> pd.DataFrame:
        """Abstract method should be implemented by each dataset to preprocess raw files to a common dataset format

        - Remap user and item ids to make them contiguous
        - Remove users and items with less than k interaction (k_core)

        Args:
            k_core (int): minimum number of interactions that a user or item should have

        Note:
            The format of data inside the dataframe should be:
            | user_id | item_id

        Returns:
            pd.DataFrame: dataset
        """
        pass

    def preprocess_dataset(self, k_core: int = 10):
        """Preprocess and save the dataset according to self._preprocess_dataset()"""
        dataset = self._preprocess_dataset(k_core=k_core)

        # create the splits folder
        preprocessed_dir = os.path.join(self.data_directory, "preprocessed")
        Path(preprocessed_dir).mkdir(exist_ok=True)
        # create the split specific folder using split_name arg
        save_dir = os.path.join(preprocessed_dir, f"k_core_{k_core}")
        Path(save_dir).mkdir(exist_ok=True)
        save_path = os.path.join(save_dir, "data.csv")
        dataset.to_csv(save_path, index=False)
        logger.info(set_color(f"Preprocessed dataset saved in: {save_path}", "yellow"))

    def dataset_statistics(self, df):
        assert has_columns(df, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])

        users = df[DEFAULT_USER_COL].unique()
        items = df[DEFAULT_ITEM_COL].unique()
        self.users_num = len(users)
        self.items_num = len(items)
        self.interactions_num = len(df)
        self.density = round(len(df) / (len(users) * len(items)), 5)
        logger.info(
            set_color(
                f"Dataset statistics:\n"
                f"users num: {self.users_num}\n"
                f"items num: {self.items_num}\n"
                f"interactions num: {self.interactions_num}\n"
                f"density: {self.density}\n",
                "blue",
            )
        )

    def user_item_degree_distr(self, df, plot: bool = False):
        assert has_columns(df, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])

        # def power_law(x, a, c, c0):
        #     return c0 + (1 / (x ** a)) * c
        #
        # def fit_power_law(x, y):
        #     popt, pcov = curve_fit(
        #         power_law,
        #         x,
        #         y,
        #         p0=(0.5, 1e-2, 0),
        #         bounds=([0.1, 1e-3, -10], [0.9, 1, 10]),
        #         maxfev=2000,
        #     )
        #     return popt

        def power_law(x, a, b):
            """
            y = ax^b
            """
            return a * np.power(x, b)

        # user degree fit
        user_degree = sorted(df.groupby([DEFAULT_USER_COL]).count().values.squeeze())[
            ::-1
        ]
        norm_user_degree = np.array(user_degree) / max(user_degree)
        u_x = np.arange(len(user_degree)) + 1

        item_degree = sorted(df.groupby([DEFAULT_ITEM_COL]).count().values.squeeze())[
            ::-1
        ]
        norm_item_degree = np.array(item_degree) / max(item_degree)
        i_x = np.arange(len(item_degree)) + 1

        # user_degree = df.groupby(DEFAULT_USER_COL).count()
        # item_degree = df.groupby(DEFAULT_ITEM_COL).count()

        if plot:
            # users
            plt.plot(u_x, norm_user_degree, color="b")
            # item
            # plt.plot(i_x, norm_item_degree, color="r")
            plt.legend(labels=["user", "item"])

            pars, cov = curve_fit(
                f=power_law,
                xdata=u_x,
                ydata=norm_user_degree,
                p0=[1, -0.5],
                bounds=(-np.inf, np.inf),
            )

            plt.plot(u_x, power_law(u_x, *pars), color="g")

            plt.legend(labels=["user", "item", "app_user"])

            plt.show()
        # return popt_user, popt_item

    def load(self, k_core: int = 10):
        """Load the dataset"""
        preprocessed_dir = os.path.join(self.data_directory, "preprocessed")
        save_dir = os.path.join(preprocessed_dir, f"k_core_{k_core}")
        load_path = os.path.join(save_dir, "data.csv")
        # check preprocessed dataset exsist
        assert os.path.exists(
            load_path
        ), f"preprocessed dataset does not exsist in {load_path}"
        dataset = pd.read_csv(load_path)
        logger.info(set_color(f"Load {self.name} with kcore {k_core}\n", "yellow"))
        self.dataset = dataset
        self.dataset_statistics(dataset)
        self.k_core = k_core
        return dataset

    def create_stratified_split(self, ratio: list, k_core: int):
        """Create stratified split for the dataset read file split_data method split_stratified()"""
        if self.k_core:
            assert self.k_core == k_core, (
                f"dataset loaded has k_core: {self.k_core} \n"
                f"requested split with k_core: {k_core}"
            )
        if self.dataset is None:
            self.load(k_core=k_core)

        split_percentage_str = "_".join(list(map(str, ratio)))
        split_name = f"stratified_{split_percentage_str}"

        splits_list = split_stratified(self.dataset, ratio=ratio)
        split_names = ["train", "val", "test"]
        split_dict = dict(zip(split_names, splits_list))

        # create the splits folder
        preprocessed_dir = os.path.join(self.data_directory, "preprocessed")
        k_core_dir = os.path.join(preprocessed_dir, f"k_core_{k_core}")
        splits_dir = os.path.join(k_core_dir, "splits")
        Path(splits_dir).mkdir(exist_ok=True)

        # create the split specific folder using split_name arg
        split_dir = os.path.join(splits_dir, split_name)
        Path(split_dir).mkdir(exist_ok=True)

        for _split_name, _split in split_dict.items():
            # check each _split is a pandas DataFrame
            if not isinstance(_split, pd.DataFrame):
                raise ValueError(
                    "split dictionary contains a split which is not a pd.DataFrame"
                )
            _split.to_csv(os.path.join(split_dir, _split_name + ".csv"), index=False)
        logger.info(
            set_color(
                f"Split stratified {split_percentage_str} saved in {split_dir}",
                "yellow",
            )
        )

    def create_hit_rate_split(self):
        # todo: implement split for hitrate
        raise NotImplementedError

    def load_split(self, k_core: int, split_name: str):
        """Load dataset split"""
        if self.dataset is None:
            self.load(k_core=k_core)
        preprocessed_dir = os.path.join(self.data_directory, "preprocessed")
        k_core_dir = os.path.join(preprocessed_dir, f"k_core_{k_core}")
        splits_dir = os.path.join(k_core_dir, "splits")

        if split_name not in os.listdir(splits_dir):
            available_splits = os.listdir(splits_dir)
            raise FileNotFoundError(
                f"split name: {split_name} not found\n available:{available_splits}"
            )

        split_dir = os.path.join(splits_dir, split_name)
        split_dict = {}
        for split in os.listdir(split_dir):
            split_dict[split.split(".")[0]] = pd.read_csv(
                os.path.join(split_dir, split)
            )

        logger.info(
            set_color(
                f"Loaded split: {split_name}",
                "yellow",
            )
        )
        return split_dict
