import os
from tqdm import tqdm
from urllib.request import urlretrieve
import tarfile


class DLProgress(tqdm):
    """Show downloading progress"""
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_caltech101(data_path):
    """Download the Caltech101 dataset if it doesn't exist"""
    if not os.path.isdir(data_path):
        # Download data
        data_tarfile = data_path + '.tar.gz'
        print('Downloading Caltech101 data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz',
                data_tarfile,
                pbar.hook)

        print('Extracting Caltech101 data...')
        with tarfile.open(data_tarfile, "r:gz") as tar:
            tar.extractall(data_path)

        os.remove(data_tarfile)


def maybe_download_caltech256(data_path):
    """Download the Caltech256 dataset if it doesn't exist"""
    if not os.path.isdir(data_path):
        # Download data
        data_tarfile = data_path + '.tar'
        print('Downloading train data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar',
                data_tarfile,
                pbar.hook)

        print('Extracting data...')
        with tarfile.open(data_tarfile, 'r') as tar:
            tar.extractall(data_path)

        os.remove(data_tarfile)
