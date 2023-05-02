import torch
from torch.utils.data import Dataset , DataLoader
from torchvision.transforms import Resize
import torchvision.transforms as T
from PIL import Image
import os

def get_dataloader(root_dir, batch_size, is_latent=True, midas=True, shuffle=True):
    """
    Returns a dataloader and the number of input channels for the given root directory.

    Args:
        root_dir (str): path to the root directory of the dataset.
        batch_size (int): batch size for the dataloader.
        is_latent (bool): True if the input data is in latent space, False otherwise. Default is True.
        midas (bool): True if the input data includes depth maps from MiDaS, False otherwise. Default is True.
        shuffle (bool): True to shuffle the dataset, False otherwise. Default is True.

    Returns:
        A tuple containing the dataloader and the number of input channels and the len of the dataset.
    """
    dataset = CustomDataset(root_dir, is_latent=is_latent, midas=midas)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    i, _ = dataset[1]
    ch_in = i.shape[0]
    len_ds = len(dataset)
    return dataloader, ch_in, len_ds



class CustomDataset(Dataset):
    """Custom dataset class for loading image data"""

    def __init__(self, root_dir='roto_latent', is_latent=False, midas=False):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            is_latent (bool): If True, load latent space data. Otherwise, load images.
            midas (bool): If True and is_latent is True, concatenate the latent space data
                          with the corresponding Midas data.
        """
        self.root_dir = root_dir
        self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "train_A")))
        self.is_latent = is_latent
        self.midas = midas

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return the input and target data for the given index"""

        image_path = self.image_paths[idx]
        
        if self.is_latent:
            # Load the latent space data
            input = torch.load(os.path.join(self.root_dir, "train_A", image_path)).squeeze()
            target = torch.load(os.path.join(self.root_dir, "train_B", image_path)).squeeze()

            if self.midas:
                # Concatenate the latent space data with the corresponding Midas data
                midas = torch.load(os.path.join(self.root_dir, "midas_A", image_path)).squeeze()
                input = torch.cat([input, midas], dim=0)
        else:
            # Load the images and preprocess them
            input = Image.open(os.path.join(self.root_dir, "train_A", image_path))
            target = Image.open(os.path.join(self.root_dir, "train_B", image_path))
            input = Resize((512, 512))(input)
            target = Resize((512, 512))(target)
            input = T.ToTensor()(input)* 2.0 - 1.0
            target = T.ToTensor()(target)* 2.0 - 1.0

        return input, target, image_path


