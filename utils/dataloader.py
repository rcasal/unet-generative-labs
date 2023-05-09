import torch
from torch.utils.data import Dataset , DataLoader
from torchvision.transforms import Resize
import torchvision.transforms as T
from PIL import Image
import os

def get_dataloader(root_dir, target_dir, batch_size, is_latent=True, midas=True, canny_edges=True, shuffle=True):
    """
    Returns a dataloader and the number of input channels for the given root directory.

    Args:
        root_dir (str): path to the root directory of the dataset.
        batch_size (int): batch size for the dataloader.
        is_latent (bool): True if the input data is in latent space, False otherwise. Default is True.
        midas (bool): True if the input data includes depth maps from MiDaS, False otherwise. Default is True.
        canny_edges (bool): True if the input data includes depth maps from canny_edges, False otherwise. Default is True.
        shuffle (bool): True to shuffle the dataset, False otherwise. Default is True.

    Returns:
        A tuple containing the dataloader and the number of input channels and the len of the dataset.
    """
    dataset = CustomDataset(root_dir, target_dir, midas=midas, canny_edges=canny_edges)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    i, _, _ = dataset[1]
    ch_in = i.shape[0]
    len_ds = len(dataset)
    return dataloader, ch_in, len_ds



class CustomDataset(Dataset):
    """Custom dataset class for loading image data"""

    def __init__(self, root_dir='roto_latent', target_dir='roto/train_B', midas=False, canny_edges=False):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            is_latent (bool): If True, load latent space data. Otherwise, load images.
            midas (bool): If True and is_latent is True, concatenate the latent space data
                          with the corresponding Midas data.
        """
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "train_A")))
        self.midas = midas
        self.canny_edges = canny_edges

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return the input and target data for the given index"""

        image_path = self.image_paths[idx]
        
        # Load the latent space data
        input = torch.load(os.path.join(self.root_dir, "train_A", image_path)).squeeze()
        target = Image.open(os.path.join(self.target_dir, "train_B", image_path.replace(".pt", ".jpg")))
        target = Resize((512, 512))(target)
        target = T.ToTensor()(target)* 2.0 - 1.0

        if self.midas:
            # Concatenate the latent space data with the corresponding Midas data
            midas = torch.load(os.path.join(self.root_dir, "canny_edges_A", image_path)).squeeze()
            input = torch.cat([input, midas], dim=0)

        if self.canny_edges:
            # Concatenate the latent space data with the corresponding canny_edges data
            canny_edges = torch.load(os.path.join(self.root_dir, "canny_edges_A", image_path)).squeeze(0)
            input = torch.cat([input, canny_edges], dim=0)

        return input, target, image_path


