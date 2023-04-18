import os
import shutil
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoencoderKL


def generate_latents(input_root_path='roto', output_path='roto_latent', resolution=512, remove_if_exist=False):
    """
    Generate latent codes for images using a VAE model and save them as PyTorch tensors.

    :param input_root_path: str, root directory containing the input images in two subdirectories 'train_A' and 'train_B'
    :param output_path: str, directory to save the generated latent codes in two subdirectories 'train_A' and 'train_B'
    :param resolution: int, size of the images after resizing
    :param remove_if_exist: bool, if True, remove the output directory if it already exists, otherwise raise an error
    :return: None
    """

    # Define input and output paths
    input_path = os.path.join(input_root_path, "train_A", "*")
    style_path = os.path.join(input_root_path, "train_B", "*")
    output_A_path = os.path.join(output_path, "train_A")
    output_B_path = os.path.join(output_path, "train_B")

    # Define the encoder model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae = vae.to('cuda')

    # Check if output folder already exists
    if os.path.exists(output_path):
        if remove_if_exist:
            print(f'Removing existing output folder {output_path}')
            shutil.rmtree(output_path)
        else:
            raise ValueError(f'Output folder {output_path} already exists')

    # Create output directories
    print(f"Creating directories in {output_path}")
    os.makedirs(output_A_path)
    os.makedirs(output_B_path)

    # Loop through the images in train_A
    print(f'Generating A samples...')
    for filename in tqdm(os.listdir(input_path)):
        # Load the image
        img = Image.open(os.path.join(input_path, filename))
        img = T.Resize((resolution, resolution))(img)
        img = T.ToTensor()(img) * 2.0 - 1.0
        img = img.unsqueeze(0)
        img = img.to('cuda').half()
        # Pass the image through the encoder
        with torch.no_grad():
            latents = vae.encode(img).latent_dist.sample() * 0.18215
        # Save the encoded image as a PyTorch tensor
        torch.save(latents, os.path.join(output_A_path, filename[:-4] + '.pt'))

    # Loop through the images in train_B
    print(f'Generating B samples...')
    for filename in tqdm(os.listdir(style_path)):
        # Load the image
        img = Image.open(os.path.join(style_path, filename))
        img = T.Resize((resolution, resolution))(img)
        img = T.ToTensor()(img) * 2.0 - 1.0
        img = img.unsqueeze(0)
        img = img.to('cuda').half()
        # Pass the image through the encoder
        with torch.no_grad():
            latents = vae.encode(img).latent_dist.sample() * 0.18215
        # Save the encoded image as a PyTorch tensor
        torch.save(latents, os.path.join(output_B_path, filename[:-4] + '.pt'))