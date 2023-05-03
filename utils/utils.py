import numpy as np
import cv2
import re
import torchvision.transforms.functional as F
from PIL import Image
import glob
from tqdm import tqdm
import os 
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import argparse
import shutil
import torch 
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def atoi(text):
    """Convert a string to integer if possible."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Sort a list of strings in natural order, as opposed to lexicographic order.
    This is useful when sorting filenames that include numbers.
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def generate_dataset(num_samples, 
                     input_root_path='roto_face_large', 
                     output_path='roto', 
                     parallelize=False, 
                     rotate=False, 
                     translate=False,
                     resolution=512,
                     midas=False,
                     canny_edges=False,
                     remove_if_exist=False):
    """
    Generates a dataset of num_samples samples by selecting images from input_path/train_A and input_path/train_B, and
    saving them to output_path/train_A and output_path/train_B respectively.

    Args:
        num_samples (int): The number of samples to generate.
        input_root_path (str): The path to the input folder containing train_A and train_B subfolders.
        output_path (str): The path to the output folder to save the generated samples to.
        parallelize (bool): Whether to parallelize the processing.
        rotate (bool): Whether to rotate images.
        translate (bool): Whether to translate images.
        resolution (int): The resolution of the output images.
        midas (bool): if True, process the train_A images using MiDaS in order to generate a side input.
        canny_edges (bool): if True, process the train_A images using canny_edges in order to generate a side input.
        remove_folder_if_exists (bool): Whether to remove the output folder if it already exists.


    Returns:
        None
    """
    # Define input and output paths
    input_path = os.path.join(input_root_path, "train_A","*")
    style_path = os.path.join(input_root_path, "train_B/","*")
    output_input_path = os.path.join(output_path, "train_A")
    output_style_path = os.path.join(output_path, "train_B")
    output_midas_path = os.path.join(output_path, "midas_A")
    output_canny_edges_path = os.path.join(output_path, "canny_edges_A")

    # Check if output folder already exists
    if os.path.exists(output_path):
        if remove_if_exist:
            print(f'Removing existing output folder {output_path}')
            shutil.rmtree(output_path)
        else:
            raise ValueError(f'Output folder {output_path} already exists')

    # Create output directories
    print(f"Creating directories in {output_path}")
    os.makedirs(output_input_path)
    os.makedirs(output_style_path)
    os.makedirs(output_midas_path) if midas else None
    os.makedirs(output_canny_edges_path) if canny_edges else None

    # Get a list of all input files
    files_A = sorted(glob.glob(input_path), key=natural_keys)
    files_B = sorted(glob.glob(style_path), key=natural_keys)

    # Limit the number of files to match the number of style files
    b_cap = len(files_B)
    files_A = files_A[:b_cap]

    # Zip the input and style files together
    files = list(zip(files_A, files_B))

    if parallelize:
        # Parallel processing
        print(f'Parallezing the process across {mp.cpu_count()} cpus')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            tasks = [(index, files, rotate, translate, resolution, output_input_path, output_style_path) for index in range(num_samples)]
            results = list(tqdm(pool.imap(generate_sample, tasks), total=num_samples))
    else:
        # Serial processing
        for index in tqdm(range(num_samples)):
            generate_sample((index, files, rotate, translate, resolution, output_input_path, output_style_path))

    # Print the number of samples generated
    print(f'{num_samples} samples generated')
    
    # midas code
    if midas:
        print(f'Generating midas samples')
        generate_midas(input_path=output_input_path, output_path=output_midas_path)

    # canny edges code
    if canny_edges:
        print(f'Generating canny edges samples')
        generate_canny_edges(input_path=output_input_path, output_path=output_canny_edges_path)


def generate_sample(args):
    index, files, rotate, translate, resolution, output_input_path, output_style_path = args
    img_a,img_b = transform(files, rotate, translate, resolution)

    img_a_name = os.path.join(output_input_path,f'{index:04d}.jpg')
    img_b_name = os.path.join(output_style_path,f'{index:04d}.jpg')

    cv2.imwrite(img_a_name,img_a)
    cv2.imwrite(img_b_name,img_b)


def transform(files, rotate=False, translate=False, resolution=512):
    """
    Transform two images randomly selected from a list of image pairs.
    The images are resized and optionally rotated and/or translated.
    The transformed images are returned as NumPy arrays.

    Args:
        files (list): A list of pairs of image file paths.
        rotate (bool): Whether to randomly rotate the images.
        translate (bool): Whether to randomly translate the images.
        resolution (int): The resolution to which the images should be resized.

    Returns:
        tuple: Two transformed images as NumPy arrays.
    """
    # Select a random pair of image files
    print(f'len {len(files)}')
    idx = np.random.randint(len(files))

    # Load the images using PIL and convert to RGB
    img_raw_a = Image.open(files[idx][0]).convert(mode='RGB')
    img_raw_b = Image.open(files[idx][1]).convert(mode='RGB')

    # Resize the images to the specified resolution
    img_a = img_raw_a.resize((resolution, resolution))
    img_b = img_raw_b.resize((resolution, resolution))

    # Randomly rotate the images
    if rotate:
        # Generate a random rotation angle
        rand_rot = np.random.randint(-365, 365)
        # Apply the rotation to both images
        img_a = F.rotate(img_a, rand_rot, fill=(255, 255, 255))
        img_b = F.rotate(img_b, rand_rot, fill=(255, 255, 255))

    # Randomly translate the images
    if translate:
        # Define the fill color for the affine transformation
        replace_val = (255, 255, 255)

        # Generate random parameters for the affine transformation
        rand_x = np.random.random()
        rand_y = np.random.random()
        rand_trans = np.random.random()
        random_x_trans = np.random.randint(-50, 50)
        random_y_trans = np.random.randint(-50, 50)

        # Flip the images horizontally with 50% probability
        if rand_x > 0.5:
            img_a = F.hflip(img_a)
            img_b = F.hflip(img_b)

        # Flip the images vertically with 50% probability
        if rand_y > 0.5:
            img_a = F.vflip(img_a)
            img_b = F.vflip(img_b)

        # Apply the affine transformation with 50% probability
        if rand_trans > 0.5:
            img_a = F.affine(img_a, 0, [random_x_trans, random_y_trans], 1.0, 0, fill=replace_val)
            img_b = F.affine(img_b, 0, [random_x_trans, random_y_trans], 1.0, 0, fill=replace_val)

    # Convert the images to NumPy arrays and convert from BGR to RGB
    img_a = np.array(img_a)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

    img_b = np.array(img_b)
    img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB)

    return img_a, img_b


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def generate_midas(input_path='roto/train_A/', output_path='roto/midas_A/'):
    
    #Load a model (see https://github.com/intel-isl/MiDaS/#Accuracy for an overview)
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    #Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    #Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Loop through the images in train_A
    for filename in tqdm(os.listdir(input_path)):
        # Load the image
        img = cv2.imread(os.path.join(input_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        # Pass the image through the encoder
        #Predict and resize to original resolution
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        # Save the image
        img_name = os.path.join(output_path,filename)
        cv2.imwrite(img_name,output)


def generate_canny_edges(input_path='roto/train_A/', output_path='roto/canny_edges_A/'):
     
    # Loop through the images in train_A
    for filename in tqdm(os.listdir(input_path)):
        # Load the image
        img = cv2.imread(os.path.join(input_path, filename))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth the image and reduce noise
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Apply Canny edge detection algorithm
        edges = cv2.Canny(blur, 40, 80)
        
        # Save the image
        img_name = os.path.join(output_path,filename)
        cv2.imwrite(img_name,edges)


def convert_pytorch_to_pil(image):
    """
    Converts a PyTorch tensor to a PIL Image object.

    Args:
        image (torch.Tensor): A PyTorch tensor representing an image.

    Returns:
        PIL.Image: A PIL Image object representing the image.
    """
    # Add a batch dimension to the tensor
    image = image.unsqueeze(0)
    # Rescale the values to the range [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    # Convert the tensor to a NumPy array and change the dimensions order
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    # Convert the values to uint8 and create a PIL Image object
    image = (image * 255).round().astype("uint8")
    decoded_image = Image.fromarray(image[0])

    return decoded_image


def weights_init(m):
    """Initialize the weights of a PyTorch model.

    Args:
        m (nn.Module): The module to initialize.

    Returns:
        None
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)


def print_image(output_batch, target_batch, input_batch, vae, loss, epoch, path):
    """
    Displays a comparison of input, output, and target images from a VAE, along with their decoded versions.
    
    Args:
    - output_batch (torch.Tensor): a batch of output images generated by the VAE
    - target_batch (torch.Tensor): a batch of target images that the VAE is trying to reconstruct
    - input_batch (torch.Tensor): a batch of input images used as the starting point for VAE's generative process
    - vae (nn.Module): the trained Variational Autoencoder model
    - loss (float): the loss value for the current epoch
    
    Returns:
    - None
    """
    
    # Rescale input, output, and target images
    input_batch = (1 / 0.18215) * input_batch
    output_batch = (1 / 0.18215) * output_batch
    target_batch = (1 / 0.18215) * target_batch
    
    with torch.no_grad():
        # Decode input, output, and target images
        decoded_input_batch = vae.decode(input_batch[0].unsqueeze(0)).sample
        decoded_batch = vae.decode(output_batch[0].unsqueeze(0)).sample
        decoded_target = vae.decode(target_batch[0].unsqueeze(0)).sample

    # Convert output and target images to PIL images
    output_pil = convert_pytorch_to_pil(output_batch[0])
    target_pil = convert_pytorch_to_pil(target_batch[0])

    # Convert decoded input, output, and target images to PIL images
    dinput_pil = convert_pytorch_to_pil(decoded_input_batch[0])
    doutput_pil = convert_pytorch_to_pil(decoded_batch[0])
    dtarget_pil = convert_pytorch_to_pil(decoded_target[0])

    # Plot all images
    fig, axs = plt.subplots(1, 5, figsize=(10, 3))
    axs[0].imshow(dinput_pil)
    axs[1].imshow(output_pil)
    axs[2].imshow(target_pil)
    axs[3].imshow(doutput_pil)
    axs[4].imshow(dtarget_pil)
    
    # Add axis titles and remove ticks
    axs[0].set_title('Decoded input')
    axs[1].set_title('Output')
    axs[2].set_title('Target')
    axs[3].set_title('Decoded output')
    axs[4].set_title('Decoded target')
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Set suptitle and adjust layout
    fig.suptitle(f'Loss: {loss:.2f}', fontsize=12, fontweight='bold', y=0.05, fontdict={'verticalalignment': 'bottom'})
    plt.tight_layout()
    
    # Save fig
    output_path = os.path.join(path,f"epoch_{epoch:04d}.jpg")
    fig.savefig(output_path)

    # Display plot ! No Anda en colab
    # plt.show()
    # plt.pause(0.001)

def print_inference_images(output, vae, path):
    """
    Displays a comparison of input, output, and target images from a VAE, along with their decoded versions.
    
    Args:
    - output (torch.Tensor): a batch of output images generated by the VAE
    - vae (nn.Module): the trained Variational Autoencoder model
    
    Returns:
    - None
    """
    
    # Rescale input, output, and target images
    output = (1 / 0.18215) * output
    
    with torch.no_grad():
        # Decode input, output, and target images
        decoded = vae.decode(output[0].unsqueeze(0)).sample

    # Convert output and target images to PIL images
    doutput_pil = convert_pytorch_to_pil(decoded)
    
    # Save fig
    save_sampled_images(doutput_pil, path, file_name)



def save_sampled_images(image_tensor_fake, path, file_name):
    '''
    Function for visualizing images: Given a tensor of imagess, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # fake
    image_tensor_fake = (image_tensor_fake + 1) / 2
    image_fake_unflat = image_tensor_fake.detach().cpu()
    image_fake_grid = make_grid(image_fake_unflat[:1], nrow=1)
    
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.imshow(image_fake_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')

    output_path = os.path.join(path,f"{file_name}.jpg") # need to change name
    fig.savefig(output_path, bbox_inches='tight')