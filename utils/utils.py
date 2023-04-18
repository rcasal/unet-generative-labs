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
        remove_folder_if_exists (bool): Whether to remove the output folder if it already exists.


    Returns:
        None
    """
    # Define input and output paths
    input_path = os.path.join(input_root_path, "train_A","*")
    style_path = os.path.join(input_root_path, "train_B/","*")
    output_input_path = os.path.join(output_path, "train_A")
    output_style_path = os.path.join(output_path, "train_B")

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
