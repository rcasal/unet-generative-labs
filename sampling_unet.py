import argparse
from utils.preprocessing import generate_latents
from utils.utils import str2bool
import os

def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--input_root_path', type=str, default="roto", help='The path to the input folder containing train_A and train_B subfolders.')
    parser.add_argument('--output_path', type=str, default="roto_latent", help='The path to the output folder to save the generated samples to')
    parser.add_argument('--resolution', type=int, default=512, help='The number of cpu to use to parallelize the processing')
    parser.add_argument('--midas', type=str2bool, nargs='?', const=True, default=False, help="process the midas_A images from input_root_path to get the embeddings.")
    parser.add_argument('--remove_if_exist', type=str2bool, nargs='?', const=True, default=False, help="Parallelize the job across all the available cpus.")

    # Dataset parameters and input paths
    parser.add_argument('--input_path_dir', type=str, default="/Users/ramirocasal/Documents/Datasets/sword_sorcery_data_for_ramiro/test_dataset", help='Path root where inputs are located. By default it will contain 3 subfolders: img, inst, label')
    parser.add_argument('--input_img_dir', type=str, default="02_output", help='Folder name for input images located in input_path_dir')
    parser.add_argument('--saved_model_path', type=str, default="Saved_Models", help='Path to the saved model')
    parser.add_argument('--model_name', type=str, default="pix2pixHD_model.pth", help='Name of the saved model')

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--output_images_path', type=str, default="Sampled_images", help='Folder name for save images during training')
    parser.add_argument('--output_images_subfolder_path', type=str, default="", help='Subfolder name for save images during training')

    parser.add_argument('--experiment_name', type=str, default="", help='A name of the training experiment')

    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")
    return parser.parse_args()

def main():
    args = parse_args()
        
    # Resume training and experiment name
    args.experiment_name = args.experiment_name
    
    # Output path dir
    args.output_path_dir = os.path.join(args.output_path_dir,args.experiment_name) 
    print('creating directories in ' + args.output_path_dir)
    os.makedirs(args.output_path_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_path_dir, args.output_images_path, args.output_images_subfolder_path), exist_ok=True)

    args.saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path, args.model_name)
    
    print('Recovering model from ' + args.saved_model_path)
    
    generate_latents(
        input_root_path=args.input_path_dir, 
        output_path=args.output_path, 
        resolution=args.resolution, 
        midas=args.midas,
        remove_if_exist=args.remove_if_exist
        )
    

if __name__ == '__main__':
    main()



"""
!python3 pix2pixhd/pix2pixhd_sample.py \
    --input_path_dir "/home/jupyter/Ramiro/datasets/character_3" \
    --input_img_dir "" \
    --output_path_dir "/home/jupyter/Ramiro/pix2pixHD_results"  \
    --experiment_name "2022_09_05_19_48_All_data_512_5000samples"  \
    --output_images_subfolder_path "character_3" \
    --model_name "bkp_model_stage2.pth" \
    --target_width 512
"""

"""
#CONVERT VIDEO
import os
import shutil
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import glob
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


path = "/home/jupyter/Ramiro/pix2pixHD_results/2022_09_05_19_48_All_data_512_5000samples/Sampled_images/character_3"
path = "/home/jupyter/DualStyleGAN/data/animation_test_data_1024/train_B_01t"

input_dir = os.path.join(path,"*.jpg")

target_imgs = sorted(glob.glob(input_dir),key=natural_keys)

output_frames = []

for i,file_name in tqdm(enumerate(target_imgs)):
  img = cv2.imread(file_name)
  output_resized = cv2.resize(img,(1024,1024),interpolation = cv2.INTER_NEAREST)
  output_frames.append(output_resized.astype(np.uint8))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out_path = os.path.join(path, "video_01trunc4.mp4")
writer = cv2.VideoWriter(out_path, fourcc, 24, (1024,1024))

for frame in output_frames:
    writer.write(frame)

writer.release() 


print("video generated")
"""