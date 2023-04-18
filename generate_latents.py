import argparse
from utils.preprocessing import generate_latents
from utils.utils import str2bool

def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--input_root_path', type=str, default="roto", help='The path to the input folder containing train_A and train_B subfolders.')
    parser.add_argument('--output_path', type=str, default="roto_latent", help='The path to the output folder to save the generated samples to')
    parser.add_argument('--resolution', type=int, default=512, help='The number of cpu to use to parallelize the processing')
    parser.add_argument('--remove_if_exist', type=str2bool, nargs='?', const=True, default=False, help="Parallelize the job across all the available cpus.")

    return parser.parse_args()

def main():
    args = parse_args()
        
    generate_latents(
        input_root_path='roto', 
        output_path='roto_latent', 
        resolution=512, 
        remove_if_exist=False
        )

if __name__ == '__main__':
    main()

