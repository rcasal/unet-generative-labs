import argparse
import os
import warnings
from utils.utils import generate_dataset, str2bool

#generate_dataset(num_samples, input_root_path='roto_face_large', output_path='roto', num_workers=None)
def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--num_samples', type=int, default=10000, help='The number of samples to generate.')
    parser.add_argument('--input_root_path', type=str, default="roto_face_large", help='The path to the input folder containing train_A and train_B subfolders.')
    parser.add_argument('--output_path', type=str, default="roto", help='The path to the output folder to save the generated samples to')
    parser.add_argument('--num_workers', type=int, default=-10, help='The number of cpu to use to parallelize the processing')
    parser.add_argument('--rotate', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")
    parser.add_argument('--translate', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")
    parser.add_argument('--resolution', type=int, default=512, help='The number of cpu to use to parallelize the processing')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.num_workers == -10:
        args.num_workers = None
        
    generate_dataset(args.num_samples, args.input_root_path, args.output_path, args.num_workers)


if __name__ == '__main__':
    main()

