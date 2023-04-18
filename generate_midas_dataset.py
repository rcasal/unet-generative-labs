import argparse
from utils.utils import generate_midas, str2bool

def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--output_input_path', type=str, default="roto/train_A/", help='The path to the input folder containing train_A files.')
    parser.add_argument('--output_path', type=str, default="roto/midas_A/", help='The path to the output folder to save the generated samples to')

    return parser.parse_args()

def main():
    args = parse_args()
        
    generate_midas(
        input_path=args.output_input_path, 
        output_path=args.output_midas_path)

if __name__ == '__main__':
    main()

