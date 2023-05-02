import argparse
from utils.utils import generate_dataset, str2bool

def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--num_samples', type=int, default=10000, help='The number of samples to generate.')
    parser.add_argument('--input_root_path', type=str, default="roto_face_large", help='The path to the input folder containing train_A and train_B subfolders.')
    parser.add_argument('--output_path', type=str, default="roto", help='The path to the output folder to save the generated samples to')
    parser.add_argument('--rotate', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")
    parser.add_argument('--translate', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")
    parser.add_argument('--resolution', type=int, default=512, help='The number of cpu to use to parallelize the processing')
    parser.add_argument('--parallelize', type=str2bool, nargs='?', const=True, default=False, help="Parallelize the job across all the available cpus.")
    parser.add_argument('--midas', type=str2bool, nargs='?', const=True, default=False, help="process the train_A images using MiDaS in order to generate a side inputs.")
    parser.add_argument('--canny_edges', type=str2bool, nargs='?', const=True, default=False, help="process the train_A images using canny_edges in order to generate a side inputs.")
    parser.add_argument('--remove_if_exist', type=str2bool, nargs='?', const=True, default=False, help="Remove output folder if exists.")

    return parser.parse_args()

def main():
    args = parse_args()
        
    generate_dataset(
        num_samples=args.num_samples, 
        input_root_path=args.input_root_path, 
        output_path=args.output_path, 
        parallelize=args.parallelize,
        rotate=args.rotate,
        translate=args.translate,
        resolution=args.resolution,
        midas=args.midas,
        canny_edges=args.canny_edges,
        remove_if_exist=args.remove_if_exist
        )

if __name__ == '__main__':
    main()

