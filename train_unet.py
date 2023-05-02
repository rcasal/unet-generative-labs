import argparse
from training.training import train_unet
from utils.utils import str2bool
import warnings
import os
import datetime

def parse_args():
    desc = "Script for training UNet"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs to train.')
    parser.add_argument('--root_dir', type=str, default='roto_latent', help='The path to the input folder containing train_A, midas_A and train_B subfolders.')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size to train')
    parser.add_argument('--debug_verbose', type=str2bool, nargs='?', const=True, default=False, help="Flag to show intermediate output. Use for debugging reasons.")
    parser.add_argument('--remove_deep_conv', type=str2bool, nargs='?', const=True, default=False, help="Add a deep convolutional layer in the UNet's bottleneck.")
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='Coefficient for MSE loss')
    parser.add_argument('--lambda_perceptual', type=float, default=0.5, help='Coefficient for MSE loss')
    parser.add_argument('--lambda_l1', type=float, default=0.01, help='Coefficient for MSE loss')
    parser.add_argument('--is_latent', type=str2bool, nargs='?', const=True, default=False, help="Flag to indicate latents as input of the UNet.")
    parser.add_argument('--midas', type=str2bool, nargs='?', const=True, default=False, help="Flag to indicate the use of midas as input of the UNet.")
    parser.add_argument('--canny_edges', type=str2bool, nargs='?', const=True, default=False, help="Flag to indicate the use of canny edges as input of the UNet.")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')
    parser.add_argument('--resume_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--saved_images_path', type=str, default="Images", help='Folder name for save images during training')
    parser.add_argument('--saved_model_path', type=str, default="Saved_Models", help='Folder name for save model')
    parser.add_argument('--saved_history_path', type=str, default="History/", help='The directory for history experiments. Compatible with TensorBoard.')

    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Output path dir
    args.output_path_dir = os.path.join(args.output_path_dir,args.experiment_name) 
    if(not os.path.exists(args.output_path_dir)):
        print('creating directories in ' + args.output_path_dir)
        os.makedirs(args.output_path_dir)
        os.makedirs(os.path.join(args.output_path_dir, args.saved_images_path))
        os.makedirs(os.path.join(args.output_path_dir, args.saved_history_path))
        os.makedirs(os.path.join(args.output_path_dir, args.saved_model_path))

    # train the model
    train_unet(
        args = args,
        root_dir = args.root_dir, 
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        debug_verbose = args.debug_verbose,
        remove_deep_conv=args.remove_deep_conv,
        lambda_mse=args.lambda_mse,
        lambda_perceptual=args.lambda_perceptual, 
        lambda_l1=args.lambda_l1,
        is_latent=args.is_latent,
        midas=args.midas, 
        canny_edges=args.canny_edges, 
        lr = args.lr
        )

if __name__ == '__main__':
    main()

