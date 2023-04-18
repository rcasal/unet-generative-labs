import argparse
from training.training import train_unet
from utils.utils import str2bool


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
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    return parser.parse_args()

def main():
    args = parse_args()
        
    train_unet(
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
        lr = args.lr
        )

if __name__ == '__main__':
    main()

