from utils.dataloader import get_dataloader
from models.unet import UNet
from PIL import Image
from utils.utils import weights_init, print_inference_images
from diffusers import AutoencoderKL
import torch 
import torch.optim as optim
import time
import os
from tqdm.auto  import tqdm
import torchvision.transforms as T

def sample_images(
        args,
        root_dir = 'roto_latent', 
        batch_size = 1,
        remove_deep_conv=True,
        resolution=512,
        is_latent=True,
        midas=True, 
        ):

    # output paths
    args.saved_images_path = os.path.join(args.output_path_dir, args.saved_images_path)
    args.saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path)

    # models
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    unet = UNet(c_in=ch_in, c_out=4,remove_deep_conv=remove_deep_conv).half().apply(weights_init)
    unet = unet.to('cuda')

    # recover from checkpoint
    path_bkp_model = os.path.join(args.saved_model_path, 'bkp_model.pth')
    cp = torch.load(path_bkp_model)
    unet.load_state_dict(cp['unet_state_dict'])

    # Preprocess to get latents
    if is_latent:
        print(f'Generating latent samples...')
        for filename in tqdm(os.listdir(root_dir)):
            # Load the image
            img = Image.open(os.path.join(root_dir, filename))
            img = T.Resize((resolution, resolution))(img)
            img = T.ToTensor()(img) * 2.0 - 1.0
            img = img.unsqueeze(0)
            img = img.to('cuda').half()
            # Pass the image through the encoder
            with torch.no_grad():
                latents = vae.encode(img).latent_dist.sample() * 0.18215
            # Save the encoded image as a PyTorch tensor
            torch.save(latents, os.path.join(output_A_path, filename[:-4] + '.pt'))


    # dataloader
    dataloader, ch_in, len_ds = get_dataloader(root_dir, 
                                               batch_size, 
                                               is_latent=is_latent, 
                                               midas=midas, 
                                               shuffle=False
                                               )
    
    for (input_batch, target_batch, filename) in dataloader:
        
        input_batch = input_batch.to(device="cuda", dtype=torch.float16) 
        target_batch = target_batch.to(device="cuda", dtype=torch.float16) 
        
        # Unet
        output_batch=unet(input_batch)
        
        print_inference_images(output_batch, path, filename)

    return print("done prediction")

