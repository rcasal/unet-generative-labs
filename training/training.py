
from utils.dataloader import get_dataloader
from models.unet import UNet
from loss.loss import UNetLoss
from utils.utils import weights_init, print_image
from diffusers import AutoencoderKL
import torch 
import torch.optim as optim
import time
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto  import tqdm

def train_unet(
        args,
        root_dir = 'roto_latent', 
        num_epochs = 1000,
        batch_size = 4,
        debug_verbose = False,
        remove_deep_conv=True,
        lambda_mse=1.0,
        lambda_perceptual=0.5, 
        lambda_l1=0.01,
        is_latent=True,
        midas=True, 
        canny_edges=True, 
        lr=1e-3,
        ):

    # output paths
    args.saved_images_path = os.path.join(args.output_path_dir, args.saved_images_path)
    args.saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path)

    # dataloader
    dataloader, ch_in, len_ds = get_dataloader(root_dir, batch_size, is_latent=is_latent, midas=midas, canny_edges=canny_edges, shuffle=True)

    # models
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    # Freeze decoder weights
    vae = vae.eval()

    unet = UNet(c_in=ch_in, c_out=4,remove_deep_conv=remove_deep_conv).half().apply(weights_init)
    unet = unet.to('cuda')

    #optimizer and loss
    #optimizer = optim.AdamW(unet.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optimizer = optim.Adadelta(unet.parameters(), lr=lr)
    loss_fn = UNetLoss(lambda_mse=lambda_mse, lambda_perceptual=lambda_perceptual, lambda_l1=lambda_l1).half()

    # Tensorboard
    layout = {
        "Loss function": {
            "loss": ["Multiline", ["loss/loss", "loss/mse", "loss/f1", "loss/perceptual"]],
        },
    }
    args.writer = SummaryWriter(
        log_dir=os.path.join(args.output_path_dir, args.saved_history_path),
        filename_suffix=args.experiment_name
        )
    args.writer.add_custom_scalars(layout)

    # recover from checkpoint
    path_bkp_model = os.path.join(args.saved_model_path, 'bkp_model.pth')
    epoch_run=0
    if(args.resume_training and os.path.exists(path_bkp_model)):
        cp = torch.load(path_bkp_model)
        epoch_run = cp['epoch']
        unet.load_state_dict(cp['unet_state_dict'])
        #best_model_wts = cp['best_model_wts']
        optimizer.load_state_dict(cp['optimizer_state_dict'])        
        print('Resuming script in epoch {}.'.format(epoch_run))   

    # Training loop
    for epoch in tqdm(range(num_epochs-epoch_run)):
        # time
        since = time.time()
        unet.train()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_l1_loss = 0.0
        running_perceptual_loss = 0.0  
        cur_step = 0
  
        for (input_batch, target_batch, _) in dataloader:
            
            input_batch = input_batch.to(device="cuda", dtype=torch.float16) 
            target_batch = target_batch.to(device="cuda", dtype=torch.float16) 
            with torch.no_grad():
                target_batch = vae.decode(target_batch).sample
            
            # Unet
            latent_output_batch=unet(input_batch)
            print('before decoder' + str(latent_output_batch.shape))
            # Decoder
            with torch.no_grad():
                output_batch = vae.decode(latent_output_batch).sample
                print('after decoder' + str(output_batch.shape))
            
            # Compute the loss
            loss, mse_loss, perceptual_loss, l1_loss = loss_fn(output_batch, target_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the running loss
            if perceptual_loss.item() != float('inf') and loss.item() != float('inf') and mse_loss.item() != float('inf') and l1_loss.item() != float('inf'):
                running_loss += loss.item() * output_batch.size(0)
                running_mse_loss += mse_loss.item() * output_batch.size(0)
                running_l1_loss += l1_loss.item() * output_batch.size(0)
                running_perceptual_loss += perceptual_loss.item() * output_batch.size(0)

            cur_step+=1

            # Print after debug steps
            if debug_verbose:
                if cur_step % 20 == 0 and cur_step > 0:
                    print_image(latent_output_batch, target_batch, input_batch[:, 0:4, :, :], vae, loss, epoch+epoch_run, args.saved_images_path)
                    print('Loss: {:.4f}, MSE Loss: {:.4f}, L1 Loss: {:.4f}, Perc Loss: {:.4f}'.format(loss.item(), mse_loss.item(), perceptual_loss.item(), l1_loss.item()))


        # Compute the epoch loss and print it
        epoch_loss = running_loss / len_ds
        epoch_mse_loss = running_mse_loss / len_ds
        epoch_l1_loss = running_l1_loss / len_ds
        epoch_perceptual_loss = running_perceptual_loss / len_ds
        # time_elapsed = time.time() - since
        # print('Epoch [{}/{}], Loss: {:.4f}, MSE Loss: {:.4f}, L1 Loss: {:.4f}, Perc Loss: {:.4f}, Time Elapsed: {:.1f} s'.format(epoch+1, num_epochs, epoch_loss, epoch_mse_loss, epoch_l1_loss, epoch_perceptual_loss, time_elapsed))
        print_image(output_batch, target_batch, input_batch[:, 0:4, :, :], vae, loss, epoch+epoch_run, args.saved_images_path)

        # Loss for TensorBoard 
        args.writer.add_scalar(f'loss/loss', epoch_loss, epoch)
        args.writer.add_scalar(f'loss/mse', lambda_mse*epoch_mse_loss, epoch)
        args.writer.add_scalar(f'loss/l1', lambda_l1*epoch_l1_loss, epoch)
        args.writer.add_scalar(f'loss/perceptual', lambda_perceptual*epoch_perceptual_loss, epoch)

        # Save checkpoint
        if args.saved_model_path is not None:
            torch.save({
                'epoch': epoch + epoch_run + 1,
                'unet_state_dict': unet.state_dict(),
                # Best models states
                'optimizer_state_dict': optimizer.state_dict(),
            }, path_bkp_model)

