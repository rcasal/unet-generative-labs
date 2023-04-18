
from utils.dataloader import get_dataloader
from models.unet import UNet
from loss.loss import UNetLoss
from utils.utils import weights_init, print_image
from diffusers import AutoencoderKL
import torch 
import torch.optim as optim
import time
#from torch.utils.tensorboard import SummaryWriter


def train_unet(
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
        lr=1e-3
        ):


    # dataloader
    dataloader, ch_in, len_ds = get_dataloader(root_dir, batch_size, is_latent=is_latent, midas=midas, suffle=True)

    # models
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    unet = UNet(c_in=ch_in, c_out=4,remove_deep_conv=remove_deep_conv).half().apply(weights_init)
    unet = unet.to('cuda')

    #optimizer and loss
    #optimizer = optim.AdamW(unet.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optimizer = optim.Adadelta(unet.parameters(), lr=lr)
    loss_fn = UNetLoss(lambda_mse=lambda_mse, lambda_perceptual=lambda_perceptual, lambda_l1=lambda_l1).half()

    # Training loop
    for epoch in range(num_epochs):
        # time
        since = time.time()
        unet.train()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_f1_loss = 0.0
        running_perceptual_loss = 0.0    
        cur_step = 0
        for input_batch, target_batch in dataloader:
            
            input_batch = input_batch.to(device="cuda", dtype=torch.float16) 
            target_batch = target_batch.to(device="cuda", dtype=torch.float16) 
            
            # Unet
            output_batch=unet(input_batch)
            
            # Compute the loss
            loss, mse_loss, perceptual_loss, f1_loss = loss_fn(output_batch, target_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the running loss
            if perceptual_loss.item() != float('inf') and loss.item() != float('inf') and mse_loss.item() != float('inf') and f1_loss.item() != float('inf'):
                running_loss += loss.item() * output_batch.size(0)
                running_mse_loss += mse_loss.item() * output_batch.size(0)
                running_f1_loss += f1_loss.item() * output_batch.size(0)
                running_perceptual_loss += perceptual_loss.item() * output_batch.size(0)

            cur_step+=1

            # Print after debug steps
            if debug_verbose:
                if cur_step % 20 == 0 and cur_step > 0:
                    print_image(output_batch, target_batch, input_batch[:, 0:4, :, :], vae, loss)
                    print('Loss: {:.4f}, MSE Loss: {:.4f}, F1 Loss: {:.4f}, Perc Loss: {:.4f}'.format(loss.item(), mse_loss.item(), perceptual_loss.item(), f1_loss.item()))


        time_elapsed = time.time() - since
        # Compute the epoch loss and print it
        epoch_loss = running_loss / len_ds
        epoch_mse_loss = running_mse_loss / len_ds
        epoch_f1_loss = running_f1_loss / len_ds
        epoch_perceptual_loss = running_perceptual_loss / len_ds
        print('Epoch [{}/{}], Loss: {:.4f}, MSE Loss: {:.4f}, F1 Loss: {:.4f}, Perc Loss: {:.4f}, Time Elapsed: {:.1f} s'.format(epoch+1, num_epochs, epoch_loss, epoch_mse_loss, epoch_f1_loss, epoch_perceptual_loss, time_elapsed))
        print_image(output_batch,target_batch, input_batch[:, 0:4, :, :], vae, epoch_loss)

    return unet