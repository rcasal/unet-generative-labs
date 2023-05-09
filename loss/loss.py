import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNetLoss(nn.Module):
    """Calculate the loss for the U-Net model.

    Args:
        lambda_mse (float, optional): Coefficient for mean squared error loss. Defaults to 1.0.
        lambda_perceptual (float, optional): Coefficient for perceptual loss. Defaults to 0.1.
        lambda_l1 (float, optional): Coefficient for l1 loss. Defaults to 0.01.
        device (str, optional): Device to use for computations. Defaults to 'cuda'.

    Attributes:
        lambda_mse (float): Coefficient for mean squared error loss.
        lambda_perceptual (float): Coefficient for perceptual loss.
        lambda_l1 (float): Coefficient for l1 loss.
        device (str): Device to use for computations.
        vgg (nn.Module): Pretrained VGG19 model.
        layers (dict): Dictionary of layers to use for perceptual loss.

    """
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.1, lambda_l1=0.01, device='cuda'):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_l1 = lambda_l1
        self.device = device
        self.vgg = models.vgg19(pretrained=True).features.to(device).half()
        self.layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_3',
            '26': 'relu4_3',
            '35': 'relu5_3'
        }
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, y_pred, y_true):
        """Forward pass to calculate the loss.

        Args:
            y_pred (torch.Tensor): Predicted output tensor.
            y_true (torch.Tensor): Ground truth tensor.

        Returns:
            tuple: A tuple containing the total loss, mean squared error loss, perceptual loss, and l1 loss.

        """
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        # Compute mean squared error loss
        mse_loss = F.mse_loss(y_pred, y_true)

        # Compute perceptual loss
        x_vgg, y_vgg = self.get_vgg_features(y_pred), self.get_vgg_features(y_true)
        perceptual_loss = 0
        for layer in self.layers:
            perceptual_loss += F.mse_loss(x_vgg[self.layers[layer]], y_vgg[self.layers[layer]])

        # Compute l1 loss
        l1_loss = F.l1_loss(y_pred, y_true)

        # Compute weighted sum of all losses
        total_loss = self.lambda_mse * mse_loss + self.lambda_perceptual * perceptual_loss + self.lambda_l1 * l1_loss

        return total_loss, mse_loss, perceptual_loss, l1_loss

    def get_vgg_features(self, x):
        """Get the VGG features for the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary of VGG features.

        """
        vgg_features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layers:
                vgg_features[self.layers[name]] = x
        return vgg_features