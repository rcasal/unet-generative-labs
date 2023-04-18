import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def get_first_param(model):
    """
    Get the first parameter of a PyTorch model.

    Args:
        model: A PyTorch model.

    Returns:
        The first parameter of the model.

    Raises:
        StopIteration: If there are no parameters in the model.
    """
    # Get an iterator for the parameters of the model
    params = iter(model.parameters())
    try:
        # Get the first parameter from the iterator
        first_param = next(params)
    except StopIteration:
        # Raise an exception if there are no parameters in the model
        raise StopIteration("The model has no parameters.")
    return first_param


class ExponentialMovingAverage:
    """
    Computes the exponential moving average of a PyTorch model's parameters.

    Args:
        beta (float): The exponential decay rate for the moving average.

    Attributes:
        beta (float): The exponential decay rate for the moving average.
        step (int): The current step of the moving average.
    """

    def __init__(self, beta):
        """
        Initializes a new instance of the ExponentialMovingAverage class.

        Args:
            beta (float): The exponential decay rate for the moving average.
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Updates the parameters of a moving average model based on a current model.

        Args:
            ma_model (nn.Module): The moving average model.
            current_model (nn.Module): The current model.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Updates the exponential moving average of a value.

        Args:
            old (float or None): The old value of the moving average.
            new (float): The new value to add to the moving average.

        Returns:
            The updated moving average value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Takes a step in the exponential moving average.

        Args:
            ema_model (nn.Module): The moving average model.
            model (nn.Module): The current model.
            step_start_ema (int, optional): The step at which to start updating the moving average.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the parameters of the moving average model to match those of the current model.

        Args:
            ema_model (nn.Module): The moving average model.
            model (nn.Module): The current model.
        """
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    Implements a self-attention module for a PyTorch model.

    Args:
        channels (int): The number of input channels.

    Attributes:
        channels (int): The number of input channels.
        mha (nn.MultiheadAttention): The multihead attention layer.
        ln (nn.LayerNorm): The layer normalization layer.
        ff_self (nn.Sequential): The feedforward neural network.
    """

    def __init__(self, channels):
        """
        Initializes a new instance of the SelfAttention class.

        Args:
            channels (int): The number of input channels.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Computes the output of the self-attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            The output of the self-attention module.
        """
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    """
    Implements a double convolutional module for a PyTorch model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        mid_channels (int, optional): The number of channels in the middle layer. Defaults to None.
        residual (bool, optional): Whether to use residual connections. Defaults to False.

    Attributes:
        residual (bool): Whether to use residual connections.
        double_conv (nn.Sequential): The double convolutional layer.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        Initializes a new instance of the DoubleConv class.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            mid_channels (int, optional): The number of channels in the middle layer. Defaults to None.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Computes the output of the double convolutional module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            The output of the double convolutional module.
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        Down module for U-Net architecture. Initialize a downsampling block with a maxpool operation followed by two DoubleConv blocks and an embedding
        layer that projects a one-dimensional tensor to a two-dimensional tensor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            emb_dim (int, optional): Dimensionality of the embedding tensor. Defaults to 256.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        Forward pass of the downsampling block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).
            t (torch.Tensor): Input tensor with shape (batch_size, emb_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height/2, width/2).
        """
        x = self.maxpool_conv(x)
        # Uncomment the following lines if `emb` is a tensor with the same spatial dimensions as `x`
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # return x + emb
        return x


class Up(nn.Module):
    """
    Up module for U-Net architecture.

    Args:
    - in_channels (int): number of input channels.
    - out_channels (int): number of output channels.
    - emb_dim (int, optional): dimensionality of embedding. Defaults to 256.
    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass of Up module.

        Args:
        - x (torch.Tensor): input tensor of shape (batch_size, in_channels, H, W).
        - skip_x (torch.Tensor): skip connection tensor of shape (batch_size, out_channels, H/2, W/2).
        - t (torch.Tensor): embedding tensor of shape (batch_size, emb_dim).

        Returns:
        - torch.Tensor: output tensor of shape (batch_size, out_channels, H, W).
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        #emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        #return x + emb
        return x