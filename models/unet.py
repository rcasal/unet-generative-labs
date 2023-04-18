from models.basic_blocks import DoubleConv, Down, SelfAttention, Up, get_first_param
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Implementation of the U-Net model
    """

    def __init__(self, c_in=3, c_out=3,remove_deep_conv=False):
        """
        Initializes the UNet model.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            remove_deep_conv (bool): Flag indicating whether to remove a convolutional block from the deep path.
        """
        super().__init__()
        #self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Computes positional encoding for the input tensor.

        Args:
            t (torch.Tensor): Input tensor.
            channels (int): Number of channels.

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=get_first_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        """
        Performs the forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.inc(x)        # 3, 512, 512
        x2 = self.down1(x1, t)  # 128, 256, 256
        x2 = self.sa1(x2)       # 
        x3 = self.down2(x2, t)  # 512, 128, 128
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)  # 256, 64, 64
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)      # 512, 64, 64 
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)  # 512, 64, 64
        x4 = self.bot3(x4)      # 256, 64, 64

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t=None):
        #t = t.unsqueeze(-1)
        #t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)
        

class UNetConditional(UNet):
    """Conditional UNet model that incorporates an additional label embedding.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        time_dim (int): The time dimension for the positional encoding.
        num_classes (int, optional): Number of classes for label embedding.
        **kwargs: Additional arguments passed to parent class UNet.
    """

    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        
        # Add an embedding layer for label information if num_classes is not None
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, c_in, height, width).
            t (torch.Tensor): Input tensor with shape (batch_size, time_dim).
            y (torch.Tensor, optional): Label tensor with shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, c_out, height, width).
        """
        t = t.unsqueeze(-1)
        
        # Apply positional encoding
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            # Add label embedding to the positional encoding
            t += self.label_emb(y)

        # Perform forward pass through the UNet
        return self.unet_forward(x, t)
