import torch
from torch import nn
from compressai.layers import GDN1
from compressai.models import CompressionModel, MeanScaleHyperprior
from collections import namedtuple

QuantizedTensor = namedtuple('QuantizedTensor', ['tensor', 'scale', 'zero_point'])

# Referred to https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py
#  and http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
def quantize_tensor(x, num_bits=8):
    # c = torch.quantize_per_tensor(x, p[0], p[1], dtype=torch.quint8)
    # print(c)

    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    # min_val, max_val = x.min(), x.max()
    min_val, max_val = -torch.std(x.flatten())*num_bits/2, torch.std(x.flatten())*num_bits/2
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = min_val
    qx = (x - zero_point)/ scale
    qx = qx.clamp(qmin, qmax).round().byte()
    return QuantizedTensor(tensor=qx, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor) + q_x.zero_point


class FactorizedPriorAE(CompressionModel):
    """Simple autoencoder with a factorized prior """
    def __init__(self, entropy_bottleneck_channels=128):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        N = entropy_bottleneck_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, 2, 2, 1)
        )

    def compress(self, x, **kwargs):
        latent = self.encoder(x)
        compressed_latent = self.entropy_bottleneck.compress(latent)
        return compressed_latent, latent.size()[2:]

    def decompress(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods