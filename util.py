import numpy as np
import torch
from torch._C import dtype
from typing import Dict


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def to_coordinates_and_features_2D(img):
    """Converts an image to a set of coordinates and features.

        Args:
        img (torch.Tensor): Shape (channels, X, Y).
        """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    # Get coordinates
    d1, d2, d3 = np.mgrid[0:img.shape[1], 0:img.shape[2], 0:img.shape[3]]

    d1 = np.reshape(d1, (img.shape[1] * img.shape[2], 1))
    d2 = np.reshape(d2, (img.shape[1] * img.shape[2], 1))

    d1 = 2 * (torch.from_numpy(d1.astype(np.float32)) / (img.shape[1] - 1) - 0.5)
    d2 = 2 * (torch.from_numpy(d2.astype(np.float32)) / (img.shape[2] - 1) - 0.5)

    coordinates = torch.ones(img.shape[1] * img.shape[2], 2)
    coordinates[:, 0] = d1[:, 0]
    coordinates[:, 1] = d2[:, 0]

    # Get features
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def to_coordinates_latent_and_features(img):
    """Converts an image to a set of coordinates and features.

        Args:
        img (torch.Tensor): Shape (channels, X, Y).
        """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    # Get coordinates
    d1, d2, d3 = np.mgrid[0:img.shape[1], 0:img.shape[2], 0:img.shape[3]]

    d1 = np.reshape(d1, (img.shape[1] * img.shape[2] * img.shape[3], 1))
    d2 = np.reshape(d2, (img.shape[1] * img.shape[2] * img.shape[3], 1))
    d3 = np.reshape(d3, (img.shape[1] * img.shape[2] * img.shape[3], 1))

    d1 = 2 * (torch.from_numpy(d1.astype(np.float32)) / (img.shape[1] - 1) - 0.5)
    d2 = 2 * (torch.from_numpy(d2.astype(np.float32)) / (img.shape[2] - 1) - 0.5)
    d3 = 2 * (torch.from_numpy(d3.astype(np.float32)) / (img.shape[3] - 1) - 0.5)

    coordinates = torch.ones(img.shape[1] * img.shape[2] * img.shape[3], 2)
    coordinates[:, 0] = d1[:, 0]
    coordinates[:, 1] = d2[:, 0]
    latent = torch.ones(img.shape[1] * img.shape[2] * img.shape[3], 1)
    latent[:, 0] = d3[:, 0]

    # Get features
    features = img.reshape(img.shape[0], -1).T
    return coordinates, latent, features


def to_coordinates_and_features_3D(img):
    """Converts an image to a set of coordinates and features.

        Args:
        img (torch.Tensor): Shape (channels, X, Y, Z).
        """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    print("Get coordinates")
    d1, d2, d3 = np.mgrid[0:img.shape[1], 0:img.shape[2], 0:img.shape[3]]
    d1 = np.reshape(d1, (img.shape[1] * img.shape[2] * img.shape[3], 1))
    d2 = np.reshape(d2, (img.shape[1] * img.shape[2] * img.shape[3], 1))
    d3 = np.reshape(d3, (img.shape[1] * img.shape[2] * img.shape[3], 1))
    d1 = 2 * (torch.from_numpy(d1.astype(np.float32)) / (img.shape[1] - 1) - 0.5)
    d2 = 2 * (torch.from_numpy(d2.astype(np.float32)) / (img.shape[2] - 1) - 0.5)
    d3 = 2 * (torch.from_numpy(d3.astype(np.float32)) / (img.shape[3] - 1) - 0.5)

    # coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    coordinates = torch.ones(img.shape[1] * img.shape[2] * img.shape[3], 3)
    coordinates[:, 0] = d1[:, 0]
    coordinates[:, 1] = d2[:, 0]
    coordinates[:, 2] = d3[:, 0]
    # Normalize coordinates to lie in [-.5, .5]
    # print("Normalize coordinates")
    # coordinates[:,0] = coordinates[:,0] / (img.shape[1] - 1) - 0.5
    # coordinates[:,1] = coordinates[:,1] / (img.shape[2] - 1) - 0.5
    # coordinates[:,2] = coordinates[:,2] / (img.shape[3] - 1) - 0.5
    # Convert to range [-1, 1]
    # coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    print("Get features")
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)
