import torch

from enum import Enum
from tifffile import imread, imwrite, TiffFile
from torch import Tensor
import json
from src.GaborSystem import GaborSystem
        
C, F, T = 0, 1, 2
        

def c_to_tiff(c: Tensor, filename: str, gs: GaborSystem, normalize: bool = True) -> None:
    """
    Encode a complex spectrogram tensor as a multi-page TIFF file.
    
    Args:
        c (Tensor): Complex spectrogram tensor [channels, bins, frames]
        filename (str): Output filename for the TIFF file
        normalize (bool, optional): Whether to normalize magnitude values to [0,1]. Defaults to True.
    
    The TIFF file will contain:
        - Page 0: Magnitude spectrogram
        - Page 1: Phase spectrogram (in radians, scaled to [0,1])
    """
    # Ensure the input is on CPU and convert to numpy
    c = c.squeeze()
    # c = torch.log1p(c)
    c = c.cpu()
    
    # Extract magnitude and phase
    magnitude = torch.abs(c)
    phase = torch.angle(c)  # -> [-π, π]
    
    # Normalize if requested
    if normalize:
        # Normalize magnitude to [0,1]
        max_mag = magnitude.max()
        magnitude = magnitude / max_mag
        
        # [-π, π] -> [0,1]
        phase = (phase + torch.pi) / (2 * torch.pi)
    
    # Stack the magnitude and phase as separate pages
    # Convert to float32 for better compatibility
    tiff_data = torch.stack([phase, magnitude], dim=C).float().flip(F)

    
    print(tiff_data.shape)
    
    # Convert to numpy for tifffile
    tiff_data_np = tiff_data.numpy()

    metadata = {
        "GaborSystem": gs.params.__dict__,
        "normalcoef": max_mag.item() if normalize else 1.0,
        "description": "Greyscale complex NSGT spectrogram encoded as magnitude and phase pages.",
        # add additional GaborSystem object fields if desired
    }
    metadata_json = json.dumps(metadata)
    
    # Write to TIFF file with metadata in the ImageDescription tag
    imwrite(filename, tiff_data_np, photometric='minisblack', description=metadata_json)
    print(f"Saved complex spectrogram to {filename} with {tiff_data_np.shape} shape")


def c_from_tiff(filename: str, denormalization_constant: float = None) -> Tensor:
    """
    Decode a complex spectrogram from a multi-page TIFF file.
    
    Args:
        filename (str): Input TIFF filename
        denormalize (bool, optional): Whether to denormalize values. Defaults to True.
    
    Returns:
        Tensor: Complex spectrogram tensor [channels, bins, frames]
    """

    P, F, T = 0, 1, 2
    # Read the TIFF file
    tiff_data = torch.tensor(imread(filename))
    # Use a fallback if the pages aren't recognised
    if tiff_data.shape[P] != 2:
        with TiffFile(filename) as tif:
            pages = [page.asarray() for page in tif.pages]

        if len(pages) != 2:
            raise ValueError("TIFF file contains more than a magnitude and phase page")

        phase = torch.tensor(pages[0])
        magnitude = torch.tensor(pages[1])

    else:
        print("Importing tiff c with shape: ", tiff_data.shape)
        phase, magnitude = tiff_data.flip(F)
        print(magnitude.shape)
        
        # Denormalize if requested
    if denormalization_constant is not None:
        magnitude = magnitude * denormalization_constant
        # Denormalize phase from [0,1] to [-π, π]
        phase = phase * (2 * torch.pi) - torch.pi
        # Magnitude is handled by from_log
    
    # Reconstruct complex spectrogram
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    
    # Create complex tensor
    c = torch.complex(real, imag)

    # add channel dim
    # c = c.unsqueeze(0)
    return c


