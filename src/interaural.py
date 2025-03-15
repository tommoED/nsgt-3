import torch
from torch import Tensor, tensor

def interaural(c: Tensor, significance_threshold = 4.0):
    """
    Computes an interaural ratio spectrogram with the centre signal cancelled.

    Args:
        c (Tensor): Complex spectrogram tensor with shape [channels, frequency, time]
                    where channels should be 2 (left and right)
        significance_threshold (float, optional): Threshold to filter out extreme ratio values.
                Values with magnitude above this threshold will be set to zero. Defaults to 4.0.
    
    Returns:
        Tensor: Interaural ratio spectrogram (right/left) with extreme values filtered out.
               Values represent the ratio between right and left channels.
    """
    ## TODO Use the log coefficients (dB)
    c_L,c_R = c
    # (Deleforge, Horoaud 2012) method for computing interaural spectrograms
    # 2D sound source localisation on the binaural manifold
    c_I = c_R / c_L
    return torch.nan_to_num(c_I) # Replace 0-division with the largest ratio storable


def lr_from_interaural(c_centre: Tensor, c_I: Tensor):
    """
    Computes from the interaural ratio a spectrogram from the centre signal.
    Args:
        c_centre (Tensor): Complex spectrogram tensor of the centre signal with shape [frequency, time].
        c_I (Tensor): Interaural ratio spectrogram tensor with shape [frequency, time].
    
    Returns:
        Tensor: Complex spectrogram tensor with shape [channels, frequency, time] where channels are left and right.
    """
    if c_I == None:
        c_I = tensor([0])

    P_L = 2 / (c_I + 1)
    P_R = 2 * c_I / (c_I + 1)

    c_lr = torch.stack((c_centre * P_L, c_centre * P_R), dim=0)
    
    assert not torch.any(c_lr.isnan())
    return c_lr