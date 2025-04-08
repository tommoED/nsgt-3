import torch
from torch import Tensor
from math import ceil

def nsgtf(
        f: Tensor,
        g: list[Tensor],
        wins: list[Tensor],
        nn: int, M: Tensor,
        reducedform: int
    ) -> Tensor:
    """
    Compute the Non-Stationary Gabor Transform (NSGT) of the input signal.

    This function applies the fast Fourier transform (FFT) to the input signal to convert it into the
    frequency domain. It then processes each frequency slice by multiplying with a corresponding analysis
    window (after necessary padding and FFT shift) and finally applies an inverse FFT to obtain the
    NSGT coefficients representing the signal in the time-frequency domain.

    Parameters:
        f (Tensor): Input signal tensor of shape (channel, signal_length) representing time-domain data.
        g (list[Tensor]): List of analysis window tensors.
        wins (list[Tensor]): List of window ranges (or index arrays) specifying the frequency bins to process for each window.
        nn (int): Total number of samples in the input signal (should match f's last dimension).
        M (Tensor): Tensor containing modulation parameters (e.g., window lengths or hop sizes) for each analysis window.
        reducedform (int): Specifies the level of frequency reduction for real signals (valid values are typically 0, 1, or 2).

    Returns:
        Tensor: 

    Notes:
        - Each analysis window in g is padded to a uniform length based on the maximum number of coefficients required.
        - An FFT shift is applied to the windowed data prior to the inverse FFT to ensure proper alignment.
    """

    fft : function = torch.fft.fft
    ifft : function = torch.fft.ifft

    assert 0 <= reducedform <= 2
    sl = slice(reducedform,len(g)//2+1-reducedform)

    maxLg = int(max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(M[sl],g[sl])))

    loopparams = []
    for mii, gii, win_range in zip(M[sl],g[sl],wins[sl]):
        Lg = len(gii)
        col = int(ceil(float(Lg)/mii))
        assert col*mii >= Lg
        assert col == 1

        p = (mii,win_range,Lg,col)
        loopparams.append(p)

    ragged_giis = [
        torch.nn.functional.pad(
            torch.unsqueeze(gii, dim=0), (0, maxLg-gii.shape[0])
        )
            for gii in g[sl]
    ]
    giis = torch.conj(torch.cat(ragged_giis))

    ft = fft(f)

    Ls = f.shape[-1]

    assert nn == Ls

    c = torch.zeros(f.shape[0], len(loopparams), maxLg, dtype=ft.dtype)

    for j, (mii,win_range,Lg,col) in enumerate(loopparams):
        t = ft[:, win_range]*torch.fft.fftshift(giis[j, :Lg])

        sl1 = slice(None,(Lg+1)//2)
        sl2 = slice(-(Lg//2),None)

        c[:, j, sl1] = t[:, Lg//2:]  # if mii is odd, this is of length mii-mii//2
        c[:, j, sl2] = t[:, :Lg//2]  # if mii is odd, this is of length mii//2

    return ifft(c)

from itertools import chain

def nsigtf(
        c: Tensor,
        gd: list[Tensor],
        wins: list[Tensor],
        nn: int,
        Ls: int,
        reducedform: int
    ) -> Tensor:
    fft : function = torch.fft.fft
    ifft : function = torch.fft.irfft

    F, T = 1, 2

    if reducedform:
        sl = lambda x: chain(
            x[reducedform            : len(gd)//2+1-reducedform],
            x[len(gd)//2+reducedform :    len(gd)+1-reducedform]
        )
    else:
        sl = lambda x: x
    
    Lgs = [len(gdii) for gdii in sl(gd)]

    maxLg = int(max(Lgs))

    ragged_gdiis = [
        torch.nn.functional.pad(
            torch.unsqueeze(gdii, dim=0), (0, maxLg-gdii.shape[0])
        )
              for gdii in sl(gd)
    ]
    gdiis = torch.conj(torch.cat(ragged_gdiis))

    assert type(c) == Tensor
    c_dtype = c.dtype
    fc = fft(c)

    fr    = torch.zeros(c.shape[0], nn, dtype=c_dtype)  # Allocate output
    temp0 = torch.empty(c.shape[0], maxLg, dtype=fr.dtype)  # pre-allocation

    fbins = c.shape[F]
        
    # Overlapp-add procedure
    for i in range(fbins):
        t = fc[:, i]
        Lg = Lgs[i]
        
        r = (Lg+1)//2
        l = (Lg//2)

        wr1 = sl(wins)[i][  :l]
        wr2 = sl(wins)[i][-r: ]

        t1 = temp0[:,     :r ]
        t2 = temp0[:, Lg-l:Lg]

        t1[:, :] = t[:,        :r]
        t2[:, :] = t[:, maxLg-l:maxLg]

        temp0[:, :Lg] *= gdiis[i, :Lg]
        temp0[:, :Lg] *= maxLg

        fr[:, wr1] += t2
        fr[:, wr2] += t1

    ftr = fr[:, :nn//2+1]
    sig = ifft(ftr, n=nn)
    sig = sig[:, :Ls] # Truncate the signal to original length (if given)
    return sig
