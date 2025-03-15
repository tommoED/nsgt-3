import torch
from torch import Tensor, tensor
from warnings import warn

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

--
        Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

def hannwin(l):
    r = torch.arange(l,dtype=float)
    r *= torch.pi*2./l
    r = torch.cos(r)
    r += 1.
    r *= 0.5
    return r

def nsgfwin(f : Tensor, q: Tensor, sr, Ls, min_win=4, do_warn=True) -> tuple[list, Tensor, Tensor]:
    """
    f: list of center frequencies
    q: list of Q-factors
    sr: sampling rate
    Ls: length of signal
    min_win: minimum window length

    Returns:
        g: list of window functions
        rfbas: list of center frequencies
        M: list of window lengths
    """

    nyquist_f = sr/2.0

    mask = f > 0
    lim = torch.argmax(mask.int())
    if mask.any():
        # f partly <= 0 
        f = f[lim:]
        q = q[lim:]
            
    mask = f >= nyquist_f
    lim = torch.argmax(mask.int())
    if mask.any():
        # f partly >= nf 
        f = f[:lim]
        q = q[:lim]
    
    # assert len(f) == len(q)
    assert torch.all((f[1:]-f[:-1]) > 0)  # frequencies must be monotonic
    assert torch.all(q > 0)  # all q must be > 0
    
    qneeded = f*(Ls/(8.*sr))
    if torch.any(q >= qneeded) and do_warn:
        warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))
    
    fbas : Tensor[int] = f
    lbas : int = len(fbas)
    frqs = torch.concatenate([
        tensor([0.0])
        ,f,
        tensor([nyquist_f])
    ])
    fbas = torch.concatenate([frqs, tensor([sr]) - torch.flip(f, (0,))])

    fbas *= float(Ls)/sr

    M : Tensor = torch.zeros(fbas.shape, dtype=int)

    M[0] = torch.round(2*fbas[1])
    for k in range(1,2*lbas+1):
        M[k] = torch.round(fbas[k+1]-fbas[k-1])
    M[-1] = torch.round(Ls-fbas[-2])

    M = torch.clip(M, min_win, torch.inf).to(int)

    g : list = [hannwin(m) for m in M]

    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = torch.round(fbas).to(int)

    return g, rfbas, M


def nsdual(g: list[Tensor], wins: list[Tensor], nn: Tensor, M: list[int]) -> list[Tensor]:
    # Construct the diagonal of the frame operator matrix explicitly
    x = torch.zeros(nn, dtype=float)
    for gi,mii,sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii

        x[sl] += xa

    gd = [gi/torch.fft.ifftshift(x[wi]) for gi,wi in zip(g,wins)]
    return gd



def calcwinrange(g : list, rfbas : Tensor, Ls : int) -> tuple[list[Tensor], int]:
    shift = torch.concatenate([
        (-rfbas[-1] % Ls).unsqueeze(0), 
        rfbas[1:]-rfbas[:-1]
    ])
    
    timepos = torch.cumsum(shift, 0)
    nn = int(timepos[-1].item())
    timepos -= shift[0] # Calculate positions from shift vector
    
    wins = []
    for gii,tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = torch.arange(-(Lg//2)+tpii, Lg-(Lg//2)+tpii, dtype=int)
        win_range %= nn

        wins.append(win_range)
        
    return wins,nn



