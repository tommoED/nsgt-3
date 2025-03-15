import torch
from torch import Tensor, tensor
from math import ceil, log2, sqrt, log10

class Scale:
    dbnd = 1.e-8
    
    F : Tensor = None
    Q : Tensor = None

    def F_by_bnd(self, bnd_index: Tensor):
        return self.F[bnd_index]
    
    def Q_by_bnd(self, bnd_index: Tensor):
        return self.Q[bnd_index]

    def __init__(self, bnds):
        self.bnds = bnds

    def __len__(self):
        return self.bnds

class OctScale(Scale):
    def __init__(self, bpo, fmin, fmax, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bpo: bands per octave (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        lfmin = log2(fmin)
        lfmax = log2(fmax)
        bnds = int(ceil((lfmax-lfmin)*bpo))+1
        Scale.__init__(self, bnds+beyond*2)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        q = sqrt(self.pow2n)/(self.pow2n-1.)/2.
        self.n_b = bpo

        self.Q = tensor([q])
        self.F = self.fmin*self.pow2n**torch.arange(self.bnds)



class LinScale(Scale):
    def __init__(self, bnds, fmin, fmax, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        self.df = float(fmax-fmin)/(bnds-1)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)-self.df*beyond
        if self.fmin <= 0:
            raise ValueError("Frequencies must be > 0.")
        self.fmax = float(fmax)+self.df*beyond
        self.n_b = bnds

        self.F = torch.arange(self.bnds)*self.df+self.fmin
        self.Q = self.F / (self.df*2)


def hz2mel(f):
    "\\cite{shannon:2003}"
    if isinstance(f, Tensor):
        return torch.log10(f/700.+1.)*2595.
    else:
        return log10(f/700.+1.)*2595


def mel2hz(m):
    "\\cite{shannon:2003}"
    return (10.0**(m/2595.)-1)*700


class MelScale(Scale):
    def __init__(self, bnds, fmin, fmax, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        mmin = hz2mel(fmin)
        mmax = hz2mel(fmax)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mbnd = (mmax-mmin)/(bnds-1)  # mels per band
        self.mmin = mmin-self.mbnd*beyond
        self.mmax = mmax+self.mbnd*beyond
        self.n_b = bnds

        mels = torch.arange(bnds) * self.mbnd + self.mmin

        self.F = mel2hz(mels)

        odivs = (torch.exp(mels/-1127.)-1.) * (-781.177/self.mbnd)
        pow2n = 2.0 ** (1./odivs)
        self.Q = pow2n.sqrt()/ (pow2n-1.)/2.
    
