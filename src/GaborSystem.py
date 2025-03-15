from dataclasses import dataclass
import json
from math import ceil
from torch import Tensor 
from src.scales import OctScale, MelScale, LinScale, Scale
from src.frame import nsgfwin, nsdual, calcwinrange


def n_coefs(M: Tensor, g: list[Tensor], sl: slice)->int:
    return int(max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(M[sl],g[sl])).item())


Oct = 0
Mel = 1
Lin = 2
    

@dataclass
class GaborSystemParams:
    scale: int
    n_b: int
    fmin: int
    fmax: int

    fs: int
    Ls: int

    reducedform: int

    @classmethod
    def from_json(cls, json_string: str):
        data = json.loads(json_string)
        return cls(**data)

@dataclass
class GaborSystem:
    params: GaborSystemParams
    scale: Scale
    
    ncoefs: int
    nn : int
    M: Tensor
    g: list[Tensor]
    gd: list[Tensor]

    wins: list[Tensor]

    @classmethod
    def from_params(cls, gs : GaborSystemParams):
        """
        Create a GaborSystem from a scale, sample rate, and signal length.
        
        Args:
            scale (Scale): The frequency scale to use (e.g., OctScale)
            fs (int): Sample rate in Hz
            Ls (int): Length of the signal in samples
            reducedform (bool, optional): Whether to use reduced form. Defaults to True.
            
        Returns:
            GaborSystem: A fully initialized GaborSystem object
        """
        if gs.scale == Oct:
            scale = OctScale(gs.n_b, gs.fmin, gs.fmax)
        elif gs.scale == Mel:
            scale = MelScale(gs.n_b, gs.fmin, gs.fmax)
        elif gs.scale == Lin:
            scale = LinScale(gs.n_b, gs.fmin, gs.fmax)

        g, rfbas, M = nsgfwin(scale.F, scale.Q, gs.fs, gs.Ls)

        assert 0 <= gs.reducedform <= 2
        sl = slice(gs.reducedform,len(g)//2+1-gs.reducedform)

        _n_coefs = n_coefs(M, g, sl)

        if gs.reducedform:
            rm = M[gs.reducedform:len(M)//2+1-gs.reducedform]
            M[:] = rm.max()
        else:
            M[:] = M.max()

        wins, nn = calcwinrange(g, rfbas, gs.Ls)
        gd = nsdual(g, wins, nn, M)
        # Set up the Gabor system
        return cls(gs, scale, _n_coefs, nn, M, g, gd, wins)