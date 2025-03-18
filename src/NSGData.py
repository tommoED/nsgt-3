import torchaudio as ta
from torch import Tensor, tensor
import torch
from dataclasses import dataclass
import json
from tifffile import TiffFile

from src.codec import c_from_tiff, c_to_tiff
from src.nsgt import nsgtf, nsigtf
from src.interaural import interaural
from src.GaborSystem import GaborSystem, GaborSystemParams




@dataclass
class NSGData:
    sourcefile: str
    stereo: bool
    GS: GaborSystem
    
    c: Tensor
    s: Tensor

    @classmethod
    def from_wav(cls, filepath: str, params: GaborSystemParams):
        s, fs = ta.load(filepath)
        Ls = s.shape[-1]
        params.Ls = Ls
        params.fs = fs

        gs = GaborSystem.from_params(params)

        c = nsgtf(s, gs.g, gs.wins, gs.nn, gs.M, gs.params.reducedform)
        stereo = c.shape[0] == 2
        I = interaural(c) if stereo else tensor([1.0])

        return cls(filepath, stereo, gs, c, s)
    
    @classmethod
    def from_tiff(cls, filepath: str, params: GaborSystemParams = None):
        """
        Load NSGT data from a TIFF file
        
        Args:
            filepath: Path to the TIFF file
            params: GaborSystem parameters
            denormalize: Whether to denormalize the magnitude and phase
            
        Returns:
            NSGData object with the loaded data
        """
        # Extract metadata from the TIFF file
        try:
            # Get the metadata from the ImageDescription tag
            with TiffFile(filepath) as tif:
                if hasattr(tif.pages[0], 'tags') and 'ImageDescription' in tif.pages[0].tags:
                    metadata_json = tif.pages[0].tags['ImageDescription'].value
                    metadata = json.loads(metadata_json)
                    
                    normalcoef = metadata.get('normalcoef', 1.0)
                    gabor_params = metadata.get('GaborSystem', {})
            
        except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
            print(f"Warning: Could not extract metadata from TIFF file: {e}")
            normalcoef = 1.0

        c = c_from_tiff(filepath, denormalization_constant=normalcoef)
        c = c.unsqueeze(0)

        gs_params = GaborSystemParams(**gabor_params)
        gs = GaborSystem.from_params(gs_params)

        s = nsigtf(c, gs.gd, gs.wins, gs.nn, gs.params.Ls, gs.params.reducedform)

        stereo = c.shape[0] == 2
        return cls(filepath, stereo, gs, c, s)
    
    def to_tiff(self, filepath: str, normalize = True, colormode = "miniswhite"):
        c_to_tiff(self.c, filepath, self.GS, normalize, colormode)

    
    # Notebook functions
    def audio(self):
        from IPython.display import Audio, display, HTML
        widget = Audio(self.s, rate=self.GS.params.fs)
        return display(widget)
    
    def show(self, fig = None):
        import matplotlib.pyplot as plt
        def spectrogram(x, ax, title, cmap='gray', n_labels=4):
            scale = self.GS.scale

            im = ax.imshow(x.squeeze(), origin='lower', aspect='auto', cmap=cmap)
            ax.set_title(title)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_yticks(range(len(scale.F)))
            ax.set_xticks([])
            step = len(scale.F) // (n_labels - 1)
            indices = list(range(0, len(scale.F), step))[:n_labels]
            labels = [f'{scale.F[i]:.0f}' if i in indices else '' for i in range(len(scale.F))]
            ax.set_yticklabels(labels)

        if fig is None:
            fig, axes = plt.subplots(3,1)
        else:
            axes = fig.subplots(3,1)
        
        spectrogram(self.c.abs(), axes[0], 'Magnitude (dB)')
        spectrogram(self.c.angle(), axes[1], 'Phase', cmap='hsv')

        time = torch.arange(self.s.shape[-1]) / self.GS.params.fs
        axes[2].plot(time, self.s.squeeze())        
        axes[2].set_title('Waveform')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')




