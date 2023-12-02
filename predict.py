# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

# from typing import Optional
import zipfile
import os
from pydub import AudioSegment

from cog import BaseModel, BasePredictor, Input, Path

from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track

STEMS = ["vocals", "bass", "drums", "guitar", "piano", "other", "all"]

MODEL = "htdemucs_6s"

"""
class ModelOutput(BaseModel):
    vocals: Path
    bass: Path
    drums: Path
    guitar: Path
    piano: Path
    other: Path
"""

def merge_tracks(exclude: str, tracks: dict) -> str:
    """
    Merge all tracks except the one specified in 'exclude'.

    Args:
    exclude (str): The track to exclude from merging.

    Returns:
    AudioSegment: The merged audio segment.
    """
    merged_track = None
    for name, filepath in tracks.items():
        if name != exclude and filepath:
            track = AudioSegment.from_mp3(filepath)
            merged_track = track if merged_track is None else merged_track.overlay(track)

    if merged_track:
        merged_filename = os.path.join("/tmp", "merged_tracks.mp3")
        merged_track.export(merged_filename, format="mp3")

    return merged_filename

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = get_model(MODEL)
        
    def predict(
        self,
        audio: Path = Input(description="Input audio"),
        stem: str = Input(
            default="drums",
            choices=STEMS,
            description="Separate this audio source.",
        ),
        output_format: str = Input(
            default="mp3",
            choices=["mp3", "wav", "flac"],
            description="Choose the output format",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        wav = load_track(str(audio), self.model.audio_channels, self.model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            self.model,
            wav[None],
            device="cuda",
            split=True,
            shifts=1,
            overlap=0.25,
            progress=True,
        )[0]
        sources = sources * ref.std() + ref.mean()

        kwargs = {
            "samplerate": self.model.samplerate,
            "bitrate": 320,
            "clip": "rescale",
            "as_float": False,
            "bits_per_sample": 24,
        }

        output = {k: None for k in STEMS}

        for source, name in zip(sources, self.model.sources):
            out = f"/tmp/{name}.{output_format}"
            save_audio(source.cpu(), out, **kwargs)
            output[name] = Path(out)

        # merger = AudioMerger(output)
        # merger.save_tracks(stem,"/tmp")

        if stem == "all":
            with zipfile.ZipFile("tracks.zip", 'w') as zipf:
                for _, path in output.items():
                    if path is not None:
                        zipf.write(path)
        else:
            merged = merge_tracks(stem, output)
            with zipfile.ZipFile("tracks.zip", 'w') as zipf:
                zipf.write(merged)
                zipf.write(output[stem])

        return Path("tracks.zip") 
