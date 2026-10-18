"""ComfyUI nodes for Video Jev: typed questions about a video, with a probability per answer."""

from __future__ import annotations

import json

import numpy as np
import torch

from . import engine
from .presets import build

CATEGORY = "Video Jev"
_CACHE: dict[tuple, tuple] = {}


def _frames(video=None, images=None, fps: float = 24.0, audio=None):
    """(T, H, W, 3) uint8 frames, their fps, mono audio and its sample rate."""
    if video is not None:
        c = video.get_components()
        images, fps, audio = c.images, float(c.frame_rate), audio if audio is not None else c.audio
    if images is None:
        raise ValueError("Connect a video, or images and their fps.")
    frames = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    wav, sr = None, 16000
    if audio is not None and audio.get("waveform") is not None:
        w = audio["waveform"]
        wav = w[0].float().mean(0).cpu().numpy() if w.ndim == 3 else w.float().mean(0).cpu().numpy()
        sr = int(audio["sample_rate"])
    return frames, float(fps), wav, sr


def _image(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0)[None]


SOURCE = {
    "optional": {
        "video": ("VIDEO", {"tooltip": "From Load Video, or any node that outputs a VIDEO."}),
        "images": ("IMAGE", {"tooltip": "Frames, if you have no VIDEO (set fps)."}),
        "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1,
                          "tooltip": "Frame rate of `images`. Ignored for VIDEO."}),
        "audio": ("AUDIO", {"tooltip": "Overrides the VIDEO's own audio."}),
    }
}


class VideoJevLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weights": ("STRING", {"default": "sp1d5r/video-jev",
                                   "tooltip": "Hugging Face repo or a local folder."}),
            "precision": (engine.PRECISIONS, {"default": "bf16", "tooltip":
                          "bf16: needs about 62 GB of VRAM. offload: the same model split "
                          "across GPU (max_gpu_gib) and system RAM; slower, for smaller GPUs."}),
            "max_gpu_gib": ("INT", {"default": 20, "min": 4, "max": 192,
                                    "tooltip": "GPU memory for offload mode."}),
        }}

    RETURN_TYPES = ("VJEV_MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, weights, precision, max_gpu_gib):
        key = (weights, precision, max_gpu_gib)
        if key not in _CACHE:
            try:
                import comfy.model_management as mm

                mm.unload_all_models()
                mm.soft_empty_cache()
            except ImportError:
                pass
            _CACHE.clear()
            _CACHE[key] = engine.load(weights, precision, max_gpu_gib)
        return (_CACHE[key],)


class VideoJevAsk:
    """Preset questions and your own Jev record, answered with a probability per option."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VJEV_MODEL",),
                "funniest_moment": ("BOOLEAN", {"default": True}),
                "magic_moment": ("BOOLEAN", {"default": False}),
                "laughing": ("BOOLEAN", {"default": True}),
                "which_sound": ("BOOLEAN", {"default": False}),
                "ai_generated": ("BOOLEAN", {"default": True}),
                "quality": ("BOOLEAN", {"default": False}),
                "described_moment": ("STRING", {"default": "", "tooltip":
                                     "Find when this happens, for example 'a man opens the fridge'."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip":
                           "The generation prompt, for the quality text-alignment question."}),
                "custom_record": ("STRING", {"default": "", "multiline": True, "tooltip":
                                  'Your own Jev record: {"questions": {"name": {"type": "noul", '
                                  '"instructions": "..."}}}'}),
                "custom_decode": (list(engine.PROFILES), {"default": "moment", "tooltip":
                                  "Decode settings for the custom record. moment: 2 fps, 60 s, audio; "
                                  "query: 1 fps, 150 s; sound: 2 fps, 10 s; ai: first 1.5 s; "
                                  "quality: 8 fps, 6 s, no audio."}),
            },
            **SOURCE,
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("answers_json", "chart")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, funniest_moment, magic_moment, laughing, which_sound, ai_generated,
            quality, described_moment, prompt, custom_record, custom_decode, video=None,
            images=None, fps=24.0, audio=None):
        choices = [name for on, name in [
            (funniest_moment, "Funniest moment"), (magic_moment, "Magic trick moment"),
            (bool(described_moment.strip()), "Find a described moment"), (laughing, "Laughing?"),
            (which_sound, "Which sound?"), (ai_generated, "AI-generated?"),
            (quality, "Video quality (5 scores)")] if on]
        groups = build(choices, described_moment, prompt)
        if custom_record.strip():
            rec = json.loads(custom_record)
            if not isinstance(rec.get("questions"), dict) or not rec["questions"]:
                raise ValueError('custom_record needs a non-empty "questions" object')
            groups[f"{custom_decode}:custom" if custom_decode in groups else custom_decode] = rec
        if not groups:
            raise ValueError("Turn on at least one question.")
        frames, fps, wav, sr = _frames(video, images, fps, audio)
        res = engine.ask(*model, groups, frames, fps, wav, sr)
        return (json.dumps(res, indent=1), _image(engine.chart(res)))


class VideoJevJudge:
    """Score a (generated) video: P(AI-generated) and the five VideoFeedback quality scores.
    Use the numbers to filter or rank generations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("VJEV_MODEL",),
            "prompt": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "Generation prompt; enables text alignment."}),
        }, **SOURCE}

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("p_ai_generated", "quality_mean", "visual_quality", "temporal_consistency",
                    "dynamic_degree", "text_alignment", "factual_consistency", "answers_json",
                    "chart")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, prompt, video=None, images=None, fps=24.0, audio=None):
        groups = build(["AI-generated?", "Video quality (5 scores)"], "", prompt)
        frames, fps, wav, sr = _frames(video, images, fps, audio)
        res = engine.ask(*model, groups, frames, fps, wav, sr)
        names = ["visual_quality", "temporal_consistency", "dynamic_degree", "text_alignment",
                 "factual_consistency"]
        scores = [res[n]["expected"] if n in res else -1.0 for n in names]  # 1 to 4; -1 = not asked
        asked = [s for s in scores if s >= 0]
        return (res["ai_generated"]["answers"]["AI-generated"], sum(asked) / len(asked), *scores,
                json.dumps(res, indent=1), _image(engine.chart(res)))


class VideoJevFindMoment:
    """When does something happen? Returns the span and its frames, ready to save as a clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("VJEV_MODEL",),
            "question": ("STRING", {"default": "At what point in the video does the funniest "
                                               "thing happen?", "multiline": True}),
            "mode": (["event (2 fps, first 60 s)", "described (1 fps, first 150 s)"], {
                "default": "event (2 fps, first 60 s)", "tooltip":
                "event: funny or magic moments, 1 s steps. described: a text query like "
                "'a man opens the fridge', 2 s steps."}),
        }, **SOURCE}

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("start_s", "end_s", "confidence", "clip_frames", "answers_json", "chart")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, question, mode, video=None, images=None, fps=24.0, audio=None):
        profile = "query" if mode.startswith("described") else "moment"
        text = question.strip()
        if profile == "query" and not text.lower().startswith("when"):
            text = f'When does this happen in the video: "{text}"?'  # the trained wording
        groups = {profile: {"state": {}, "questions": {"moment": {"type": "moment",
                                                                   "instructions": text}}}}
        frames, fps, wav, sr = _frames(video, images, fps, audio)
        res = engine.ask(*model, groups, frames, fps, wav, sr)
        sp = res["moment"]["span"]
        src = images if video is None else video.get_components().images
        a, b = int(sp["start_s"] * fps), max(int(sp["start_s"] * fps) + 1, int(sp["end_s"] * fps))
        return (sp["start_s"], sp["end_s"], sp["p_span"], src[a:min(b, len(src))],
                json.dumps(res, indent=1), _image(engine.chart(res)))


NODE_CLASS_MAPPINGS = {
    "VideoJevLoader": VideoJevLoader,
    "VideoJevAsk": VideoJevAsk,
    "VideoJevJudge": VideoJevJudge,
    "VideoJevFindMoment": VideoJevFindMoment,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoJevLoader": "Video Jev Loader",
    "VideoJevAsk": "Video Jev Ask",
    "VideoJevJudge": "Video Jev Judge (AI video quality)",
    "VideoJevFindMoment": "Video Jev Find Moment",
}
