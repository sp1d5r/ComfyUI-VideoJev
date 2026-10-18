"""Video Jev without ComfyUI: load the model, resample frames and audio to each task's training
settings, and answer Jev records. nodes.py wraps this.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from PIL import Image

from vjev import clef_schema as cs
from vjev import model as vm

from .presets import MATCHED_FRAMES, MATCHED_SIDE, PROFILES, Profile

PRECISIONS = ["bf16", "offload"]
TITLES = {"ai_generated": "AI-generated?", "laughing": "Laughing?", "sound": "Which sound?"}


def load(weights: str = "sp1d5r/video-jev", precision: str = "bf16",
         max_gpu_gib: int = 20) -> tuple[vm.TypedOmni, object]:
    """bf16: about 62 GB of VRAM. offload: the same bf16 weights split between the GPU
    (max_gpu_gib) and system RAM, then disk; same answers, slower.

    4-bit loading is not offered: bitsandbytes leaves this model's fused expert weights in
    bf16, so it saved only 3 GB in our test."""
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    path = pathlib.Path(weights)
    if not path.exists():
        path = pathlib.Path(snapshot_download(weights, allow_patterns=[
            "adapter_model.safetensors", "head.safetensors", "vjev_config.json"]))
    cfg = json.loads((path / "vjev_config.json").read_text())
    base = snapshot_download(cfg["base_model"])
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    # offload loads on CPU first: the LoRA is merged into the weights before they are split
    # across devices, because applying it to offloaded (meta) weights breaks the model.
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0} if precision == "bf16" else {"": "cpu"})
    full.disable_talker()
    thinker = full.thinker
    thinker.requires_grad_(False)
    out_weight = thinker.lm_head.weight.detach().float().to("cuda")
    thinker.lm_head = torch.nn.Identity()
    thinker = get_peft_model(thinker, LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                                                 lora_dropout=0.0, target_modules=vm.LORA_TARGETS))
    set_peft_model_state_dict(thinker, load_file(path / "adapter_model.safetensors"))
    if precision == "offload":
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_max_memory

        thinker = thinker.merge_and_unload()
        device_map = infer_auto_device_map(
            thinker, max_memory={**get_max_memory(), 0: f"{max_gpu_gib}GiB"},
            no_split_module_classes=thinker._no_split_modules, dtype=torch.bfloat16)
        thinker = dispatch_model(thinker, device_map, main_device=0,
                                 offload_dir=str(pathlib.Path(base) / ".vjev_offload"))
    hidden = full.config.thinker_config.text_config.hidden_size
    head = cs.JointSchemaHead(hidden_size=hidden, dropout=0.0, **vm.HEAD_CONFIG)
    head.load_state_dict(load_file(path / "head.safetensors"))
    net = vm.TypedOmni(thinker, head.to("cuda"), out_weight).eval()
    return net, Qwen3OmniMoeProcessor.from_pretrained(base)


def prepare(frames: np.ndarray, fps_in: float, audio: np.ndarray | None, sr_in: int,
            p: Profile) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Resample (T, H, W, 3) uint8 frames at fps_in to the profile, like the training decode.
    Returns frames, 16 kHz mono audio or None, and the clip duration used."""
    duration = min(len(frames) / fps_in, p.seconds)
    n = max(2, round(duration * p.fps))
    idx = np.minimum((np.arange(n) / p.fps * fps_in).round().astype(int), len(frames) - 1)
    picked = frames[idx]
    h, w = picked.shape[1:3]
    if p.matched:
        picked = picked[:MATCHED_FRAMES]
        if len(picked) < MATCHED_FRAMES:
            picked = np.concatenate([picked, np.repeat(picked[-1:], MATCHED_FRAMES - len(picked), 0)])
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        picked = np.stack([np.asarray(Image.fromarray(f[top:top + s, left:left + s]).resize(
            (MATCHED_SIDE, MATCHED_SIDE), Image.BICUBIC)) for f in picked])
    elif max(h, w) != p.max_side:
        scale = p.max_side / max(h, w)
        def even(x: float) -> int:  # ffmpeg's -2: nearest even size, ties up
            return max(2, int(x / 2 + 0.5) * 2)

        size = (even(w * scale), even(h * scale))
        picked = np.stack([np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC)) for f in picked])
    if audio is None or not p.audio:
        return picked, None, duration
    mono = audio if audio.ndim == 1 else audio.mean(0)
    mono = mono[: int(duration * sr_in)]
    if sr_in != vm.SR and len(mono):
        t_out = np.arange(int(len(mono) / sr_in * vm.SR)) / vm.SR
        mono = np.interp(t_out, np.arange(len(mono)) / sr_in, mono)
    return picked, mono.astype(np.float32), duration


def ask(net, proc, groups: dict[str, dict], frames: np.ndarray, fps_in: float,
        audio: np.ndarray | None = None, sr_in: int = vm.SR) -> dict:
    """groups: {profile: Jev record} (see presets.build). Returns {question: result}."""
    results = {}
    for key, rec in groups.items():
        p = PROFILES[key.split(":")[0]]
        f, a, duration = prepare(frames, fps_in, audio, sr_in, p)
        rec = json.loads(json.dumps(rec))
        rec.setdefault("state", {})
        if p.duration_in_state:
            rec["state"].setdefault("duration_seconds", round(duration, 1))
        enc = vm.encode(proc, rec, f, a, fps=p.fps)
        with torch.no_grad():
            probs = [o.float().softmax(-1).cpu().numpy() for o in net(enc)]
        results.update(summarise(rec, probs, p.fps, len(f) - len(f) % 2))
    return results


def summarise(record: dict, probs: list[np.ndarray], fps: float, n_frames: int) -> dict:
    result = {}
    for (name, q), p in zip(record["questions"].items(), probs, strict=True):
        if q["type"] == "moment":
            bins = vm.moment_bins(n_frames, fps)
            result[name] = {"type": "moment", "question": q["instructions"],
                            "bins": [[round(a, 1), round(b, 1), round(float(x), 4)]
                                     for (a, b), x in zip(bins, p, strict=True)]}
            result[name]["span"] = span(result[name]["bins"])
        else:
            if q["type"] == "noul" and "criteria" not in q:
                labels = ["yes", "no"]  # Clef's default noul options are long sentences
            else:
                labels = [str(text).split(":")[0] if q["type"] == "score" else str(text)
                          for _key, text in cs.question_options(q)]
            result[name] = {"type": q["type"], "question": q["instructions"],
                            "answers": {lab: round(float(x), 4)
                                        for lab, x in zip(labels, p, strict=True)}}
            if q["type"] == "score":
                result[name]["expected"] = round(float((np.arange(len(p)) + 1) @ p), 3)
    return result


def span(bins: list[list[float]]) -> dict:
    """The peak bin grown over neighbours at 50% or more of the peak, the paper's R1 rule."""
    ps = [b[2] for b in bins]
    i = int(np.argmax(ps))
    lo = hi = i
    while lo > 0 and ps[lo - 1] >= 0.5 * ps[i]:
        lo -= 1
    while hi < len(ps) - 1 and ps[hi + 1] >= 0.5 * ps[i]:
        hi += 1
    return {"start_s": bins[lo][0], "end_s": bins[hi][1], "peak_s": bins[i][0],
            "p_peak": ps[i], "p_span": round(sum(ps[lo:hi + 1]), 4)}


def chart(results: dict, width: int = 900) -> np.ndarray:
    """Draw the answers as bars, (H, W, 3) uint8, for a preview or a post."""
    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=15)
    except TypeError:  # Pillow < 10.1
        font = small = ImageFont.load_default()
    rows: list[tuple] = []
    for name, r in results.items():
        rows.append(("title", TITLES.get(name, name.replace("_", " ").capitalize())))
        if r["type"] == "moment":
            rows.append(("hist", r["bins"], r["span"]))
        else:
            items = r["answers"].items()
            if r["type"] == "choice":
                items = sorted(items, key=lambda kv: -kv[1])
            best = max(r["answers"], key=r["answers"].get)
            rows += [("bar", lab, p, lab == best) for lab, p in items]
        rows.append(("gap",))
    heights = {"title": 30, "hist": 110, "bar": 24, "gap": 14}
    img = Image.new("RGB", (width, 20 + sum(heights[r[0]] for r in rows)), (250, 250, 252))
    d = ImageDraw.Draw(img)
    ink, grey, blue, track = (20, 24, 33), (148, 163, 184), (47, 85, 212), (226, 230, 238)
    y, x0, x1 = 12, 290, width - 70
    for r in rows:
        if r[0] == "title":
            d.text((16, y + 4), r[1], fill=ink, font=font)
        elif r[0] == "bar":
            _, lab, p, top = r
            d.text((24, y + 3), lab[:34], fill=ink, font=small)
            d.rounded_rectangle((x0, y + 7, x1, y + 17), 5, fill=track)
            d.rounded_rectangle((x0, y + 7, x0 + max(4, (x1 - x0) * p), y + 17), 5,
                                fill=blue if top else grey)
            d.text((x1 + 10, y + 3), f"{p:.0%}", fill=ink, font=small)
        elif r[0] == "hist":
            _, bins, sp = r
            hi = max(b[2] for b in bins) or 1.0
            bw = (width - 48) / len(bins)
            for k, (a, _b, p) in enumerate(bins):
                inside = sp["start_s"] <= a < sp["end_s"]
                bx = 24 + k * bw
                d.rectangle((bx, y + 80 - 70 * p / hi, bx + max(1, bw - 1), y + 80),
                            fill=blue if inside else grey)
            d.text((24, y + 84), "0 s", fill=ink, font=small)
            label = f"{sp['start_s']:g} to {sp['end_s']:g} s ({sp['p_span']:.0%})"
            d.text((width // 2 - 80, y + 84), label, fill=blue, font=small)
            d.text((width - 70, y + 84), f"{bins[-1][1]:g} s", fill=ink, font=small)
        y += heights[r[0]]
    return np.asarray(img)
