"""Qwen3-Omni thinker with Clef's joint schema head.

The thinker reads video frames and the clip's audio on one interleaved timeline
(`use_audio_in_video=True`). The head turns its final hidden states into one logit per
allowed option, exactly as in Clef. The head starts from scratch, because Clef's
trained head expects a different backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from vjev import clef_schema as cs

OMNI_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
FPS = 2.0
SR = 16_000
# Clef-flash's head sizes (joint_head_config.json), with the hidden size taken from the thinker.
HEAD_CONFIG = {"width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}
# LoRA on the text model's attention only. Fullmatch, so the audio tower's layers are excluded.
LORA_TARGETS = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
# Optional: also adapt the vision tower's attention, so frame features can tune to retention.
VISION_LORA_TARGETS = r"visual\.blocks\.\d+\.attn\.(qkv|proj)"
MEDIA_KEYS = ("input_features", "feature_attention_mask", "pixel_values_videos", "video_grid_thw",
              "video_second_per_grid")
PREFIX = f"<|im_start|>system\n{cs.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"


@dataclass
class Encoded:
    record: cs.EncodedRecord
    media: dict[str, torch.Tensor]
    audio: bool


def moment_bins(n_frames: int, fps: float, bin_s: float | None = None) -> list[tuple[float, float]]:
    """Time range of each option of a `moment` question. The video processor merges frames
    in pairs, so the finest step is 2 / fps seconds."""
    step = 2 / fps
    groups = n_frames // 2
    k = max(1, round((bin_s or step) / step))
    return [(i * step, min(i + k, groups) * step) for i in range(0, groups, k)]


def video_group_spans(ids: list[int], video_id: int, per_group: int) -> list[tuple[int, int]]:
    """Token span of each frame pair, in time order. Video tokens come in blocks of
    per_group (grid h * w / 4); with audio, audio tokens sit between the blocks."""
    pos = [i for i, t in enumerate(ids) if t == video_id]
    return [(pos[i], pos[i + per_group - 1] + 1) for i in range(0, len(pos), per_group)]


def _moment_to_score(record: dict, n_frames: int, fps: float) -> tuple[dict, dict[str, list]]:
    """A `moment` question becomes a `score` question with one option per time bin. Its option
    spans are later moved onto that bin's video (and audio) tokens."""
    questions, bins = {}, {}
    for k, q in record["questions"].items():
        if q["type"] == "moment":
            b = moment_bins(n_frames, fps, q.get("bin_s"))
            bins[k] = b
            q = {"type": "score", "instructions": q["instructions"],
                 "criteria": [f"{t0:g} to {t1:g} s" for t0, t1 in b]}
        questions[k] = q
    return {**record, "questions": questions}, bins


def encode(processor, record: dict, frames: np.ndarray, audio: np.ndarray | None,
           fps: float = FPS) -> Encoded:
    """Encode text with Clef's format, then splice the media tokens in after the prefix."""
    tok = processor.tokenizer
    if len(frames) % 2:  # the video processor groups frames in pairs
        frames = frames[:-1]
    record, bins = _moment_to_score(record, len(frames), fps)
    base = cs.encode_record(tok, record)
    kwargs = {"fps": fps, "do_sample_frames": False, "return_tensors": "pt"}
    if audio is not None:
        n = int(len(frames) / fps * SR)
        audio = np.pad(audio[:n], (0, max(0, n - len(audio))))  # silent clips get silence
        out = processor(text=["<|vision_start|><|video_pad|><|vision_end|>\n"], videos=[frames],
                        audio=[audio], use_audio_in_video=True, sampling_rate=SR, **kwargs)
    else:
        out = processor(text=["<|vision_start|><|video_pad|><|vision_end|>\n"], videos=[frames],
                        **kwargs)
    media_ids = out["input_ids"][0].tolist()
    p, m = len(cs._tokens(tok, PREFIX)), len(media_ids)
    ids = base.input_ids
    questions = tuple(
        replace(q,
                question_span=(q.question_span[0] + m, q.question_span[1] + m),
                option_spans=tuple((a + m, b + m) for a, b in q.option_spans))
        for q in base.questions
    )
    if bins:
        _t, h, w = (int(x) for x in out["video_grid_thw"][0])
        groups = video_group_spans(media_ids, tok.convert_tokens_to_ids("<|video_pad|>"),
                                   h * w // 4)
        assert len(groups) == len(frames) // 2, (len(groups), len(frames))
        step = 2 / fps
        moved = []
        for q in questions:
            if q.question_id in bins:
                spans = []
                for t0, t1 in bins[q.question_id]:
                    a, b = round(t0 / step), round(t1 / step)
                    spans.append((p + groups[a][0], p + groups[b - 1][1]))
                q = replace(q, option_spans=tuple(spans))
            moved.append(q)
        questions = tuple(moved)
    record_out = replace(base, input_ids=(*ids[:p], *media_ids, *ids[p:]), questions=questions)
    media = {}
    for k in MEDIA_KEYS:
        if k in out:
            v = out[k]
            media[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    return Encoded(record_out, media, audio is not None)


class TypedOmni(torch.nn.Module):
    def __init__(self, thinker, head: cs.JointSchemaHead, out_weight: torch.Tensor):
        super().__init__()
        self.thinker = thinker
        self.head = head
        # lm_head is replaced with Identity, so this is the only copy of the output embeddings.
        self.register_buffer("out_weight", out_weight, persistent=False)

    def forward(self, enc: Encoded) -> list[torch.Tensor]:
        dev = self.out_weight.device
        ids = torch.tensor([enc.record.input_ids], device=dev)
        mask = torch.ones_like(ids)
        media = {k: v.to(dev) for k, v in enc.media.items()}
        for k in ("input_features", "pixel_values_videos"):
            if k in media:
                media[k] = media[k].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.thinker(input_ids=ids, attention_mask=mask, use_cache=False,
                                  use_audio_in_video=enc.audio, **media).logits
        # The head runs in fp32. Two bf16 runs went NaN near step 720, and this is the likely cause.
        return self.head(hidden.float(), ids, mask, [enc.record], self.out_weight)[0]


def load(path: str, lora_r: int, lora_alpha: int, dropout: float = 0.05,
         vision_lora: bool = False):
    """Return (model, processor). Only LoRA and the head are trainable."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa")
    full.disable_talker()
    thinker = full.thinker
    thinker.requires_grad_(False)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out_weight = thinker.lm_head.weight.detach().float()
    thinker.lm_head = torch.nn.Identity()  # we only need hidden states, not 152k-way logits
    targets = f"({LORA_TARGETS}|{VISION_LORA_TARGETS})" if vision_lora else LORA_TARGETS
    thinker = get_peft_model(thinker, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout, target_modules=targets))
    hidden = full.config.thinker_config.text_config.hidden_size
    head = cs.JointSchemaHead(hidden_size=hidden, dropout=0.1, **HEAD_CONFIG).to("cuda")
    processor = Qwen3OmniMoeProcessor.from_pretrained(path)
    return TypedOmni(thinker, head, out_weight), processor
