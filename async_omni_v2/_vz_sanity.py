"""Offline sanity check: encode the first frame with and without VisionZip,
verify shapes/token counts and that the captured attention lines up (S == N*4)."""
import dataclasses, av
from config import AsyncOmniConfig
from backend import Qwen3VLBackend

VIDEO = "/iopsstor/scratch/cscs/dbartaula/system_3/Highlights ｜ France 3-1 Senegal ｜ FIFA World Cup 2026™ [n3JDGlOwMJ4].webm"


def first_frame():
    c = av.open(VIDEO)
    for fr in c.decode(video=0):
        img = fr.to_image()
        c.close()
        return img


def main():
    img = first_frame()
    base_cfg = AsyncOmniConfig(video_path=VIDEO, dtype="bfloat16", device="cuda:0")

    b = Qwen3VLBackend(base_cfg)
    e_full = b.embed_frame(img)
    print(f"[full]   tokens/frame = {e_full.shape[1]}  shape={tuple(e_full.shape)}")
    del b

    pcfg = dataclasses.replace(base_cfg, prune_img_tokens=True,
                               prune_dominant_frac=0.65, prune_contextual_frac=0.05)
    bp = Qwen3VLBackend(pcfg)
    e_pruned = bp.embed_frame(img)
    attn = bp._vz_attn_mod._vz_attn
    print(f"[prune]  captured attn shape = {tuple(attn.shape)} (S={attn.shape[-1]})")
    print(f"[prune]  tokens/frame = {e_pruned.shape[1]}  shape={tuple(e_pruned.shape)}")
    full_n = attn.shape[-1] // bp.merge_unit
    print(f"[check]  merged N (=S/{bp.merge_unit}) = {full_n}  "
          f"retention = {e_pruned.shape[1]}/{full_n} = {e_pruned.shape[1]/full_n:.2%}")
    print(f"[check]  finite = {bool(e_pruned.isfinite().all())}  dtype={e_pruned.dtype}")


if __name__ == "__main__":
    main()
