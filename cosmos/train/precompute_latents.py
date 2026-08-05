# 17프레임 창(stride 8)을 독립적으로 VAE 인코딩해 저장 — 추론과 동일한 인코딩 방식 보장.
# 산출물: latents/<episode행번호>_<시작프레임>.pt = {latent (16,5,40,64) bf16, act (16,6) fp16, motion float}
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
import bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch

from load_dit import CKPT_DIR
from so100_dataset import TRAIN_ROOT, INDEX_PATH, NUM_FRAMES, letterbox_batch, read_frames

OUT_DIR = Path(__file__).resolve().parent / "latents"
STRIDE = 8


def main() -> None:
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    device = torch.device("cuda")
    vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda")
    stats = json.loads((TRAIN_ROOT / "so100_action_statistics.json").read_text(encoding="utf-8"))
    a_mean = np.array(stats["mean"], dtype=np.float32)
    a_std = np.array(stats["std"], dtype=np.float32)

    import os

    episodes = pd.read_parquet(INDEX_PATH)
    episodes = episodes[episodes.split == "train"].reset_index(drop=True)
    OUT_DIR.mkdir(exist_ok=True)
    order = list(range(len(episodes)))
    if os.environ.get("REVERSE") == "1":
        order = order[::-1]  # 두 번째 워커용: 뒤에서부터 처리해 중간에서 만남

    done, t0 = 0, time.time()
    rows = list(episodes.itertuples())
    for n_proc, ep_i in enumerate(order):
        row = rows[ep_i]
        acts_all = np.stack(
            pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])["action"].to_numpy()
        ).astype(np.float32)
        starts = list(range(0, row.num_frames - NUM_FRAMES + 1, STRIDE))
        todo = [s for s in starts if not (OUT_DIR / f"{ep_i:06d}_{s:04d}.pt").exists()]
        if not todo:
            continue
        try:
            # 에피소드 영상은 한 번만 디코드
            frames_all = read_frames(TRAIN_ROOT / row.video, 0, row.num_frames)
        except Exception as e:
            print(f"skip ep{ep_i}: {e}", flush=True)
            continue
        for s in todo:
            clip = torch.from_numpy(frames_all[s : s + NUM_FRAMES]).float().permute(0, 3, 1, 2) / 255.0
            clip = letterbox_batch(clip).permute(1, 0, 2, 3)[None] * 2.0 - 1.0  # (1,3,17,320,512)
            with torch.no_grad():
                latent = vae.encode(clip.to(device, torch.bfloat16))[0].cpu()  # (16,5,40,64)
            act = (acts_all[s : s + NUM_FRAMES - 1] - a_mean) / a_std  # (16,6)
            motion = float(np.abs(np.diff(act, axis=0)).mean()) if len(act) > 1 else 0.0
            torch.save(
                {"latent": latent, "act": torch.from_numpy(act).to(torch.float16), "motion": motion},
                OUT_DIR / f"{ep_i:06d}_{s:04d}.pt",
            )
            done += 1
        if (n_proc + 1) % 200 == 0:
            el = time.time() - t0
            print(f"proc {n_proc + 1}/{len(order)} windows {done} elapsed {el / 60:.0f}min", flush=True)
    print(f"done: {done} windows", flush=True)


if __name__ == "__main__":
    main()
