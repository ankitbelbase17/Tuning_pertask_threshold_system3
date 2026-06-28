"""
visionzip.py — training-free visual token pruning (VisionZip) for Qwen3-VL.

Port of https://github.com/JIA-Lab-research/VisionZip (Qwen2.5-VL) to Qwen3-VL.
Two ideas, both training-free and TEXT-AGNOSTIC (no LLM/question involved):

  1. Dominant selection: tap the LAST vision block's self-attention and keep the
     tokens that RECEIVE the most attention (column-sum of the attention map).
     These few tokens already aggregate most of the scene's information.
  2. Contextual merging: the non-dominant tokens are not thrown away -- they are
     grouped by KEY (K) cosine-similarity (uniform-split target leaders + nearest
     assignment) and averaged into a small set of "contextual" tokens, so
     background context survives as a few representative tokens.

Qwen3-VL specifics (vs the 2.5 reference):
  * No window attention -> no window_index / reverse_indices permutation.
  * The PatchMerger fuses spatial_merge_unit (=4) CONSECUTIVE patches into one
    LLM token, so per-patch attention is downsampled by averaging groups of 4 to
    line up with the merged [1, N, H] tokens that embed_frame returns.
  * DeepStack features are already dropped by embed_frame, so pruning the merged
    tokens needs no extra bookkeeping.

The vision attention forward normally discards its weights, so we monkeypatch
ONLY the last block's attention with an eager equivalent that stashes the
attention map + keys. Enabled solely when cfg.prune_img_tokens is set; the normal
path is untouched otherwise.
"""
import types
import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision


def _capturing_attn_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """Eager attention identical to Qwen3VLVisionAttention.forward, but it keeps
    the attention map and post-rotary keys on the module. One image per call =>
    a single attention chunk, so we attend over the whole sequence (correct for
    embed_frame, which encodes one frame at a time)."""
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.transpose(0, 1).unsqueeze(0)          # [1, heads, S, hd]
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(seq_length, -1)
    out = self.proj(out)

    self._vz_attn = attn.detach()               # [1, heads, S, S]
    self._vz_key = k.detach()                   # [1, heads, S, hd]
    return out


def enable_capture(visual):
    """Patch the last vision block's attention to capture its weights + keys."""
    last_attn = visual.blocks[-1].attn
    last_attn.forward = types.MethodType(_capturing_attn_forward, last_attn)
    return last_attn


@torch.no_grad()
def prune_tokens(embeds, attn, key, merge_unit, dominant_frac, contextual_frac):
    """Reduce merged tokens `embeds` [1, N, H] using the captured patch-level
    attention `attn` [1, heads, S, S] and keys `key` [1, heads, S, hd], where
    S = N * merge_unit. Returns [1, N', H] with N' = dominant + contextual."""
    N = embeds.shape[1]
    S = attn.shape[-1]
    if S != N * merge_unit:                      # shape mismatch -> skip safely
        return embeds

    # per-merged-token importance: attention RECEIVED, averaged over the 4 patches
    recv = attn[0].mean(0).sum(0)                # [S]  mean heads, sum over queries
    attn_tok = recv.view(N, merge_unit).mean(-1)  # [N]

    # per-merged-token key (mean over the 4 patches and over heads)
    k = key[0]                                   # [heads, S, hd]
    k = k.view(k.shape[0], N, merge_unit, k.shape[-1]).mean(2)  # [heads, N, hd]
    k = k.mean(0).unsqueeze(0).float()          # [1, N, hd]

    dominant_num = max(int(dominant_frac * N), 1)
    contextual_num = max(int(contextual_frac * N), 1)
    if dominant_num >= N:                        # nothing to prune
        return embeds

    # 1) dominant: top-k by received attention (kept in original order)
    dom_idx = torch.topk(attn_tok, dominant_num).indices
    keep_mask = torch.zeros(N, dtype=torch.bool, device=embeds.device)
    keep_mask[dom_idx] = True
    ctx_mask = ~keep_mask

    dominant = embeds[:, torch.sort(dom_idx).values]          # [1, D, H]

    # 2) contextual: merge the rest by key-similarity into contextual_num tokens
    k_ctx = k[:, ctx_mask]                                    # [1, M, hd]
    M = k_ctx.shape[1]
    if M == 0:
        return dominant
    k_ctx = k_ctx / k_ctx.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    contextual_num = min(contextual_num, M)

    step = max(1, M // contextual_num)
    target_idx = torch.arange(0, M, step, device=embeds.device)[:contextual_num]
    is_target = torch.zeros(M, dtype=torch.bool, device=embeds.device)
    is_target[target_idx] = True

    targets = k_ctx[:, target_idx]                           # [1, C, hd]
    to_merge = k_ctx[:, ~is_target]                          # [1, M-C, hd]

    h_ctx = embeds[:, ctx_mask]                              # [1, M, H]
    h_target = h_ctx[:, target_idx]                          # [1, C, H]
    h_to_merge = h_ctx[:, ~is_target].float()               # [1, M-C, H]

    if to_merge.shape[1] == 0:                               # all leaders, no merge
        contextual = h_target
    else:
        sim = torch.bmm(to_merge, targets.transpose(1, 2))   # [1, M-C, C]
        C = targets.shape[1]
        assign = torch.zeros(1, to_merge.shape[1], C, device=embeds.device)
        assign.scatter_(2, sim.argmax(dim=2, keepdim=True), 1.0)
        counts = assign.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [1, C, 1]
        agg = torch.bmm(assign.transpose(1, 2), h_to_merge) / counts  # [1, C, H]
        contextual = h_target.float() + agg                  # leader + group mean

    return torch.cat([dominant, contextual.to(embeds.dtype)], dim=1)  # [1, N', H]
