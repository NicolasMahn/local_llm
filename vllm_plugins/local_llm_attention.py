"""Prefill attention that stays fast at long context on this GPU (a vLLM plugin).

vLLM's Triton attention kernel is shaped for generating one token at a time.
For prefill, each GPU work unit takes two prompt tokens and reads the whole
context on its own, so with Gemma 4's 512-wide full-attention heads a
512-token step grows by ~0.1 s per 1,000 tokens of context. Somewhere past
64k a single call outlasts the GPU watchdog (2 s), which resets the GPU and
takes the desktop with it. The same happens in Gemma's sliding-window layers
once a request holds an image: the kernel then stops skipping keys outside
the window.

Those cases are computed here as matrix multiplies over chunks of the
context, with the softmax folded in chunk by chunk (as flash attention does).
That runs close to the GPU's matrix speed: ~20x faster at 32k, in calls of
a few milliseconds, and a step at 256k takes ~2 s. Everything else
(generating tokens, other models) keeps using vLLM's kernel.
"""

import torch

CHUNK = 4096  # keys per matrix multiply; larger was not faster, only took more memory
NO_WINDOW = 1 << 30


def register():
    # Deliberately unguarded: if a vLLM update moves this function, the model
    # must fail to start, not run its long context on the slow kernel.
    import vllm.v1.attention.backends.triton_attn as triton_attn

    stock = triton_attn.unified_attention
    if getattr(stock, "_local_llm", False):
        return
    # Generating a token splits each KV head's context into this many parts;
    # vLLM's 16 leave most of this GPU idle at long context (32 work units
    # for Gemma). 64 made a 255k-token layer a third faster, short ones no slower.
    triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = 64

    def unified_attention(**kw):
        if takes_over(kw):
            prefill_attention(stock, **kw)
        else:
            stock(**kw)

    unified_attention._local_llm = True
    triton_attn.unified_attention = unified_attention


def takes_over(kw):
    """Plain causal prefill that is either 512 wide or windowed with images.

    Images must stay inside the window, as Gemma 4 has them: without that
    clamp vLLM's kernel has its own rules for image keys past the window.
    """
    q, window = kw["q"], kw["window_size"][0]
    windowed_images = (window >= 0 and kw.get("mm_prefix_range") is not None
                       and kw.get("mm_prefix_clamp_sliding_window"))
    long_way = q.shape[2] > 256 or windowed_images
    return (
        long_way
        and kw["max_seqlen_q"] > 1
        and kw["causal"] is True
        and q.dtype == kw["k"].dtype  # an unquantized KV cache
        and not kw.get("softcap")
        and all(kw.get(name) is None for name in
                ("alibi_slopes", "sinks", "output_scale", "qq_bias", "rswa_prefix_lens", "q_descale"))
        and kw.get("chunk_lookback", -1) == -1
        and not torch.cuda.is_current_stream_capturing()
    )


def prefill_attention(stock, *, q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                      softmax_scale, window_size, mm_prefix_range=None, **rest):
    starts = cu_seqlens_q.tolist()
    num_seqs = len(starts) - 1
    seq_lens = seqused_k[:num_seqs].tolist()
    ranges = mm_prefix_range.tolist() if mm_prefix_range is not None else [[]] * num_seqs
    window = window_size[0] if window_size[0] >= 0 else NO_WINDOW

    decoding = []
    for i in range(num_seqs):
        first, last = starts[i], starts[i + 1]
        if last - first <= 1:
            decoding.append(i)
            continue
        images = [(a, b) for a, b in ranges[i] if a < b]
        out[first:last] = attend(q[first:last], k, v, block_table[i], seq_lens[i], softmax_scale,
                                 window, images)

    if decoding:
        # The requests generating a token go to vLLM's kernel as a batch of their own.
        index = torch.tensor(decoding, device=q.device)
        rows = torch.tensor([starts[i] for i in decoding], device=q.device)
        part = torch.empty_like(out[rows])
        stock(q=q[rows], k=k, v=v, out=part,
              cu_seqlens_q=torch.arange(len(decoding) + 1, dtype=cu_seqlens_q.dtype, device=q.device),
              max_seqlen_q=1, seqused_k=seqused_k[index],
              max_seqlen_k=max(seq_lens[i] for i in decoding), block_table=block_table[index],
              softmax_scale=softmax_scale, causal=True, window_size=window_size,
              mm_prefix_range=mm_prefix_range[index] if mm_prefix_range is not None else None,
              **{name: value for name, value in rest.items() if name not in ("max_seqlen_q", "max_seqlen_k", "causal")})
        out[rows] = part


def attend(q, k_cache, v_cache, blocks, seq_len, scale, window, images):
    """Attention of one request's new tokens, which end its seq_len-long sequence.

    q is (tokens, query heads, head size); the caches are paged,
    (blocks, block size, KV heads, head size).
    """
    tokens, heads, size = q.shape
    block, kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = heads // kv_heads
    device = q.device
    # The query heads that share a KV head become rows of one matrix.
    rows = q.view(tokens, kv_heads, group, size).permute(1, 2, 0, 3).reshape(kv_heads, group * tokens, size)
    positions = torch.arange(seq_len - tokens, seq_len, device=device)
    image_ranges = torch.tensor(images or [(0, -1)], device=device)
    query_images = image_of(positions, image_ranges).repeat(group)
    positions = positions.repeat(group)

    first_key = max(0, seq_len - tokens - window)
    best = torch.full((kv_heads, group * tokens, 1), float("-inf"), device=device)
    total = torch.zeros((kv_heads, group * tokens, 1), device=device)
    acc = torch.zeros((kv_heads, group * tokens, size), device=device)
    for start in range(first_key, seq_len, CHUNK):
        end = min(start + CHUNK, seq_len)
        pages = blocks[start // block:(end + block - 1) // block]
        skip = start % block
        keys = k_cache[pages].reshape(-1, kv_heads, size)[skip:skip + end - start].permute(1, 2, 0)
        values = v_cache[pages].reshape(-1, kv_heads, size)[skip:skip + end - start].transpose(0, 1)
        key_positions = torch.arange(start, end, device=device)
        weights, best, total, rescale = softmax_step(
            torch.bmm(rows, keys), best, total, positions, query_images,
            key_positions, image_of(key_positions, image_ranges), scale, window, q.dtype)
        acc = torch.addcmul(torch.bmm(weights, values).float(), acc, rescale)
    out = (acc / total).to(q.dtype)
    return out.view(kv_heads, group, tokens, size).permute(2, 0, 1, 3).reshape(tokens, heads, size)


def image_of(positions, ranges):
    """Which image each position belongs to, -1 for text. Range ends are inclusive."""
    inside = (positions[:, None] >= ranges[:, 0]) & (positions[:, None] <= ranges[:, 1])
    return torch.where(inside.any(1), inside.int().argmax(1), -1)


@torch.compile(dynamic=True)
def softmax_step(scores, best, total, positions, query_images, key_positions, key_images,
                 scale, window, dtype):
    """Fold one chunk of keys into the running softmax.

    Visible: earlier keys inside the window, plus keys of the same image in
    either direction, also kept inside the window.
    """
    distance = positions[None, :, None] - key_positions[None, None, :]
    in_window = distance <= window
    same_image = (query_images[None, :, None] == key_images[None, None, :]) & (query_images[None, :, None] >= 0)
    visible = ((distance >= 0) | same_image) & in_window
    scores = (scores.float() * scale).masked_fill(~visible, float("-inf"))
    new_best = torch.maximum(best, scores.amax(-1, keepdim=True))
    # A row that saw no visible key yet keeps -inf; shift it by 0 instead of -inf.
    shift = torch.where(new_best == float("-inf"), 0.0, new_best)
    weights = (scores - shift).exp()
    rescale = (best - shift).exp()
    return weights.to(dtype), new_best, total * rescale + weights.sum(-1, keepdim=True), rescale
