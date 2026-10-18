"""Small native-GPU reference checks for the retained training operators.

Invoked by gb10_kernel_probe.py; these checks are not a model/training benchmark.
"""
import torch
import torch.nn.functional as F


def relative_error(actual, expected, limit=0.15):
    assert torch.isfinite(actual).all().item()
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert error.item() < limit, f"relative L2 error {error.item():.6f} >= {limit}"


def check_compiled_eval_mlp():
    """Regression: GB10 fp8 training must not enter Hopper TMA in validation.

    The old GPT._mlp(..., fp8_args=None) path for GB10_MLP_BACKEND=fp8
    compiled linear_relu_square_kernel with 131072 bytes of shared memory,
    exceeding SM121's 101376-byte limit even after disabling pipelining.
    Run with that backend as well as bf16 and fp8_sm121.
    """
    from track_1_short.model.gpt import ACTIVE_MLP_BACKEND, GPT

    @torch.compile(dynamic=False, fullgraph=True)
    def evaluate(x, w1, w2):
        return GPT._mlp(None, x, w1, w2, None)

    w1, w2 = [torch.randn(2816, 768, device="cuda", dtype=torch.bfloat16) * 0.01
              for _ in range(2)]
    with torch.no_grad():
        for rows in (128, 129, 8192):
            x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16) * 0.5
            actual = evaluate(x, w1, w2)
            reference = F.linear(F.relu(F.linear(x.float(), w1.float())).square(), w2.float().T)
            torch.cuda.synchronize()
            assert actual.shape == x.shape and actual.dtype == torch.bfloat16
            relative_error(actual, reference, 0.03)
            print(f"[ok] compiled eval MLP backend={ACTIVE_MLP_BACKEND} rows={rows}: FP32 reference L2 < 0.03")


def check_ce():
    from track_1_short.perf.kernels.cross_entropy import (
        FusedSoftcappedCrossEntropy, _ce_logit_gemm, lm_head_inverse_scale, quantize_lm_head_dual,
    )
    from track_1_short.perf.kernels.sampled_cross_entropy import SampledSoftcappedCrossEntropy
    for width in (50304, 10240, 14336, 24576):
        rows, dim = 32, 64
        x = (torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
        w = (torch.randn(dim, 50304, device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_()
        xs, ws, gs = 100 / 448, 2 / 448, (0.75 / 8) / 448
        wr = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        wc = torch.empty_strided(w.shape, (1, dim), device="cuda", dtype=wr.dtype)
        quantize_lm_head_dual(w.detach(), wr, wc, lm_head_inverse_scale(ws, "cuda"))
        targets = torch.arange(rows, device="cuda", dtype=torch.int64) % width
        prefix = targets.clone()
        prefix[::2] = -1
        mtp = torch.ones(1, device="cuda")
        pw = torch.tensor([0.25], device="cuda")
        if width == 50304:
            loss = FusedSoftcappedCrossEntropy.apply(x, targets, mtp, prefix, pw, w, wc, wr, xs, ws, gs)
        else:
            # Nontrivial candidate mapping also exercises zeroing noncandidate wgrad.
            candidates = torch.arange(width, device="cuda") * 2
            pos = torch.full((50304,), -1, device="cuda", dtype=torch.int64)
            pos[candidates] = torch.arange(width, device="cuda")
            candidate_rows = wc.T[candidates].contiguous()
            loss = SampledSoftcappedCrossEntropy.apply(
                x, mtp, targets, prefix, pw, w, candidate_rows, candidate_rows.T.contiguous(), pos, xs, ws, gs)
        dx, dw = torch.autograd.grad(loss.sum(), (x, w))
        xq = (x.detach().div(xs).to(torch.float8_e4m3fn).float() * xs).requires_grad_()
        wq = (wc.detach().float() * ws).requires_grad_()
        selected = wq if width == 50304 else wq[:, candidates]
        raw = xq @ selected
        # The actual GEMM stores e4m3 logits. Straight-through rounding models
        # the custom backward, which differentiates the quantized forward.
        quant = _ce_logit_gemm(x.detach().div(xs).to(wr.dtype),
                              wc.T.contiguous().T if width == 50304 else candidate_rows.T, xs, ws).float()
        logits = raw + (quant - raw).detach()
        sigma = ((logits + 5) / 7.5).sigmoid()
        half_sigma = sigma.half().float()
        sigma = sigma + (half_sigma - sigma).detach()
        logp = (23 * sigma).log_softmax(-1)
        reference = -logp[torch.arange(rows, device="cuda"), targets]
        reference = reference - 0.25 * logp[torch.arange(rows, device="cuda"), prefix.clamp_min(0)] * (prefix >= 0)
        rdx, rdw = torch.autograd.grad(reference.sum(), (xq, wq))
        torch.testing.assert_close(loss, reference, atol=2e-4, rtol=2e-4)
        relative_error(dx, rdx)
        relative_error(dw, rdw)
        if width != 50304:
            assert torch.count_nonzero(dw[:, pos < 0]).item() == 0
        print(f"[ok] actual CE FP8 forward/backward width={width}; loss 2e-4, gradient relative L2 < 0.15")


def check_sparse_update():
    from track_1_short.perf.kernels.row_scatter import scatter_add_rows
    from track_1_short.perf.kernels.ngram_adam import claim_rows, adam_rows_, bring_rows_current
    ids = torch.tensor([1, 1, 7, 1, 7], device="cuda", dtype=torch.int32)
    # Binary-exact values make repeated atomic addition order irrelevant here.
    values = torch.arange(5, device="cuda", dtype=torch.float32)[:, None].expand(5, 768).contiguous() / 16
    merged = torch.zeros(8, 768, device="cuda", dtype=torch.bfloat16)
    scatter_add_rows(values, ids, merged)
    ref = torch.zeros_like(merged).index_add_(0, ids.long(), values.bfloat16())
    torch.testing.assert_close(merged, ref, atol=0, rtol=0)
    grad = merged[ids.long()].contiguous()
    p = torch.full_like(merged, 0.5)
    v = torch.ones(8, device="cuda")
    last = torch.zeros(8, device="cuda", dtype=torch.int32)
    history = torch.full((5,), 0.9, device="cuda")
    owners = torch.empty_like(last)
    claim_rows(ids, owners)
    adam_rows_(p, v, last, history, grad, ids, owners, 2, beta2=0.9, eps=1e-8,
               step_size=0.01, decay=0.001, grad_mul=1.0, sq_mul=0.1)
    for row in (1, 7):
        g = ref[row].float()
        expected_v = 0.9 ** 2 + 0.1 * g.square().mean()
        update = g / (expected_v.sqrt() + 1e-8) * 0.01
        expected_p = (0.5 - update - 0.0005).bfloat16()
        torch.testing.assert_close(v[row], expected_v, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(p[row], expected_p, atol=0, rtol=0)
        assert last[row].item() == 2
    bring_rows_current(p, v, last, history, torch.tensor([1, 7], device="cuda", dtype=torch.int32), 3)
    torch.testing.assert_close(v[[1, 7]], (0.9 ** 2 + 0.1 * ref[[1, 7]].float().square().mean(1)) * 0.9)
    assert torch.all(p[[0, 2, 3, 4, 5, 6]] == 0.5).item()
    print("[ok] repeated sparse-row merge + single row Adam commit + lazy moment replay")


def check_bf16_refresh():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from track_1_short.model.gpt import GPT, ACTIVE_MLP_BACKEND
    from track_1_short.perf.kernels.cross_entropy import lm_head_inverse_scale
    if ACTIVE_MLP_BACKEND != "bf16":
        raise RuntimeError("BF16 refresh regression check requires GB10_MLP_BACKEND=bf16")
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.01
    model = SimpleNamespace(
        mlp_quantize_calls=0, _quantize_mlp_weights_and_scales=Mock(side_effect=AssertionError("unused FP8 MLP refresh")),
        lm_head=SimpleNamespace(weight=w),
        _lm_head_f8_row=torch.zeros_like(w, dtype=torch.float8_e4m3fn),
        _lm_head_f8_col=torch.empty_strided(w.shape, (1, 64), dtype=torch.float8_e4m3fn, device="cuda"),
        _lm_head_inv_w_s=lm_head_inverse_scale(2 / 448, "cuda"),
    )
    GPT.quantize_mlp_fp8(model, refresh_lm=False)
    assert model.mlp_quantize_calls == 1
    assert torch.count_nonzero(model._lm_head_f8_row.float()).item() == 0
    GPT.quantize_mlp_fp8(model, refresh_lm=True)
    expected = w.div(2 / 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(model._lm_head_f8_row.float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(model._lm_head_f8_col.float(), expected, atol=0, rtol=0)
    assert model.mlp_quantize_calls == 2
    print("[ok] BF16 mode skips MLP quantizer, preserves bootstrap counter and bit-exact LM-head refresh")
    from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs
    from track_1_short.model.gpt import FP8_EXACT_SCALE_CALLS
    module = GPT.__new__(GPT)
    torch.nn.Module.__init__(module)
    module.lm_head = torch.nn.Module()
    module.lm_head.weight = torch.nn.Parameter(w)
    for name in ("_lm_head_f8_row", "_lm_head_f8_col", "_lm_head_inv_w_s"):
        module.register_buffer(name, getattr(model, name))
    module.register_buffer("_mlp_sentinel", torch.full((32,), 7.0, device="cuda"))
    module.attn_fp8 = torch.nn.Module()
    module.attn_fp8.register_buffer("state", torch.zeros(32, device="cuda"))
    module.quantize_attn_fp8 = lambda: module.attn_fp8.state.add_(1)
    module._quantize_mlp_weights_and_scales = model._quantize_mlp_weights_and_scales
    module.mlp_quantize_calls = FP8_EXACT_SCALE_CALLS
    refresh = Fp8RefreshGraphs(module)
    refresh.capture()
    assert set(refresh.captured) == {"attn", "mlp+lm_head"}
    assert all(not name.startswith("_mlp_") for name in refresh.mlp_outputs)
    refresh.refresh_attn()
    refresh.refresh_mlp(True)
    refresh.refresh_mlp(False)
    refresh.seal()
    with torch.no_grad():
        w.mul_(0.5)
    refresh.refresh_attn()
    refresh.refresh_mlp(True)
    torch.cuda.synchronize()
    expected = w.div(2 / 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(module._lm_head_f8_row.float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(module._lm_head_f8_col.float(), expected, atol=0, rtol=0)
    assert module.mlp_quantize_calls == FP8_EXACT_SCALE_CALLS + 3
    assert torch.all(module._mlp_sentinel == 7).item()
    print("[ok] BF16 refresh graphs capture/seal/replay live LM-head weights; no empty MLP graph")



def check_packed_qkv():
    from track_1_short.model.attention import Yarn
    from track_1_short.model.gpt import FP8_ATTN_X_SCALE, FP8_ATTN_GRAD_SCALE
    from track_1_short.perf.kernels.qkv_rope import PackedFP8QKV
    from track_1_short.perf.kernels.fp8_attention_quant import PackedQKVFP8Cache
    for qdim, vdim in ((64, 128), (128, 64), (64, 64), (128, 128)):
        variants = [(False, False), (True, False)]
        if qdim == 128:
            variants.append((False, True))
        for paired, offset in variants:
            tokens, heads, dim = 32, 6, 64
            x = (torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
            qw = (torch.randn(2 * heads * qdim, dim, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
            vw = (torch.randn(heads * vdim, dim, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
            cache = PackedQKVFP8Cache(1, qw.shape[0] + vw.shape[0], dim, "cuda")
            cache.refresh(qw.detach()[None], vw.detach()[None])
            xs, gs = [torch.tensor([s], device="cuda") for s in (FP8_ATTN_X_SCALE, FP8_ATTN_GRAD_SCALE)]
            gain = torch.tensor(0.75, device="cuda", requires_grad=True)
            xf = x.detach().div(xs).to(torch.float8_e4m3fn).contiguous()
            yarn = Yarn(qdim, tokens, paired, attn_scale=0.125, device="cuda")
            q, k, v = PackedFP8QKV(x, qw, vw, cache.row[0], cache.col[0], cache.scales,
                                   gain, xs, gs, yarn.factor1, yarn.factor2, heads, paired, offset, 64,
                                   xf, xf.T.contiguous())
            # Reference uses quantized GEMM inputs, BF16 epilogue, then independent
            # FP32 rms-norm/rotary math. STE represents the custom quantizer gradient.
            rx = (xf.float() * xs).requires_grad_()
            rw = (cache.row[0].float() * cache.scales).requires_grad_()
            rg = gain.detach().clone().requires_grad_()
            flat = (rx @ rw.T * rg).bfloat16().float()
            qk, rv = flat.split((qw.shape[0], vw.shape[0]), -1)
            qk = qk.reshape(tokens, 2 * heads, qdim)
            normalized = qk * (qk.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).rsqrt()
            swap = normalized.reshape(tokens, 2 * heads, qdim // 2, 2).flip(-1).flatten(-2)
            if paired:
                f1 = yarn.factor1.float().reshape(tokens, 2, qdim)[:, torch.arange(2 * heads, device="cuda") % 2]
                f2 = yarn.factor2.float().reshape(tokens, 2, qdim)[:, torch.arange(2 * heads, device="cuda") % 2]
            else:
                f1, f2 = yarn.factor1.float()[:, None], yarn.factor2.float()[:, None]
            rotated = (normalized * f1 + swap * f2).bfloat16()
            rq, rk = rotated[:, :heads], rotated[:, heads:]
            if offset:
                rk = torch.cat((rk[..., :64], torch.cat((normalized[:1, heads:, 64:],
                                                       normalized[:-1, heads:, 64:])).bfloat16()), -1)
            if paired:
                rq, rk = [t.reshape(tokens * 2, heads // 2, qdim) for t in (rq, rk)]
            rv = rv.bfloat16().reshape(tokens, heads, vdim)
            for actual, expected in zip((q, k, v), (rq, rk, rv)):
                torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
            # Random incoming gradients exercise Q/K norm derivatives and V packing.
            incoming = [torch.randn_like(t) * 0.01 for t in (q, k, v)]
            actual_grads = torch.autograd.grad((q, k, v), (x, qw, vw, gain), incoming)
            packed_grad, = torch.autograd.grad((rq, rk, rv), (flat,), incoming, retain_graph=True)
            reference_grads = torch.autograd.grad((rq, rk, rv), (rx, rw, rg), incoming)
            rdx, rdw, rdg = reference_grads
            for actual, expected in zip(actual_grads[:3], (rdx, rdw[:qw.shape[0]], rdw[qw.shape[0]:])):
                relative_error(actual, expected, 0.18)
            assert torch.isfinite(actual_grads[3]).item()
            # The scalar derivative uses the quantized backward GEMM and original
            # BF16 weights, with BF16 product/reduction rounding. A derivative of
            # the idealized forward using quantized weights is a different value,
            # especially when the Q/K norm's scale derivative cancels near zero.
            packed = (packed_grad.bfloat16().float() / gs).to(torch.float8_e4m3fn).float() * gs
            quant_dw = (packed.T @ rx.detach()).bfloat16()
            quant_dx = (packed @ rw.detach() * gain.detach()).bfloat16()
            for actual, expected in zip(actual_grads[:3], (
                    quant_dx, quant_dw[:qw.shape[0]] * gain.detach(), quant_dw[qw.shape[0]:] * gain.detach())):
                relative_error(actual, expected, 0.05)
            scalar_reference = (quant_dw * torch.cat((qw.detach(), vw.detach()))).sum().float()
            torch.testing.assert_close(actual_grads[3], scalar_reference, atol=0.03, rtol=0.15)
            print(f"[ok] packed FP8 QKV+RoPE QK={qdim} V={vdim} paired={paired} offset={offset}")


def check_anvil():
    from track_1_short.optim.anvil import (
        ANVIL_MAPS, FROBENIUS_MARGIN, FROBENIUS_EPS, RAIL_SLOW_BETA, anvil_cascade,
    )
    for rows, cols in ((128, 64), (64, 128)):
        g = torch.randn(1, rows, cols, device="cuda", dtype=torch.bfloat16) * 0.01
        velocity = torch.zeros(2, *g.shape, device="cuda")
        momentum, beta, weight = [torch.tensor(v, device="cuda") for v in (0.85, 0.85, 0.4385)]
        actual = anvil_cascade(g, velocity, momentum, False, beta, weight)
        expected_velocity = torch.stack((g.float() * (1 - beta), g.float() * (1 - RAIL_SLOW_BETA)))
        torch.testing.assert_close(velocity, expected_velocity, atol=1e-8, rtol=1e-5)
        blend = weight * expected_velocity[0] + (1 - weight) * expected_velocity[1]
        z = torch.lerp(g.float(), blend, momentum).bfloat16().float()
        # Independent dense FP32 spectral polynomial reference. Triton/ANVIL
        # rounds its intermediate GEMMs to BF16, so compare relative L2.
        z = z / (FROBENIUS_MARGIN * z.norm(dim=(-2, -1), keepdim=True) + FROBENIUS_EPS)
        for a, b, c in ANVIL_MAPS:
            gram = z.transpose(-2, -1) @ z if rows > cols else z @ z.transpose(-2, -1)
            polynomial = b * gram + c * (gram @ gram)
            z = a * z + (z @ polynomial if rows > cols else polynomial @ z)
        relative_error(actual, z, 0.12)
        print(f"[ok] ANVIL twin-rail/cascade {rows}x{cols}: momentum state + dense reference L2 < 0.12")


def benchmark_refresh():
    """Isolated before/after two-step refresh pair, using actual default-sized buffers."""
    import statistics
    from types import SimpleNamespace
    from track_1_short.model.gpt import GPT, NUM_LAYERS, NUM_MLP_SLOTS, FP8_EXACT_SCALE_CALLS, ACTIVE_MLP_BACKEND
    if ACTIVE_MLP_BACKEND != "bf16":
        raise RuntimeError("Refresh benchmark requires GB10_MLP_BACKEND=bf16")
    from track_1_short.perf.kernels.cross_entropy import quantize_lm_head_dual, lm_head_inverse_scale
    model = GPT.__new__(GPT)
    torch.nn.Module.__init__(model)
    model.device, model.mlp_hdim = "cuda", 2816
    model.mlp_bank = torch.nn.Parameter(torch.randn(NUM_MLP_SLOTS, 2, 2816, 768, device="cuda", dtype=torch.bfloat16) * 0.01)
    model.post_lambdas = torch.nn.Parameter(torch.ones(NUM_LAYERS, 2, device="cuda"))
    model.init_mlp_fp8(768)
    w = torch.randn(768, 50304, device="cuda", dtype=torch.bfloat16) * 0.005
    model.lm_head = SimpleNamespace(weight=w)
    model._lm_head_f8_row = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    model._lm_head_f8_col = torch.empty_strided(w.shape, (1, 768), dtype=torch.float8_e4m3fn, device="cuda")
    model._lm_head_inv_w_s = lm_head_inverse_scale(2 / 448, "cuda")
    model.mlp_quantize_calls = FP8_EXACT_SCALE_CALLS

    @torch.no_grad()
    def before():
        # Exact pre-change body for one non-Adam and one Adam update.
        model._quantize_mlp_weights_and_scales()
        model._quantize_mlp_weights_and_scales()
        quantize_lm_head_dual(w, model._lm_head_f8_row, model._lm_head_f8_col, model._lm_head_inv_w_s)

    def after():
        model.quantize_mlp_fp8(False)
        model.quantize_mlp_fp8(True)

    for body in (before, after):
        for _ in range(5):
            body()
    torch.cuda.synchronize()
    for label, body in (("before", before), ("after", after)):
        timings = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                body()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / 200)
        print(f"[timing] {label} eager refresh ms/optimizer-update median={statistics.median(timings):.6f}")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
        timings = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / 200)
        print(f"[timing] {label} captured refresh ms/optimizer-update median={statistics.median(timings):.6f}")
    print("Isolated refresh timings exclude attention, model and optimizer; not end-to-end throughput.")


def check_ce_mtp_boundaries():
    """Compare raw fused loss/gradient to autograd, including truncated tail rows."""
    from track_1_short.perf.kernels.cross_entropy import ce_fwd_bwd
    from track_1_short.perf.kernels.sampled_cross_entropy import sampled_ce_fwd_bwd
    rows, gs = 32, (0.75 / 8) / 448
    for width in (50304, 10240, 14336, 24576):
        logits = (torch.randn(rows, width, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        targets = torch.arange(rows, device="cuda", dtype=torch.int64) // 2
        prefix = targets.clone()
        prefix[::2] = -1
        pw = torch.tensor([0.25], device="cuda")
        for n_predict in (1, 2, 3):
            mtp = torch.tensor([1.0, 0.5, 0.25][:n_predict], device="cuda")
            loss = torch.empty(rows, device="cuda")
            grad = torch.empty(rows, width, dtype=torch.float8_e5m2, device="cuda")
            kernel = ce_fwd_bwd if width == 50304 else sampled_ce_fwd_bwd
            kernel(logits, targets, mtp, prefix, pw, loss, grad, rows, n_predict, 23.0, 5.0, 7.5, gs)
            raw = logits.float().requires_grad_()
            sigma = ((raw + 5) / 7.5).sigmoid()
            sigma = sigma + (sigma.half().float() - sigma).detach()
            logp = (23 * sigma).log_softmax(-1)
            reference = torch.stack([
                sum(-mtp[k] * logp[i, targets[i + k]] for k in range(n_predict) if i + k < rows)
                for i in range(rows)
            ])
            reference = reference - 0.25 * logp[torch.arange(rows, device="cuda"), prefix.clamp_min(0)] * (prefix >= 0)
            expected, = torch.autograd.grad(reference.sum(), raw)
            torch.testing.assert_close(loss, reference, atol=2e-4, rtol=2e-4)
            # FP8 e5m2 gradients have up to 12.5% rounding error. Check each
            # element, not just an L2 norm dominated by the target columns.
            torch.testing.assert_close(grad.float() * gs, expected, atol=2e-6, rtol=0.16)
            torch.testing.assert_close(grad[-1, 100:].float() * gs, expected[-1, 100:], atol=1e-6, rtol=0.16)
            print(f"[ok] CE MTP={n_predict} width={width}: truncated tail, repeated targets, prefix loss + gradients")
