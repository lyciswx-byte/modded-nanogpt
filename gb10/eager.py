"""Graph-free eager training step for sm_121 bring-up.

Keeps ANVIL/Adam gradient accumulation semantics, sparse row-gradient sink
and communication ordering. GPU graphs are deliberately NOT captured until
the native GB10 kernels are benchmarked and their pointer rules validated.
"""
import torch


class EagerStepRunner:
    def __init__(self, model, optimizer, ngram_table, value_embeds):
        self.model = model
        self.params = list(optimizer.param_cfgs)
        self.anvil = {p for p, cfg in optimizer.param_cfgs.items() if cfg.optim == "anvil"}
        self.ngram = ngram_table
        self.value_embeds = value_embeds
        self._pending = None

    def forward(self, step, batch, ngram_slots, cfg):
        sink = self.ngram.grad_sink(batch.inputs.numel())
        loss = self.model(batch.inputs, batch.targets, batch.cum_seqlens,
                          ngram_slots, cfg, ngram_sink=sink,
                          value_embed_grad=self.value_embeds.grad_accum).sum()
        self._pending = (loss, sink)

    def backward(self):
        loss, sink = self._pending
        grads = torch.autograd.grad(loss, self.params + [sink], allow_unused=True)
        for p, g in zip(self.params, grads[:-1]):
            if g is None:
                continue
            g = g.detach()
            if p in self.anvil:
                p.grad = g
            elif p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
        self._pending = None
        assert grads[-1] is not None, "n-gram gradient sink was not used"
        return grads[-1].detach()

    @staticmethod
    def hold_ngram_grad(grad, pending, event_this_step):
        # Eager grad outputs own storage and are not overwritten by replay.
        return grad


class NoFp8Refresh:
    """Defer weights' all-gather waits, without Hopper's FP8 refresh graphs."""
    def refresh_attn(self):
        pass

    def refresh_mlp(self, refresh_lm: bool):
        pass
