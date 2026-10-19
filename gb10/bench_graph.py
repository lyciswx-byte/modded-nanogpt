"""On-GB10 fwd+bwd benchmark: eager graph-safe attention vs CUDA graph replay.

Measures a fixed shape but changes GPU document boundaries on replay,
and verifies both output and gradients on each boundary. Does NOT touch
NCCL or n-gram parameters and is not a complete 2xGB10 training benchmark.
"""
import argparse
import statistics
import torch
from gb10.graph_attention import graph_sdpa


def time_cuda(call, iters):
    ms=[]
    for _ in range(iters):
        a=torch.cuda.Event(enable_timing=True)
        b=torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return statistics.median(ms), statistics.mean(ms)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,default=1024)
    p.add_argument("--qk",type=int,choices=(64,128),default=64)
    p.add_argument("--v",type=int,choices=(64,128),default=128)
    p.add_argument("--window",type=int,default=384)
    p.add_argument("--iterations",type=int,default=8)
    args=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,1):
        raise RuntimeError("Requires a real SM121 GB10")
    torch.manual_seed(12)
    n=args.tokens
    q=torch.randn(n,6,args.qk,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.randn(n,6,args.v,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    offsets=torch.tensor([0,n//4,n//2,n,n],device="cuda",dtype=torch.int32)
    def fwd_bwd():
        y=graph_sdpa(q,k,v,offsets,args.window,.13 if args.qk==64 else .085)
        grads=torch.autograd.grad(y.float().square().mean(),(q,k,v))
        return y,grads
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fwd_bwd()
    torch.cuda.current_stream().wait_stream(stream)
    eager_p50,eager_mean=time_cuda(fwd_bwd,args.iterations)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        result,grads=fwd_bwd()
    graph.replay()
    torch.cuda.synchronize()
    # Test fresh BOS boundaries on SAME graph, no recapture.
    offsets.copy_(torch.tensor([0,n//3,2*n//3,n,n],device="cuda",dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    replay_output=result.detach().clone()
    replay_grads=[g.detach().clone() for g in grads]
    expected,expected_grads=fwd_bwd()
    torch.testing.assert_close(replay_output.float(),expected.float(),atol=.02,rtol=.02)
    for a,b in zip(replay_grads,expected_grads):
        torch.testing.assert_close(a.float(),b.float(),atol=.07,rtol=.07)
    replay_p50,replay_mean=time_cuda(graph.replay,args.iterations)
    print(f"[GB10 graph] tokens={n} (qk,v)=({args.qk},{args.v}) window={args.window} "
          f"query_tile={__import__('os').environ.get('GB10_GRAPH_QUERY_TILE','128')} "
          f"segment_lookup={__import__('os').environ.get('GB10_GRAPH_SEGMENT_LOOKUP','separate')} "
          f"eager_p50_ms={eager_p50:.3f} graph_p50_ms={replay_p50:.3f} "
          f"eager_avg_ms={eager_mean:.3f} graph_avg_ms={replay_mean:.3f} "
          f"speedup={eager_p50/max(replay_p50,1e-9):.2f}x "
          f"reserved_GiB={torch.cuda.memory_reserved()/2**30:.2f}",flush=True)


if __name__=="__main__":
    main()
