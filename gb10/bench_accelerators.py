"""On-GB10 microbenchmark for sm_121 kernels (fwd+bwd, CUDA events).

Run on EACH GB10 after parity tests:
 python -m gb10.bench_accelerators --kind all

Do not extrapolate one kernel benchmark to the 2-node training wall clock.
"""
import argparse
import statistics
import torch
import torch.nn.functional as F


def bench(name, call, repeats=20):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    elapsed=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True)
        b=torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record()
        b.synchronize()
        elapsed.append(a.elapsed_time(b))
    print(f"{name:34} p50={statistics.median(elapsed):.3f}ms "
          f"avg={statistics.mean(elapsed):.3f}ms "
          f"peak_allocated={torch.cuda.max_memory_allocated()/2**30:.3f}GiB",flush=True)


def attention(repeats):
    from gb10.attention import segmented_sdpa
    from gb10.fa4 import packed_fa4
    ends=torch.tensor([0,128,384,896,1792,1792],dtype=torch.int32)
    n=1792
    for d,v_d in ((64,128),(64,64),(128,128)):
        q=torch.randn(n,6,d,dtype=torch.bfloat16,device="cuda",requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(n,6,v_d,dtype=torch.bfloat16,device="cuda",requires_grad=True)
        for backend,impl in (("native",segmented_sdpa),("FA4-sm121",packed_fa4)):
            def run():
                y=impl(q,k,v,ends,640,0.13 if d==64 else 0.085)
                y.float().square().mean().backward()
                q.grad=k.grad=v.grad=None
            bench(f"attention-{backend}-({d},{v_d})",run,repeats)


def linear(repeats):
    from gb10.fp8_linear import linear as fp8_linear
    a=torch.randn(1024,768,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(2816,768,device="cuda")/20).to(torch.bfloat16).requires_grad_()
    for label,impl in (("BF16",F.linear),("FP8E4M3",fp8_linear)):
        def run():
            y=impl(a,w)
            y.float().square().mean().backward()
            a.grad=w.grad=None
        bench(f"MLP-projection-{label}",run,repeats)


def ce(repeats):
    from gb10.fused_fp8_ce import fused_loss
    T,V=128,14336
    x=(torch.randn(T,V,device="cuda")*1.3).to(torch.bfloat16).requires_grad_()
    target=torch.randint(0,V,(T,),device="cuda",dtype=torch.int64)
    prefix=torch.randint(0,V,(T,),device="cuda",dtype=torch.int64)
    w=torch.tensor([1.,0.5,0.25],device="cuda")
    pw=torch.tensor([0.2],device="cuda")
    def fused():
        fused_loss(x,target,prefix,w,pw,0,T).sum().backward()
        x.grad=None
    def reference():
        logp=F.log_softmax(23*torch.sigmoid((x.float()+5)/7.5),dim=-1)
        idx=torch.arange(T,device="cuda")
        loss=-logp[idx,target]*w[0]-logp[idx,prefix]*pw[0]
        for k in (1,2):
            loss[:T-k]-=w[k]*logp[:T-k,target[k:]]
        loss.sum().backward()
        x.grad=None
    bench("CE-BF16-reference",reference,repeats)
    bench("CE-FP8-Triton-fused",fused,repeats)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--kind",choices=("attention","linear","ce","all"),default="all")
    parser.add_argument("--repeats",type=int,default=20)
    args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,1):
        raise RuntimeError("Run in the SM121 GB10 container")
    print(f"[GB10] {torch.cuda.get_device_name()} CC={torch.cuda.get_device_capability()} "
          f"torch={torch.__version__} CUDA={torch.version.cuda}")
    if args.kind in ("attention","all"):
        attention(args.repeats)
    if args.kind in ("linear","all"):
        linear(args.repeats)
    if args.kind in ("ce","all"):
        ce(args.repeats)


if __name__=="__main__":
    main()
