"""Report the visible CUDA device and time a small half-precision matrix product.

Run with CUDA_VISIBLE_DEVICES set to a single GPU index. The script prints the
PyTorch and CUDA versions, the device name and free memory, and the measured
throughput of twenty 4096 x 4096 fp16 matrix products.
"""

import time

import torch


def main() -> None:
    cudnn = torch.backends.cudnn.version()
    print(f"torch {torch.__version__}, cuda build {torch.version.cuda}, cudnn {cudnn}")
    print(f"cuda available: {torch.cuda.is_available()}, visible devices: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    print(f"device: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")
    n, reps = 4096, 20
    x = torch.randn(n, n, device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = reps * 2 * n**3 / elapsed / 1e12
    finite = bool(torch.isfinite(y).all())
    print(f"fp16 matmul {n}^3 x {reps}: {elapsed * 1e3:.0f} ms, {tflops:.1f} TFLOP/s, finite: {finite}")


if __name__ == "__main__":
    main()
