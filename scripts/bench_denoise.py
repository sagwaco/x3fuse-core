#!/usr/bin/env python3
"""Benchmark full denoising stages and DNG conversion using fresh CLI processes.

Outputs are temporary. GPU mode is forced, so an unavailable adapter fails the
benchmark instead of silently measuring CPU fallback. Device creation, copies,
and all denoising stages count toward latency; filesystem caches may be warm.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "target/release/x3f_extract")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--backends", nargs="+", choices=["cpu", "gpu", "auto"], default=["cpu", "gpu"])
    parser.add_argument("--denoiser", choices=["nlm", "bm3d"], default="nlm")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    stage_pattern = re.compile(r"Denoise stage: [^\n]*, ([\d.]+) ms")
    writer = csv.writer(sys.stdout)
    writer.writerow(["file", "denoiser", "backend", "runs", "first_conversion_ms", "median_conversion_ms", "median_denoise_ms", "p95_denoise_ms"])
    with tempfile.TemporaryDirectory(prefix="x3f-denoise-bench-") as output:
        for path in args.files:
            for backend in args.backends:
                wall_times, denoise_times = [], []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    result = subprocess.run(
                        [str(args.binary.resolve()), "-v", "-dng", "-denoiser", args.denoiser, "-denoise", "10", "-denoise-backend", backend, "-o", output, str(path.resolve())],
                        capture_output=True, text=True, timeout=300,
                    )
                    wall_times.append((time.perf_counter() - start) * 1000)
                    log = result.stdout + result.stderr
                    if result.returncode:
                        raise RuntimeError(f"{path} ({backend}) failed:\n{log}")
                    stages = stage_pattern.findall(log)
                    if not stages:
                        raise RuntimeError(f"{path}: no denoising stage timings; use an instrumented build")
                    denoise_times.append(sum(map(float, stages)))
                p95 = sorted(denoise_times)[math.ceil(len(denoise_times) * 0.95) - 1]
                writer.writerow([path.name, args.denoiser, backend, args.iterations, *(f"{v:.3f}" for v in [wall_times[0], statistics.median(wall_times), statistics.median(denoise_times), p95])])
                sys.stdout.flush()


if __name__ == "__main__":
    main()
