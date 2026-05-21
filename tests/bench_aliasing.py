#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
import statistics
import time
import resource
import numpy as np

import p3.aoSystem as aoSystemMain
from p3.aoSystem import gpuEnabled, cp, asnumpy
from p3.aoSystem.fourierModel import fourierModel


def p3_root():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def cpu_peak_mb():
    # On macOS ru_maxrss is in bytes; on Linux it is typically KB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10_000_000:  # heuristic for bytes
        return rss / (1024 * 1024)
    return rss / 1024


def gpu_pool_used_mb():
    if not gpuEnabled:
        return 0.0
    try:
        used = cp.get_default_memory_pool().used_bytes()
        if used is None:
            return 0.0
        return float(used) / (1024 * 1024)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Benchmark aliasingPSD time and memory.')
    parser.add_argument('--ini', default='tests/scao_test_wvl1100nm.ini', help='Path to ini file, relative to P3 root.')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--method', choices=['chunked', 'streaming', 'limited'], default='chunked')
    parser.add_argument('--shift-batch', type=int, default=8)
    parser.add_argument('--layer-chunk', type=int, default=5)
    parser.add_argument('--n-times-limit', type=int, default=2)
    parser.add_argument('--compare-reference', action='store_true', help='Compute rel error vs chunked baseline once.')
    parser.add_argument('--no-precompute', action='store_true', help='Disable persistent aliasing precompute cache.')
    args = parser.parse_args()

    root = p3_root()
    ini = str(pathlib.Path(root) / args.ini)

    model = fourierModel(
        ini,
        path_root=root,
        calcPSF=False,
        display=False,
        verbose=False,
        reduce_memory=False,
    )
    model.clearAliasingPrecompute()

    # Warmup
    use_precompute = not args.no_precompute
    for _ in range(args.warmups):
        kwargs = {
            'method': args.method,
            'shift_batch': args.shift_batch,
            'layer_chunk': args.layer_chunk,
            'use_precompute': use_precompute,
        }
        if args.method == 'limited':
            kwargs['n_times_limit'] = args.n_times_limit
        _ = model.aliasingPSD(**kwargs)

    rel_err = None
    if args.compare_reference and args.method != 'chunked':
        ref = asnumpy(model.aliasingPSD(method='chunked', layer_chunk=args.layer_chunk))
        test = asnumpy(
            model.aliasingPSD(
                method=args.method,
                shift_batch=args.shift_batch,
                layer_chunk=args.layer_chunk,
                n_times_limit=args.n_times_limit if args.method == 'limited' else None,
                use_precompute=use_precompute,
            )
        )
        rel_err = np.linalg.norm(test - ref) / max(np.linalg.norm(ref), 1e-30)

    t_samples = []
    cpu_peaks = []
    gpu_used = []

    for _ in range(args.repeats):
        t0 = time.perf_counter()
        kwargs = {
            'method': args.method,
            'shift_batch': args.shift_batch,
            'layer_chunk': args.layer_chunk,
            'use_precompute': use_precompute,
        }
        if args.method == 'limited':
            kwargs['n_times_limit'] = args.n_times_limit
        _ = model.aliasingPSD(**kwargs)
        t_samples.append((time.perf_counter() - t0) * 1000.0)
        cpu_peaks.append(cpu_peak_mb())
        gpu_used.append(gpu_pool_used_mb())

    print('Aliasing benchmark')
    print(f'  ini: {ini}')
    print(f'  gpuEnabled: {gpuEnabled}')
    print(f'  method: {args.method}')
    print(f'  use_precompute: {use_precompute}')
    if args.method == 'limited':
        print(f'  n_times_limit: {args.n_times_limit}')
    print(f'  samples_ms: {[round(x, 3) for x in t_samples]}')
    print(f'  avg_ms: {statistics.mean(t_samples):.3f}')
    print(f'  med_ms: {statistics.median(t_samples):.3f}')
    print(f'  stdev_ms: {statistics.stdev(t_samples) if len(t_samples) > 1 else 0.0:.3f}')
    print(f'  peak_cpu_mb(max): {max(cpu_peaks):.3f}')
    print(f'  precompute_cache_mb(est): {model.aliasingPrecomputeMemoryMB():.3f}')
    if rel_err is not None:
        print(f'  rel_err_vs_chunked: {rel_err:.3e}')
    if gpuEnabled:
        print(f'  peak_gpu_pool_mb(max): {max(gpu_used):.3f}')


if __name__ == '__main__':
    main()
