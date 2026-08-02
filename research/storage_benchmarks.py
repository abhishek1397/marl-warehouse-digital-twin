"""RolloutBuffer and GAE Benchmark Suite measuring throughput, sampling, and memory transfer."""

import json
import os
import sys
import time
from typing import Dict

import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl.storage import RolloutBuffer, Transition, compute_gae


def run_storage_benchmarks() -> None:
    """Executes benchmark suite measuring trajectory insertion throughput, GAE computation, and sampling."""
    print("=" * 65)
    print("      ROLLOUT BUFFER & GAE STORAGE BENCHMARK SUITE")
    print("=" * 65)

    buffer_size = 10000
    mini_batch_size = 64
    num_epochs = 4

    buffer = RolloutBuffer(capacity=buffer_size)

    # 1. Measure Transition Insertion Throughput
    start_insert = time.perf_counter()
    for step in range(buffer_size):
        obs = torch.randn(32)
        trans = Transition(
            observation=obs,
            action=int(step % 8),
            reward=1.0,
            value_estimate=0.5,
            log_prob=-0.5,
            agent_id=f"robot_{step % 3}",
            timestep=step,
        )
        buffer.insert(trans)
    end_insert = time.perf_counter()

    insert_duration = end_insert - start_insert
    insertion_throughput = buffer_size / insert_duration

    # 2. Measure GAE Calculation Speed
    start_gae = time.perf_counter()
    buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95, normalize_adv=True)
    end_gae = time.perf_counter()

    gae_duration_ms = (end_gae - start_gae) * 1000.0

    # 3. Measure Mini-Batch Generator Sampling Latency
    start_sample = time.perf_counter()
    num_batches = 0
    for batch in buffer.get_generator(mini_batch_size=mini_batch_size, num_epochs=num_epochs):
        num_batches += 1
    end_sample = time.perf_counter()

    sample_duration_ms = (end_sample - start_sample) * 1000.0

    results = {
        "buffer_size": buffer_size,
        "insertion_throughput_trans_per_sec": round(insertion_throughput, 2),
        "gae_calculation_latency_ms": round(gae_duration_ms, 4),
        "total_mini_batches_generated": num_batches,
        "sampling_latency_ms": round(sample_duration_ms, 4),
    }

    # Print summary
    print(f"\n| Metric Name                         | Value              |")
    print("| ----------------------------------- | ------------------ |")
    print(f"| Insertion Throughput                | {results['insertion_throughput_trans_per_sec']:<18,.2f} trans/sec |")
    print(f"| GAE Advantage Calculation           | {results['gae_calculation_latency_ms']:<18.4f} ms |")
    print(f"| Mini-Batch Sampling ({num_epochs} Epochs)     | {results['sampling_latency_ms']:<18.4f} ms |")
    print(f"| Total Mini-Batches Generated        | {results['total_mini_batches_generated']:<18} batches |")

    # Export results JSON
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "storage_benchmark_results.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Storage benchmark results saved to: {out_file}\n")


if __name__ == "__main__":
    run_storage_benchmarks()
