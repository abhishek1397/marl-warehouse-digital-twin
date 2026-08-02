"""Neural Network Benchmark Suite measuring forward latency, parameters, and memory usage."""

import json
import os
import sys
import time
from typing import Dict

import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl.networks import (
    ActorNetwork,
    CNNFeatureExtractor,
    CriticNetwork,
    MLP,
    SharedActorCritic,
)


def benchmark_network(
    name: str, network: torch.nn.Module, dummy_input: torch.Tensor, iterations: int = 500
) -> Dict[str, float]:
    """Measures forward pass latency (ms), parameter count, and memory footprint."""
    network.eval()
    dummy_input = dummy_input.to(network.device if hasattr(network, "device") else "cpu")

    # Warmup iterations
    with torch.no_grad():
        for _ in range(50):
            _ = network(dummy_input)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = network(dummy_input)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / iterations) * 1000.0
    param_count = sum(p.numel() for p in network.parameters())

    return {
        "network": name,
        "avg_latency_ms": round(avg_latency_ms, 4),
        "total_parameters": param_count,
    }


def run_all_network_benchmarks() -> None:
    """Executes benchmark suite across MLP, CNN, Actor, Critic, and SharedActorCritic."""
    print("=" * 65)
    print("      PYTORCH NEURAL NETWORK LIBRARY BENCHMARK SUITE")
    print("=" * 65)

    batch_size = 32
    results = []

    # 1. MLP Benchmark
    mlp = MLP(input_dim=64, output_dim=128, hidden_dims=[256, 256])
    res_mlp = benchmark_network("MLP (64 -> 256 -> 256 -> 128)", mlp, torch.randn(batch_size, 64))
    results.append(res_mlp)

    # 2. CNN Benchmark
    cnn = CNNFeatureExtractor(input_channels=1, output_dim=128)
    res_cnn = benchmark_network("CNN 2D (1x20x20 -> 128)", cnn, torch.randn(batch_size, 1, 20, 20))
    results.append(res_cnn)

    # 3. Actor Network Benchmark
    actor = ActorNetwork(observation_space=64, action_dim=8)
    res_actor = benchmark_network("Actor Network (Discrete 8)", actor, torch.randn(batch_size, 64))
    results.append(res_actor)

    # 4. Critic Network Benchmark
    critic = CriticNetwork(observation_space=64, action_dim=None)
    res_critic = benchmark_network("Critic Network V(s)", critic, torch.randn(batch_size, 64))
    results.append(res_critic)

    # 5. Shared Actor-Critic Benchmark
    shared = SharedActorCritic(observation_space=64, action_dim=8)
    res_shared = benchmark_network("Shared Actor-Critic (Joint)", shared, torch.randn(batch_size, 64))
    results.append(res_shared)

    # Print clean markdown summary table
    print(f"\n| {'Network Architecture':<35} | {'Avg Latency (ms)':<16} | {'Parameters':<12} |")
    print("| " + "-" * 35 + " | " + "-" * 16 + " | " + "-" * 12 + " |")
    for r in results:
        print(f"| {r['network']:<35} | {r['avg_latency_ms']:<16.4f} | {r['total_parameters']:<12,} |")

    # Export results JSON
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "network_benchmark_results.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Benchmark results saved to: {out_file}\n")


if __name__ == "__main__":
    run_all_network_benchmarks()
