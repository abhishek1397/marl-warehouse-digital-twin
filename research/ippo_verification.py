"""IPPO Verification Pipeline executing all diagnostic verifier modules and outputting docs/IPPO_VERIFICATION_REPORT.md."""

import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.benchmark_report import DiagnosticReportCompiler
from research.buffer_verifier import BufferVerifier
from research.environment_sync_verifier import EnvironmentSyncVerifier
from research.gae_verifier import GAEVerifier
from research.observation_verifier import ObservationVerifier
from research.pettingzoo_api_verifier import PettingZooAPIVerifier
from research.policy_sync_verifier import PolicySyncVerifier
from research.reward_verifier import RewardVerifier
from research.rollout_verifier import RolloutVerifier
from research.trainer_verifier import SingleAgentEquivalenceVerifier


def run_full_ippo_verification() -> None:
    """Orchestrates 10 diagnostic verifiers and outputs comprehensive report to docs/IPPO_VERIFICATION_REPORT.md."""
    print("=" * 75)
    print("      IPPO DIAGNOSTIC VERIFICATION & ROOT-CAUSE ANALYSIS PIPELINE")
    print("=" * 75)

    results = {}

    # Part 1: PettingZoo API Compliance
    print("\n[+] Part 1: Verifying PettingZoo Parallel API Compliance...")
    results["pettingzoo_api"] = PettingZooAPIVerifier.verify_api_compliance()
    print(f"    Status: {results['pettingzoo_api']['status']}")

    # Part 2: Single Agent Equivalence
    print("\n[+] Part 2: Verifying Single-Agent PPO vs 1-Robot IPPO Equivalence...")
    results["single_agent_equivalence"] = SingleAgentEquivalenceVerifier.verify_single_agent_equivalence(timesteps=600)
    print(f"    Status: {results['single_agent_equivalence']['status']} (PPO Gym: {results['single_agent_equivalence']['ppo_gym_eval_reward']:.2f}, IPPO 1-Robot: {results['single_agent_equivalence']['ippo_1robot_eval_reward']:.2f})")

    # Part 3: Observation Verification
    print("\n[+] Part 3: Verifying Observation Shapes and Values...")
    results["observation"] = ObservationVerifier.verify_observations()
    print(f"    Status: {results['observation']['status']}")

    # Part 4: Reward Verification
    print("\n[+] Part 4: Verifying Reward Assignment and Leakage...")
    results["reward"] = RewardVerifier.verify_reward_assignment()
    print(f"    Status: {results['reward']['status']}")

    # Part 5: Rollout Verification
    print("\n[+] Part 5: Verifying Transition Rollout Storage...")
    results["rollout"] = RolloutVerifier.verify_rollout_collection()
    print(f"    Status: {results['rollout']['status']}")

    # Part 6: GAE Verification
    print("\n[+] Part 6: Verifying GAE Advantage Computation...")
    results["gae"] = GAEVerifier.verify_gae_calculation()
    print(f"    Status: {results['gae']['status']}")

    # Part 7: Buffer Verification
    print("\n[+] Part 7: Verifying Buffer Isolation...")
    results["buffer"] = BufferVerifier.verify_buffer_isolation()
    print(f"    Status: {results['buffer']['status']}")

    # Part 8: Policy Sync Verification
    print("\n[+] Part 8: Verifying Policy Parameter Updates & Gradient Flow...")
    results["policy_sync"] = PolicySyncVerifier.verify_policy_updates()
    print(f"    Status: {results['policy_sync']['status']}")

    # Part 9: Environment Sync Verification
    print("\n[+] Part 9: Verifying Trajectory Step-by-Step Sync...")
    results["environment_sync"] = EnvironmentSyncVerifier.compare_trajectories(num_steps=30)
    print(f"    Status: {results['environment_sync']['status']}")

    # Output Diagnostic Report
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs/IPPO_VERIFICATION_REPORT.md"))
    DiagnosticReportCompiler.compile_report(results, doc_path)

    print("\n" + "=" * 75)
    print(f"[+] Full Diagnostic Verification Complete!")
    print(f"[+] Report compiled successfully to: {doc_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_full_ippo_verification()
