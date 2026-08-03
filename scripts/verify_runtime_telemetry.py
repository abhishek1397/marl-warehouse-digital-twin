"""Script executing runtime telemetry verification and synchronization bug auditing."""

import csv
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.simulation_service import SimulationService


def run_verification() -> None:
    print("=========================================================")
    print("       STARTING SIMULATOR RUNTIME TELEMETRY VERIFICATION  ")
    print("=========================================================")

    # 1. Clean previous telemetry CSV if exists
    log_dir = os.path.join(os.getcwd(), "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "runtime_telemetry.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[INFO] Cleared stale telemetry CSV at {csv_path}")

    # 2. Instantiate SimulationService
    service = SimulationService()
    service.create_simulation(grid_width=8, grid_height=8, num_robots=2)
    service.start()

    # 3. Advance simulation for 30 steps
    print("[INFO] Stepping simulation forward for 30 timesteps...")
    service.step(steps=30)

    # 4. Verify CSV creation
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Telemetry CSV file was not created at {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[INFO] Total telemetry CSV rows collected: {len(rows)}")
    assert len(rows) == 60, f"Expected 60 rows (2 robots x 30 steps), got {len(rows)}"

    # Group by step
    steps_data = {}
    for r in rows:
        step_idx = int(r["step"])
        if step_idx not in steps_data:
            steps_data[step_idx] = {}
        steps_data[step_idx][r["robot_id"]] = r

    # 5. Audit Runtime Synchronization Rule (>10 consecutive identical obs and actions)
    consecutive_sync_count = 0
    max_sync_count = 0

    print("\n---------------------------------------------------------")
    print(" Step | Robot 0 Pos | Robot 0 Target | Robot 1 Pos | Robot 1 Target | Robot 0 Action | Robot 1 Action")
    print("---------------------------------------------------------")

    diff_obs_count = 0
    diff_target_count = 0
    diff_action_count = 0
    diff_pkg_count = 0

    for step_idx in sorted(steps_data.keys()):
        r0 = steps_data[step_idx].get("robot_0", {})
        r1 = steps_data[step_idx].get("robot_1", {})

        p0, t0, a0, o0, pkg0 = r0.get("position"), r0.get("target_position"), r0.get("selected_action"), r0.get("observation_summary"), r0.get("assigned_package")
        p1, t1, a1, o1, pkg1 = r1.get("position"), r1.get("target_position"), r1.get("selected_action"), r1.get("observation_summary"), r1.get("assigned_package")

        if o0 != o1:
            diff_obs_count += 1
        if t0 != t1:
            diff_target_count += 1
        if a0 != a1:
            diff_action_count += 1
        if pkg0 != pkg1:
            diff_pkg_count += 1

        print(f" {step_idx:4d} | {p0:11s} | {t0:14s} | {p1:11s} | {t1:14s} | {a0:14s} | {a1:14s}")

        # Check synchronization bug condition
        if o0 == o1 and a0 == a1:
            consecutive_sync_count += 1
            max_sync_count = max(max_sync_count, consecutive_sync_count)
        else:
            consecutive_sync_count = 0

    print("---------------------------------------------------------")
    print(f"[METRICS] Different Observations Steps: {diff_obs_count}/30 ({diff_obs_count/30*100:.1f}%)")
    print(f"[METRICS] Different Targets Steps:      {diff_target_count}/30 ({diff_target_count/30*100:.1f}%)")
    print(f"[METRICS] Different Actions Steps:      {diff_action_count}/30 ({diff_action_count/30*100:.1f}%)")
    print(f"[METRICS] Different Packages Steps:     {diff_pkg_count}/30 ({diff_pkg_count/30*100:.1f}%)")
    print(f"[METRICS] Max Consecutive Synchronized Steps: {max_sync_count}")

    if max_sync_count > 10:
        print("\n[CRITICAL BUG DETECTED] Synchronization bug found!")
        print("Robots executed identical observations and actions for >10 consecutive timesteps.")
        sys.exit(1)
    else:
        print("\n[VERIFICATION PASS] No synchronization bug detected. Multi-agent independence empirically verified!")


if __name__ == "__main__":
    run_verification()
