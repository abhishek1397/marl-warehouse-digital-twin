"""Unit tests for MultiRobotPlanner prioritized multi-robot planning."""

from simulator.grid import Grid
from simulator.planner import MultiRobotPlanner, PlanningRequest
from simulator.position import Position
from simulator.robot import Robot


def test_multi_robot_planner_conflict_avoidance() -> None:
    grid = Grid(10, 10)
    planner = MultiRobotPlanner()

    r1 = Robot("r1", Position(0, 0))
    r2 = Robot("r2", Position(2, 0))

    # Request r1: moves (0,0) -> (2,0), priority 10
    # Request r2: moves (2,0) -> (0,0), priority 5
    req1 = PlanningRequest(robot=r1, goal_position=Position(2, 0), priority=10)
    req2 = PlanningRequest(robot=r2, goal_position=Position(0, 0), priority=5)

    results = planner.plan_joint_paths(grid, [req1, req2])

    assert results["r1"].success is True
    assert results["r2"].success is True

    path1 = results["r1"].path
    path2 = results["r2"].path

    # Verify no vertex or swap collision between path1 and path2
    for t in range(max(len(path1), len(path2))):
        pos1 = path1[min(t, len(path1) - 1)]
        pos2 = path2[min(t, len(path2) - 1)]
        assert pos1 != pos2, f"Collision detected at step {t} at position {pos1}"
