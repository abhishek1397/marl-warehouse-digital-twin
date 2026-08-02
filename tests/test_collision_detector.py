"""Unit tests for CollisionDetector validation diagnostics."""

from simulator.cell import CellType
from simulator.collision_detector import CollisionDetector, CollisionType
from simulator.position import Position
from simulator.warehouse import Warehouse


def test_collision_detector_clean_paths() -> None:
    wh = Warehouse(10, 10)
    cd = CollisionDetector()

    paths = {
        "r1": [Position(0, 0), Position(1, 0), Position(2, 0)],
        "r2": [Position(0, 5), Position(1, 5), Position(2, 5)],
    }

    diagnostics = cd.detect_collisions(wh, paths)
    assert len(diagnostics) == 0
    assert cd.has_collisions(wh, paths) is False


def test_collision_detector_vertex_collision() -> None:
    wh = Warehouse(10, 10)
    cd = CollisionDetector()

    paths = {
        "r1": [Position(0, 0), Position(1, 1), Position(2, 2)],
        "r2": [Position(2, 0), Position(1, 1), Position(0, 2)],
    }

    diagnostics = cd.detect_collisions(wh, paths)
    assert len(diagnostics) == 1
    assert diagnostics[0].collision_type == CollisionType.VERTEX_COLLISION
    assert diagnostics[0].timestep == 1
    assert diagnostics[0].position == Position(1, 1)
    assert set(diagnostics[0].robot_ids) == {"r1", "r2"}


def test_collision_detector_edge_swap_collision() -> None:
    wh = Warehouse(10, 10)
    cd = CollisionDetector()

    paths = {
        "r1": [Position(1, 0), Position(2, 0)],
        "r2": [Position(2, 0), Position(1, 0)],
    }

    diagnostics = cd.detect_collisions(wh, paths)
    assert len(diagnostics) == 1
    assert diagnostics[0].collision_type == CollisionType.EDGE_SWAP_COLLISION
    assert diagnostics[0].timestep == 1
    assert set(diagnostics[0].robot_ids) == {"r1", "r2"}


def test_collision_detector_static_obstacle() -> None:
    wh = Warehouse(10, 10)
    wh.set_cell_type(Position(1, 0), CellType.OBSTACLE)
    cd = CollisionDetector()

    paths = {
        "r1": [Position(0, 0), Position(1, 0), Position(2, 0)],
    }

    diagnostics = cd.detect_collisions(wh, paths)
    assert len(diagnostics) == 1
    assert diagnostics[0].collision_type == CollisionType.STATIC_OBSTACLE_COLLISION
    assert diagnostics[0].position == Position(1, 0)
