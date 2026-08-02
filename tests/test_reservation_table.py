"""Unit tests for time-expanded ReservationTable."""

from simulator.position import Position
from simulator.reservation_table import ReservationTable


def test_reservation_table_vertex_reserve_and_query() -> None:
    rt = ReservationTable()
    p1 = Position(2, 2)

    assert rt.is_vertex_reserved(p1, timestep=5) is False
    assert rt.reserve_vertex("r1", p1, timestep=5) is True
    assert rt.is_vertex_reserved(p1, timestep=5) is True
    assert rt.is_vertex_reserved(p1, timestep=5, ignore_robot_id="r1") is False

    # Second robot cannot reserve same vertex at same timestep
    assert rt.reserve_vertex("r2", p1, timestep=5) is False


def test_reservation_table_edge_and_swap_reserve() -> None:
    rt = ReservationTable()
    p1 = Position(1, 1)
    p2 = Position(1, 2)

    assert rt.reserve_edge("r1", p1, p2, timestep=3) is True
    assert rt.is_edge_reserved(p1, p2, timestep=3) is True

    # Reverse swap check: r2 moving p2 -> p1 at timestep 3 should be flagged as reserved
    assert rt.is_edge_reserved(p2, p1, timestep=3) is True
    assert rt.reserve_edge("r2", p2, p1, timestep=3) is False


def test_reservation_table_path_reserve_and_release() -> None:
    rt = ReservationTable()
    path = [Position(0, 0), Position(1, 0), Position(2, 0)]

    assert rt.reserve_path("r1", path, start_timestep=0) is True
    assert rt.is_vertex_reserved(Position(1, 0), timestep=1) is True
    assert rt.is_edge_reserved(Position(0, 0), Position(1, 0), timestep=1) is True

    # Path conflict for r2 at step 1
    path2 = [Position(1, 0), Position(1, 0), Position(2, 0)]
    assert rt.reserve_path("r2", path2, start_timestep=0) is False

    # Release path
    rt.release_path("r1", path, start_timestep=0)
    assert rt.is_vertex_reserved(Position(1, 0), timestep=1) is False
    assert rt.reserve_path("r2", path2, start_timestep=0) is True
