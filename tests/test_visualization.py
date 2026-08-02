"""Unit tests for ASCII visualization grid rendering."""

from simulator.position import Position
from simulator.robot import Robot
from simulator.visualization import render_ascii_grid
from simulator.warehouse import Warehouse


def test_ascii_grid_rendering() -> None:
    wh = Warehouse(5, 5)
    robot = Robot("r1", Position(1, 1))

    ascii_view = render_ascii_grid(wh, fleet={"r1": robot})

    assert "+" in ascii_view
    assert "|" in ascii_view
    assert "Legend:" in ascii_view
    assert "1" in ascii_view or "r1" in ascii_view
