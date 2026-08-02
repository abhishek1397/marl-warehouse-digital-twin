"""Unit tests for Shelf, ChargingStation, Obstacle, and Package entities."""

import pytest
from simulator.charging_station import ChargingStation
from simulator.exceptions import InvalidPlacementError, WarehouseError
from simulator.obstacle import Obstacle
from simulator.package import Package, PackageStatus
from simulator.position import Position
from simulator.shelf import Shelf


def test_shelf_lifecycle() -> None:
    shelf = Shelf("s1", Position(1, 1), capacity=2)
    assert shelf.shelf_id == "s1"
    assert shelf.position == Position(1, 1)
    assert shelf.capacity == 2
    assert shelf.is_empty() is True

    pkg1 = Package("p1", Position(0, 0), Position(5, 5))
    pkg2 = Package("p2", Position(0, 0), Position(5, 5))
    pkg3 = Package("p3", Position(0, 0), Position(5, 5))

    shelf.add_package(pkg1)
    assert shelf.current_load == 1
    assert pkg1.current_shelf_id == "s1"
    assert shelf.get_package("p1") == pkg1

    shelf.add_package(pkg2)
    assert shelf.is_full() is True

    with pytest.raises(WarehouseError):
        shelf.add_package(pkg3)

    removed_pkg = shelf.remove_package("p1")
    assert removed_pkg == pkg1
    assert pkg1.current_shelf_id is None
    assert shelf.is_full() is False

    with pytest.raises(InvalidPlacementError):
        Shelf("s2", Position(0, 0), capacity=0)


def test_charging_station_lifecycle() -> None:
    cs = ChargingStation("c1", Position(3, 3), charge_rate=20.0, capacity=1)
    assert cs.station_id == "c1"
    assert cs.position == Position(3, 3)
    assert cs.charge_rate == 20.0
    assert cs.is_available() is True

    cs.dock_robot("r1")
    assert cs.is_docked("r1") is True
    assert cs.is_available() is False

    with pytest.raises(WarehouseError):
        cs.dock_robot("r2")

    cs.undock_robot("r1")
    assert cs.is_docked("r1") is False
    assert cs.is_available() is True

    with pytest.raises(WarehouseError):
        ChargingStation("c2", Position(0, 0), charge_rate=-5.0)


def test_obstacle_properties() -> None:
    obs = Obstacle("o1", Position(2, 2), name="Pillar", description="Concrete pillar")
    assert obs.obstacle_id == "o1"
    assert obs.position == Position(2, 2)
    assert obs.name == "Pillar"
    assert obs.description == "Concrete pillar"
    assert "Pillar" in str(obs) or "Obstacle" in str(obs)
