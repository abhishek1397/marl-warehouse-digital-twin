"""Unit tests for configuration loading and environment building."""

import os
from main import build_simulation_environment, load_configuration
from simulator.simulation import SimulationState


def test_config_loading_and_environment_building() -> None:
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "warehouse_config.json"
    )
    config = load_configuration(config_path)

    assert "warehouse" in config
    assert "robots" in config
    assert "shelves" in config
    assert "charging_stations" in config
    assert "packages" in config

    engine = build_simulation_environment(config)
    assert engine.warehouse.width == 30
    assert engine.warehouse.height == 20
    assert len(engine.fleet) == 3
    assert engine.task_manager.total_tasks == 3
    assert engine.state == SimulationState.READY
