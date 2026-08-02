"""Test suite verifying live simulation flow and policy inference integration."""

import pytest
from backend.app.services.algorithm_service import AlgorithmService
from backend.app.services.simulation_service import SimulationService


def test_live_simulation_service_and_algorithm_stepping() -> None:
    service = SimulationService.get_instance()

    # 1. Create Simulation (2 Robots, 8x8 Grid)
    state_init = service.create_simulation(grid_width=8, grid_height=8, num_robots=2)
    assert state_init.is_initialized is True
    assert state_init.step_count == 0
    assert len(state_init.robots) == 2

    # 2. Select Algorithm (Spatial MAPPO)
    algo_meta = AlgorithmService.set_active_algorithm("Spatial MAPPO")
    assert algo_meta.name == "Spatial MAPPO"
    assert AlgorithmService.get_active_algorithm() == "Spatial MAPPO"

    # 3. Start & Step Simulation
    service.start()
    state_step1 = service.step(steps=1)
    assert state_step1.step_count == 1
    assert state_step1.is_running is True

    # 4. Switch Algorithm (IPPO) & Step
    AlgorithmService.set_active_algorithm("IPPO")
    state_step2 = service.step(steps=2)
    assert state_step2.step_count == 3

    # 5. Pause & Reset
    service.pause()
    assert service.is_paused is True

    state_reset = service.reset()
    assert state_reset.step_count == 0
    assert state_reset.is_running is False
