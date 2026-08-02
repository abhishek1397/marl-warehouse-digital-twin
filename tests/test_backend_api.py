"""Test suite verifying FastAPI backend REST API endpoints and exception handlers."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_system_endpoints() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "running"

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_version = client.get("/api/version")
    assert res_version.status_code == 200
    assert "version" in res_version.json()


def test_simulation_flow_endpoints() -> None:
    # 1. Create Simulation
    create_payload = {
        "grid_width": 8,
        "grid_height": 8,
        "num_robots": 2,
        "enable_pbrs": True,
        "enable_dam": True,
    }
    res_create = client.post("/api/simulation/create", json=create_payload)
    assert res_create.status_code == 200
    data = res_create.json()
    assert data["is_initialized"] is True
    assert data["grid_size"] == [8, 8]
    assert len(data["robots"]) == 2

    # 2. Start Simulation
    res_start = client.post("/api/simulation/start")
    assert res_start.status_code == 200
    assert res_start.json()["is_running"] is True

    # 3. Step Simulation
    res_step = client.post("/api/simulation/step", json={"steps": 2})
    assert res_step.status_code == 200
    assert res_step.json()["step_count"] == 2

    # 4. Get Simulation State
    res_state = client.get("/api/simulation/state")
    assert res_state.status_code == 200
    assert res_state.json()["step_count"] == 2

    # 5. Pause Simulation
    res_pause = client.post("/api/simulation/pause")
    assert res_pause.status_code == 200
    assert res_pause.json()["is_paused"] is True

    # 6. Reset Simulation
    res_reset = client.post("/api/simulation/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["step_count"] == 0


def test_algorithm_endpoints() -> None:
    # List algorithms
    res_list = client.get("/api/algorithms")
    assert res_list.status_code == 200
    algos = res_list.json()["algorithms"]
    assert len(algos) == 7

    # Get algorithm detail
    res_detail = client.get("/api/algorithms/Spatial MAPPO")
    assert res_detail.status_code == 200
    assert res_detail.json()["name"] == "Spatial MAPPO"

    # Select algorithm
    res_select = client.post("/api/algorithms/select", json={"algorithm_name": "IPPO"})
    assert res_select.status_code == 200
    assert res_select.json()["algorithm"]["name"] == "IPPO"

    # Test 404 Unknown Algorithm
    res_err = client.get("/api/algorithms/NonExistentAlgo")
    assert res_err.status_code == 404
    assert "error" in res_err.json()


def test_experiment_endpoints() -> None:
    # List experiments
    res_list = client.get("/api/experiments")
    assert res_list.status_code == 200
    exps = res_list.json()["experiments"]
    assert len(exps) >= 3

    # Get experiment detail
    res_detail = client.get("/api/experiments/exp_001")
    assert res_detail.status_code == 200
    assert res_detail.json()["experiment"]["id"] == "exp_001"

    # Test 404 Unknown Experiment
    res_err = client.get("/api/experiments/exp_999")
    assert res_err.status_code == 404
    assert "error" in res_err.json()
