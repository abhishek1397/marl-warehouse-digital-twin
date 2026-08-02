# Cloud-Native Multi-Agent Reinforcement Learning Warehouse Digital Twin for Intelligent Robot Coordination

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-red.svg)](https://github.com/astral-sh/ruff)

## Project Overview

This repository hosts the system architecture, research framework, digital twin simulator, and deployment pipeline for **Cloud-Native Multi-Agent Reinforcement Learning Warehouse Digital Twin for Intelligent Robot Coordination**.

Modern automated fulfillment centers demand high-throughput, collision-free, and energy-efficient coordination among hundreds of Autonomous Mobile Robots (AMRs). Traditional centralized pathfinding algorithms (such as Conflict-Based Search or Multi-Agent A*) suffer from exponential computational complexity as fleet sizes scale. This project addresses these challenges by developing a cloud-native digital twin environment coupled with Multi-Agent Reinforcement Learning (MARL) algorithms to achieve real-time, decentralized, collision-avoiding robot coordination.

---

## Motivation

As global e-commerce volumes surge, automated warehouse infrastructure requires dynamic adaptability to high order density, static and dynamic obstacles, battery constraints, and robot hardware failures. 

Key motivations include:
1. **Scalability Bottlenecks:** Centralized path planners fail under heavy real-time traffic due to NP-hard path coordination complexity.
2. **Dynamic Deadlock Resolution:** Static rule-based systems struggle with emergent grid deadlocks, non-deterministic task arrivals, and variable battery drop-offs.
3. **Digital Twin Fidelity:** Bridging the gap between simulation and real-world deployment requires high-fidelity, event-driven state replication accessible via cloud APIs and interactive web dashboards.
4. **Academic & Production Impact:** Providing a publication-quality research testbed suitable for MTech dissertations, top-tier IEEE/Springer conferences, and production cloud infrastructure.

---

## Objectives

* **High-Fidelity 2D Warehouse Digital Twin:** Model discrete/continuous warehouse grids, rack configurations, package pickup/drop stations, charging stations, and kinematics.
* **Hybrid Motion Planning Architecture:** Integrate deterministic grid search algorithms ($A^*$, BFS, DFS) with deep MARL policies (e.g., MAPPO, QMIX) for safe path execution and dynamic collision avoidance.
* **Cloud-Native System Infrastructure:** Expose low-latency API services using FastAPI, decoupled simulation workers, dynamic web dashboards via React + TypeScript, and automated CI/CD containerized deployments.
* **Publication & Open-Source Benchmark:** Conduct rigorous empirical evaluations assessing throughput, average task completion time, energy consumption, and collision frequency against classical baselines.

---

## Key Features

* **Grid & Asset Modeling:** Configurable warehouse topology supporting dynamic shelf layouts, obstacles, pickup/drop zones, and battery charging hubs.
* **Multi-Robot Physics & Kinematics:** Discrete dynamic kinematics modeling velocity, rotation, load status, and battery discharge curves.
* **Decentralized MARL Coordination:** Shared and individual observation spaces enabling local decision-making with global team rewards.
* **Real-time Digital Twin Dashboard:** Web-based 2D visualizer displaying live robot telemetry, path reservations, battery status, and spatial collision heatmaps.
* **Containerized Cloud Deployment:** Ready for deployment on Google Cloud Platform (GCP) Cloud Run with automated GitHub Actions CI/CD workflows.

---

## Technology Stack

* **Core & Simulation:** Python 3.11, NumPy, SciPy
* **Deep Learning & MARL:** PyTorch, PettingZoo, Gymnasium, Ray / RLlib
* **Backend API & Service:** FastAPI, Uvicorn, Pydantic
* **Frontend Web Twin:** React, TypeScript, HTML5 Canvas / SVG
* **Code Quality & Testing:** Pytest, Black, isort, Ruff, MyPy
* **Containerization & Cloud:** Docker, GitHub Actions CI/CD, Google Cloud Platform (GCP Artifact Registry, Cloud Run, Cloud DNS)

---

## High-Level System Architecture

```
+-----------------------------------------------------------------------+
|                           User / Browser                              |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Frontend Web Twin (React + TS)                     |
+-----------------------------------------------------------------------+
                                   |
                         (REST API / WebSockets)
                                   v
+-----------------------------------------------------------------------+
|                       Backend API Service (FastAPI)                   |
+-----------------------------------------------------------------------+
                                   |
                +------------------+------------------+
                |                                     |
                v                                     v
+-------------------------------+   +-----------------------------------+
|    Inference & MARL Engine    |   |    Warehouse Digital Twin         |
|     (PyTorch / Ray RLlib)     |   |    Simulator Engine (Python)      |
+-------------------------------+   +-----------------------------------+
                ^                                     ^
                |                                     |
                +------------------+------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Training Engine & Replay Buffer                    |
+-----------------------------------------------------------------------+
```

---

## Folder Structure

```
.
├── configs/             # Configuration files (JSON/YAML) for simulation & MARL
├── docs/                # Project documentation and master SYSTEM_DESIGN.md
├── tests/               # Unit, integration, and performance tests
├── research/            # Research paper drafts, LaTeX sources, experiment scripts
├── notebooks/           # Jupyter notebooks for exploratory data analysis & plots
├── checkpoints/         # Trained model checkpoints (.pt, .pth)
├── runs/                # TensorBoard logs and experiment metrics
├── backend/             # FastAPI REST services and websocket endpoints
├── frontend/            # React + TypeScript web application dashboard
├── simulator/           # 2D Warehouse digital twin physics & state engine
├── marl/                # Multi-Agent RL environments, policies, and trainers
├── deployment/          # Dockerfiles, CI/CD pipelines, GCP cloud scripts
├── requirements.txt     # Python dependency specifications
├── pyproject.toml       # Tool configuration (Black, Ruff, pytest, etc.)
├── LICENSE              # MIT License
└── README.md            # Repository overview (this file)
```

---

## Development Roadmap

* **Phase 0: Repository & Architecture Initialization** *(Current)*
* **Phase 1: Warehouse Simulator & Grid Modeling**
* **Phase 2: Graph Search & Classical Path Planning ($A^*$, BFS, DFS)**
* **Phase 3: Gymnasium & PettingZoo MARL Environment Wrappers**
* **Phase 4: MARL Algorithm Design & Training Pipeline (PyTorch)**
* **Phase 5: Evaluation Methodology & Empirical Benchmarking**
* **Phase 6: FastAPI Backend Integration & Telemetry API**
* **Phase 7: React + TypeScript Dashboard Development**
* **Phase 8: Docker Containerization & Cloud Deployment (GCP)**
* **Phase 9: Research Paper Compilation & Final Documentation**

---

## Installation & Setup

### Environment Requirements
* **Operating System:** Windows / Linux / macOS
* **Python Environment:** Anaconda / Miniconda (`marl_env`)
* **Python Version:** 3.11

### Step-by-Step Setup

1. **Activate `marl_env` Conda Environment:**
   ```bash
   conda activate marl_env
   ```

2. **Verify Python Interpreter:**
   ```bash
   C:\Users\OMEN\anaconda3\envs\marl_env\python.exe --version
   # Output should indicate Python 3.11.x
   ```

3. **Install Dependencies:**
   ```bash
   C:\Users\OMEN\anaconda3\envs\marl_env\python.exe -m pip install -r requirements.txt
   ```

4. **Verify Code Quality Formatting Tools:**
   ```bash
   ruff check .
   black --check .
   pytest
   ```

---

## Git Workflow

We follow a strict Feature Branch Git Workflow:
1. `main`: Production-ready releases and validated research code.
2. `develop`: Integration branch for active research features.
3. `feature/<feature-name>`: Short-lived branches for modular components (e.g., `feature/astar-planner`, `feature/marl-reward`).

Commit messages adhere to Conventional Commits standard:
* `feat:` New feature implementation
* `fix:` Bug fix
* `docs:` Documentation updates
* `style:` Formatting, missing semi-colons, etc.
* `refactor:` Code refactoring without functionality changes
* `test:` Adding or updating test cases

---

## Future Work

* **3D Physics High-Fidelity Simulation:** Integrating PyBullet or Isaac Sim for low-level mechanical dynamics.
* **Heterogeneous Fleets:** Coordination among mixed robot types (e.g., AGVs, AMRs, robotic arms).
* **Sim-to-Real Transfer:** Hardware-in-the-loop validation using physical ROS2-enabled differential drive micro-robots.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
