# Warehouse Digital Twin — AI MARL Robotics Research Platform

[![CI Pipeline](https://github.com/abhishek1397/marl-warehouse-digital-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/abhishek1397/marl-warehouse-digital-twin/actions/workflows/ci.yml)

A production-grade Multi-Agent Reinforcement Learning (MARL) research platform for autonomous multi-robot warehouse fleet coordination, featuring PPO, IPPO, MAPPO (MLP Critic), Spatial MAPPO (2D CNN Centralized Critic), Potential-Based Reward Shaping (PBRS), and Dynamic Action Masking (DAM).

---

## Workspace Structure

```
.
├── backend/                  # FastAPI REST API Backend Skeleton
│   ├── app/
│   │   └── main.py           # FastAPI app entry point (health & info endpoints)
│   └── requirements.txt      # FastAPI, Uvicorn, Pydantic dependencies
│
├── frontend/                 # React 18 + Vite + TypeScript + Tailwind UI
│   ├── src/
│   │   ├── components/       # Reusable UI component library (Navbar, Footer, StatCard, etc.)
│   │   ├── pages/            # 7 Research Platform Pages (Home, Simulation, Algorithms, etc.)
│   │   ├── router/           # React Router v6 setup
│   │   ├── store/            # Zustand state management stores
│   │   └── styles/           # Tailwind CSS & glassmorphism tokens
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── marl/                     # Multi-Agent Reinforcement Learning Algorithms Package
│   ├── algorithms/           # PPO, IPPO, MAPPO, Spatial MAPPO (S-MAPPO)
│   └── parallel_env.py       # PettingZoo Parallel Environment Wrapper
│
├── research/                 # Diagnostic Framework & Scientific Benchmark Scripts
├── simulator/                # Warehouse Digital Twin Simulator & A* Reservation Table
└── docs/                     # Technical Research Reports (MAPPO_DIAGNOSTIC_REPORT.md, SPATIAL_MAPPO.md)
```

---

## Quick Start Guide

### 1. Python Backend Setup (FastAPI)

```powershell
# Navigate to workspace directory
cd "d:\PG\summer training\MARL"

# Activate marl_env Anaconda environment
C:\Users\OMEN\anaconda3\envs\marl_env\python.exe -m pip install -r backend/requirements.txt

# Launch FastAPI development server on http://localhost:8000
C:\Users\OMEN\anaconda3\envs\marl_env\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

- **Root Info Endpoint**: [http://localhost:8000/](http://localhost:8000/)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 2. Frontend Development Server (React 18 + Vite)

```powershell
# Navigate to frontend folder
cd frontend

# Install Node dependencies
npm install

# Launch Vite development server on http://localhost:3000
npm run dev
```

---

## MARL Python Pytest Verification

```powershell
# Run full Pytest suite across all algorithms & simulator
C:\Users\OMEN\anaconda3\envs\marl_env\python.exe -m pytest
```

---

## Continuous Integration (CI) Pipeline

Automated testing and build validation powered by GitHub Actions ([.github/workflows/ci.yml](file:///d:/PG/summer%20training/MARL/.github/workflows/ci.yml)):
- **Python Test Suite**: Setup Python 3.11 with `pip` caching, executes pytest suite across `backend/app`, `marl`, and `simulator`, and uploads XML coverage reports.
- **Frontend Build Verification**: Setup Node.js 20 with `npm` caching, compiles TypeScript and Vite production bundle (`dist/`).
- **Docker Image Build**: Compiles `backend/Dockerfile`, `frontend/Dockerfile`, and `nginx/Dockerfile` images and validates `docker compose config` syntax.

