# Docker Compose Container Orchestration Guide (`docker-compose.yml`)

Production and development container orchestration setup for the **Warehouse Digital Twin MARL Platform**.

---

## 1. Multi-Container System Overview

```
                        ┌───────────────────────────────┐
                        │   React 18 + Vite Frontend    │
                        │     (warehouse-frontend)      │
                        │       Port 3000 -> 80         │
                        └───────────────┬───────────────┘
                                        │
                                        │ REST API Calls (/api)
                                        ▼
                        ┌───────────────────────────────┐
                        │    FastAPI Backend API        │
                        │     (warehouse-backend)       │
                        │          Port 8000            │
                        └───────────────┬───────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
               ┌──────────────────┐          ┌──────────────────┐
               │ Digital Twin Sim │          │ MARL Algorithms  │
               │   (simulator/)   │          │     (marl/)      │
               └──────────────────┘          └──────────────────┘
```

---

## 2. Service Specifications

| Service | Container Name | Host Port | Internal Port | Health Check | Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backend** | `warehouse-backend` | `8000` | `8000` | `GET /api/health` | None |
| **Frontend** | `warehouse-frontend` | `3000` | `80` | Nginx Ping | `backend` (service_healthy) |

---

## 3. Docker Compose Command Operations

### 1. Build and Start Full Application
To build all container images and start services in detached mode:

```bash
docker compose up -d --build
```

### 2. View Service Status & Health
To verify running container status and backend health:

```bash
docker compose ps
```

### 3. Stream Application Logs
To view unified log streams across both frontend and backend containers:

```bash
docker compose logs -f
```

To view logs for a specific service:

```bash
# Backend logs
docker compose logs -f backend

# Frontend logs
docker compose logs -f frontend
```

### 4. Rebuild Specific Service
To rebuild and restart only one container service (e.g., frontend):

```bash
docker compose up -d --build frontend
```

### 5. Shutdown & Clean Up
To stop and remove containers, networks, and volumes:

```bash
docker compose down -v
```

---

## 4. Internal Networking & Health Checks

- **Bridge Network**: `warehouse-marl-network` (`warehouse-network`) facilitates isolated container-to-container communication.
- **Health Dependency**: The `frontend` container waits for the `backend` container to pass its `healthcheck` (`GET /api/health`) before starting.
- **Log Persistence**: Named volume `warehouse-backend-logs` persists backend API execution logs across container restarts.

---

## 5. Endpoints Quick Access

- **React Web Application**: [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
