# FastAPI Backend Container Infrastructure Documentation (`backend/Dockerfile`)

Production-ready Docker configuration for packaging and serving the **Warehouse Digital Twin MARL Platform Backend API**.

---

## 1. Container Specification Summary

- **Base Image**: `python:3.11-slim`
- **User Security**: Non-root user `appuser` (UID: `10001`, GID: `10001`)
- **Exposed Port**: `8000`
- **Environment Flags**:
  - `PYTHONDONTWRITEBYTECODE=1` (Prevents `.pyc` creation)
  - `PYTHONUNBUFFERED=1` (Forces stdout/stderr stream flushing)
  - `PYTHONPATH=/app` (Ensures package resolution for `backend`, `simulator`, `marl`)
- **Healthcheck Endpoint**: `GET /api/health` (`http://localhost:8000/api/health`)

---

## 2. Docker Operations Guide

### Build Docker Image
To build the backend Docker image from the repository root:

```bash
docker build -t warehouse-marl-backend:1.0.0 -f backend/Dockerfile .
```

### Run Container
To start the FastAPI backend container on port 8000:

```bash
docker run -d \
  --name warehouse-backend-container \
  -p 8000:8000 \
  --restart unless-stopped \
  warehouse-marl-backend:1.0.0
```

### Inspect Container Logs
To stream container execution logs:

```bash
docker logs -f warehouse-backend-container
```

### Check Container Health Status
To check health check status:

```bash
docker inspect --format='{{json .State.Health}}' warehouse-backend-container
```

### Debug Interactive Shell
To inspect running container filesystem as `appuser`:

```bash
docker exec -it warehouse-backend-container /bin/sh
```

### Stop & Remove Container
To stop and clean up container resources:

```bash
docker stop warehouse-backend-container
docker rm warehouse-backend-container
```

---

## 3. Production Verification

- **Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check Endpoint**: `http://localhost:8000/api/health`
