# Nginx Production Reverse Proxy Infrastructure Documentation (`nginx/`)

Production Nginx reverse proxy service (`warehouse-proxy`) acting as the single public gateway for the **Warehouse Digital Twin MARL Platform**.

---

## 1. Network Topology & Routing Architecture

```
                       ┌───────────────────────────────┐
                       │     Client Browser (Port 80)   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Nginx Reverse Proxy Gateway  │
                       │       (warehouse-proxy)       │
                       │           Port 80             │
                       └───────┬───────────────┬───────┘
                               │               │
            / (Frontend UI)    │               │ /api/* (Backend REST API)
                               ▼               ▼
                ┌──────────────────┐       ┌──────────────────┐
                │ React Frontend   │       │ FastAPI Backend  │
                │ (warehouse-front)│       │ (warehouse-back) │
                │     Port 80      │       │    Port 8000     │
                └──────────────────┘       └──────────────────┘
```

---

## 2. Reverse Proxy Routing Table

| Request Path | Destination Upstream | Description |
| :--- | :--- | :--- |
| `/` | `http://warehouse-frontend:80` | React Single Page Application distribution bundle |
| `/api/*` | `http://warehouse-backend:8000/api/*` | FastAPI REST API endpoints |
| `/docs` | `http://warehouse-backend:8000/docs` | OpenAPI Swagger Documentation |
| `/openapi.json` | `http://warehouse-backend:8000/openapi.json` | OpenAPI JSON Schema Specification |

---

## 3. Forwarded Headers & Security Features

### Reverse Proxy Headers
- `Host`: `$host`
- `X-Real-IP`: `$remote_addr`
- `X-Forwarded-For`: `$proxy_add_x_forwarded_for`
- `X-Forwarded-Proto`: `$scheme`
- `Upgrade` & `Connection`: Support WebSocket protocol upgrades.

### Security Headers Enforced
- `X-Frame-Options`: `SAMEORIGIN` (Mitigates clickjacking attacks)
- `X-Content-Type-Options`: `nosniff` (Prevents MIME sniffing)
- `X-XSS-Protection`: `1; mode=block` (Enforces browser XSS filter)
- `Referrer-Policy`: `strict-origin-when-cross-origin` (Protects referrer telemetry)

### Performance & Compression
- **Gzip Compression**: Activated for `text/plain`, `text/css`, `application/json`, `application/javascript`, `image/svg+xml`.
- **Proxy Timeouts**: Configured to 60s for long-running simulation steps.

---

## 4. Container Deployment Quick Start

```bash
# Build and start all services (Proxy on port 80)
docker compose up -d --build

# Verify container status
docker compose ps

# Stream Nginx reverse proxy logs
docker compose logs -f proxy
```

Access Points:
- **Application Portal**: `http://localhost`
- **Backend API**: `http://localhost/api/health`
- **Swagger Docs**: `http://localhost/docs`
