# Frontend Container Infrastructure Documentation (`frontend/Dockerfile`)

Production multi-stage Docker configuration for building and serving the **React 18 + Vite + TypeScript Frontend UI** using Nginx Alpine.

---

## 1. Multi-Stage Build Architecture

- **Stage 1 (Builder)**: `node:20-alpine` compiling static JavaScript/CSS dist bundle via `npm run build`.
- **Stage 2 (Runner)**: `nginx:alpine` serving static SPA bundle from `/usr/share/nginx/html`.
- **SPA Routing**: `nginx.conf` fallback (`try_files $uri $uri/ /index.html;`) resolving React Router routes (`/simulation`, `/research`).
- **Exposed Port**: `80` (Mapped to `3000` on host).

---

## 2. Docker Operations Guide

### Build Frontend Docker Image
Build the frontend image from the repository root:

```bash
docker build -t warehouse-marl-frontend:1.0.0 -f frontend/Dockerfile .
```

### Run Frontend Container
Run the container mapping host port `3000` to container port `80`:

```bash
docker run -d \
  -p 3000:80 \
  --name warehouse-frontend \
  warehouse-marl-frontend:1.0.0
```

The application will be accessible at **[http://localhost:3000](http://localhost:3000)**.

### Stream Container Logs
Inspect Nginx access and error logs:

```bash
docker logs -f warehouse-frontend
```

### Stop & Cleanup Container
```bash
docker stop warehouse-frontend
docker rm warehouse-frontend
```

---

## 3. Environment Variable Configuration

To specify a custom backend API URL during build time:

```bash
docker build \
  --build-arg VITE_API_BASE_URL=http://your-backend-domain:8000/api \
  -t warehouse-marl-frontend:1.0.0 \
  -f frontend/Dockerfile .
```
