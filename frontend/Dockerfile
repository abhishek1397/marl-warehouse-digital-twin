# ==============================================================================
# Multi-Stage Production Dockerfile for React + TypeScript + Vite Frontend
# Stage 1: Builder (Node 20 Alpine)
# Stage 2: Production Runner (Nginx Alpine)
# ==============================================================================

# STAGE 1: BUILD ENVIRONMENT
FROM node:20-alpine AS builder

WORKDIR /app/frontend

# Copy dependency manifests
COPY frontend/package*.json ./

# Install npm dependencies cleanly
RUN npm install

# Copy frontend source code
COPY frontend/ ./

# Pass environment build arguments
ARG VITE_API_BASE_URL=http://localhost:8000/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Compile production Vite distribution build
RUN npm run build

# STAGE 2: PRODUCTION RUNNER ENVIRONMENT
FROM nginx:alpine AS runner

# Copy custom Nginx configuration for SPA routing and asset caching
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from builder stage
COPY --from=builder /app/frontend/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
