# ==============================================================================
# Production Nginx Reverse Proxy Dockerfile
# Base Image: Nginx Alpine
# ==============================================================================

FROM nginx:alpine

# Remove default configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx global configuration and virtual host reverse proxy definitions
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port 80
EXPOSE 80

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
