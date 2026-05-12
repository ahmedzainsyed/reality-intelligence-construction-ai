# =============================================================================
# Reality Intelligence Platform – Frontend Dockerfile
# Multi-stage: build with Node, serve with NGINX
# =============================================================================

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

ARG VITE_API_URL=http://localhost:8000/api/v1
ARG VITE_WS_URL=ws://localhost:8000/ws
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION

ENV VITE_API_URL=$VITE_API_URL \
    VITE_WS_URL=$VITE_WS_URL \
    VITE_BUILD_DATE=$BUILD_DATE \
    VITE_GIT_COMMIT=$GIT_COMMIT \
    VITE_VERSION=$VERSION \
    NODE_ENV=production

COPY package*.json ./
RUN npm ci --prefer-offline --no-audit

COPY . .
RUN npm run build && \
    echo "Build size:" && du -sh dist/

# ── Stage 2: Development (hot-reload) ─────────────────────────────────────────
FROM node:20-alpine AS development

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]

# ── Stage 3: Production (NGINX static serving) ─────────────────────────────────
FROM nginx:1.25-alpine AS production

LABEL maintainer="engineering@reality-intelligence.io"
LABEL description="RIP Frontend Dashboard"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our nginx config
COPY docker/nginx-frontend.conf /etc/nginx/conf.d/default.conf

# Copy build output
COPY --from=builder /app/dist /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost:80 || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
