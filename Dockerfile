# Multi-stage build for Python backend + Next.js frontend
FROM node:18-slim as frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Main Python application
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/ ./src/
COPY backend/ ./backend/
COPY agent_config.json* ./
COPY conversation_history.json* ./
COPY vector_stores/ ./vector_stores/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package.json ./frontend/package.json

# Install Node.js for serving static files (optional - can serve from Python)
COPY --from=node:18-slim /usr/local/bin/node /usr/local/bin/
COPY --from=node:18-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

# Create directory for SQLite database
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app/src
ENV DATABASE_PATH=/app/data/conversations.db

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]