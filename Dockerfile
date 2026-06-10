#  Python 3.12 as the base image (already contains Python and pip)
FROM python:3.12-slim

# Environment variables:
# PYTHONDONTWRITEBYTECODE=1
#   Prevents Python from creating .pyc files.
#
# PYTHONUNBUFFERED=1
#   Ensures logs appear immediately in Cloud Run
#   instead of being buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container.
# All following commands run from /app.
WORKDIR /app

# Copy requirements.txt first.
COPY requirements.txt .

# Install all Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
# into the container.
COPY . .

# Tell Docker that the application listens on port 8000.
# This is mainly documentation; Cloud Run will use PORT.
EXPOSE 8000

# Start the FastAPI application using Uvicorn.
#
# ai_agent.server:app
#   ai_agent/server.py
#   variable named "app"
#
# --host 0.0.0.0
#   Allows connections from outside the container.
#
# --port 8000
# Use $PORT if it exists
# Otherwise use port 8000
CMD ["sh", "-c", "uvicorn ai_agent.server:app --host 0.0.0.0 --port ${PORT:-8000}"]