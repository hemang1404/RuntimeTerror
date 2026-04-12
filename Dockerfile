FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (for Docker layer caching)
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

# Set PYTHONPATH so server imports work correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose port 8000 (required by Hackathon Validator)
EXPOSE 8000

# Run the FastAPI server on port 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

