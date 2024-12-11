FROM python:3.10-slim

# Set non-interactive mode for APT to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update system packages and install dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for better security
ARG UNAME=user1
RUN useradd --create-home --shell /bin/bash $UNAME

# Set the working directory and user
WORKDIR /app
RUN chown -R $UNAME:$UNAME /app
USER $UNAME

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and activate a virtual environment
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY --chown=$UNAME:$UNAME requirements.txt /app/requirements.txt
RUN pip install --upgrade pip==24.0 && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the application code
COPY --chown=$UNAME:$UNAME . /app

# Default command to run the application
CMD ["python3", "server.py"]
