FROM python:3.10-slim

# set working directory
WORKDIR /app

# requirements for opencv
RUN apt-get  update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# install requirements
RUN pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "server_socketio.py"]
