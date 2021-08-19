FROM python:3.7-slim

# set working directory
WORKDIR /app

# requirements for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# install requirements
RUN pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

CMD ["python3", "server_socketio.py"]
