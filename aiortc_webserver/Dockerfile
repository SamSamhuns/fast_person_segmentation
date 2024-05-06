FROM python:3.10-slim

# requirements for opencv
RUN apt-get  update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set username & workdir
ARG UNAME=user1
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# setup virtual env for python
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN pip install pip==24.0
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . /app

CMD ["python3", "server.py"]
