# Person Segmentation with flask webserver

## Using Docker

```bash
$ docker build -t pseg_wcam_server:latest .
$ bash run_docker_server.sh -h 8000
```

## Using local system

```bash
$ python -m venv venv
$ source venv/bin.activate
$ pip install -r requirements.txt
$ python3 server.py
```
