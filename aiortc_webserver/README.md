# Person Segmentation with peer-to-peer aiortc

## Using Docker

NOTE: There might be building issues with docker for MacOS with the M-series arm chips. Use local setup as an alternative.

```bash
docker build -t pseg_wcam_aiortc_server:latest .
bash run_docker_server.sh -h 8080
# go to http://localhost:8080/ in your browser
```

## Local setup

```bash
# inside a virtual env
pip install -r requirements.txt
python server.py
# go to localhost:8080
```