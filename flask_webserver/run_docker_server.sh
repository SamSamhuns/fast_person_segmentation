#!/bin/bash

# check for 2 cmd args
if [ $# -ne 2 ]
  then
    echo "HTTP port must be specified for flask web server."
		echo "eg. \$ bash run_docker_server.sh -h 8080"
		exit
fi

# get the http port for flask server
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--http) http="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";
	exit 1 ;;
    esac
    shift
done

echo "Running docker with exposed fast api http port: $http"

docker run \
      --rm \
      --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
      --name pseg_wcam_server_cont \
      -p $http:8080 \
      pseg_wcam_server:latest
