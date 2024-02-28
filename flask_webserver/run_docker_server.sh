#!/bin/bash

def_cont_name=pseg_wcam_server_cont

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

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    # Stop the container
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    echo "Stopped container '$def_cont_name'"
fi

echo "Running docker with exposed fast api http port: $http"

docker run \
      --rm \
      --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
      --name "$def_cont_name" \
      -p $http:8080 \
      --env HTTP_PORT=$http \
      pseg_wcam_server:latest
