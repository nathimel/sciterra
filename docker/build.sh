#!/bin/bash

# Build the image
docker build --progress plain -f ./docker/Dockerfile \
    -t sciterra:latest "$@" .