#!/bin/bash

sudo docker build -t raftflow:0.1 .
docker stop raftFlowContainer
docker remove raftFlowContainer
docker-compose up -d
code .

# fix line endings
# sed -i -e 's/\r$//' run.sh