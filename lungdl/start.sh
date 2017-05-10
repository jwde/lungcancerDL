#!/bin/sh
sudo nvidia-docker run -ti --ipc=host -v /a/data/lungdl/:/a/data/lungdl  -v $PWD/../:/notebooks/sharedfolder jwde/pytorchdockergpu bash
