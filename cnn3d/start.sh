#!/bin/sh
sudo nvidia-docker run -ti --ipc=host -v $PWD/../:/notebooks/sharedfolder jwde/pytorchdockergpu bash
