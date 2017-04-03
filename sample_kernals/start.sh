docker run -p 8889:8888 -v $PWD/../input:/tmp/input -v $PWD:/tmp/sample_kernals -v $PWD/../src:/tmp/src -w=/tmp/working --rm -it kaggle/python 
