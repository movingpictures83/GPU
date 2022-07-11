nvcc template.cu -rdc=true -o test -std=c++11 -gencode arch=compute_70,code=sm_70 -I/home/vsm2/gputeachingkit-labs/libgputk -I/usr/local/cuda/include -lcudadevrt --default-stream per-thread

