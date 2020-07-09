gcc clapack-ep.c -lblas -llapack -lcblas -lm -o clapack-cla.out && ./clapack-cla.out && rm clapack-cla.out
nvcc cuda-ep.cu -lcublas -lcusolver && ./a.out && rm a.out
