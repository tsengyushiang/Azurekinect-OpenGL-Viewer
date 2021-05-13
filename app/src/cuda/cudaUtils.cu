

#include "cudaUtils.cuh"

int test() {
	int* a;
	cudaMalloc(&a,100);
	cudaFree(a);
	return 0;
}