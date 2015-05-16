import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import compiler, gpuarray, tools

start = cuda.Event()
end = cuda.Event()
MATRIX_SIZE = 2
seq = 30
kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N,
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""
# Run on CPU
a_cpu = np.matrix('1 1; 1 0').astype(np.float32)
b_cpu = np.matrix('1 0; 1 0').astype(np.float32)
c_cpu = a_cpu

start.record()
for x in range(0, 29):
    c_cpu = c_cpu * a_cpu
c_cpu = c_cpu * b_cpu
end.record()
end.synchronize()

print "-" * 80
print "CPU time: %fs" %(start.time_till(end) * 1e-3)
print "Fibonacci(30) by CPU:",
print c_cpu.item(2)


# Run on GPU
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = gpuarray.to_gpu(a_cpu)

kernel_code = kernel_code_template % { 'MATRIX_SIZE': MATRIX_SIZE }
mod = compiler.SourceModule(kernel_code)
matrixmul = mod.get_function("MatrixMulKernel")

start.record()
for x in range(0, 29):
    matrixmul(c_gpu, a_gpu, c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1),)
matrixmul(c_gpu, b_gpu, c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1),)
end.record()
end.synchronize()

print "-" * 80
print "GPU time: %fs" %(start.time_till(end) * 1e-3)
print "Fibonacci(30) by GPU:",
print c_gpu.get().item(2)

print "-" * 80
print "CPU-GPU difference:"
print c_cpu.item(2) - c_gpu.get().item(2)

np.allclose(c_cpu, c_gpu.get())
