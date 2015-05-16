import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.tools, pycuda.autoinit, pycuda.compiler
import numpy, math

start = cuda.Event()
end = cuda.Event()
d = [2.0, 3.0, 3.0, 3.0, 13.0, 5.0, 20.0, 5.0, 19.0, 12.0, 10.0, 17.0, 6.0, 2.0, 3.0, 2.0, 39.0, 14.0, 2.0, 31.0, 113.0, 33.0, 74.0, 61.0, 23.0, 21.0, 12.0, 4.0, 300.0, 8.0, 2.0, 89.0, 25.0, 2.0, 39.0, 7.0, 9.0, 2.0, 52.0, 39.0, 3.0, 4.0, 78.0, 2.0, 3.0, 5.0, 10.0, 10.0, 9.0, 4.0, 4.0, 19.0, 3.0, 2.0, 84.0, 43.0, 44.0, 2.0, 2.0, 3.0, 5.0, 25.0, 2.0, 16.0, 34.0, 4.0, 13.0, 68.0, 3.0, 2.0, 2.0, 25.0, 77.0, 5.0, 15.0, 5.0, 4.0, 4.0, 8.0, 10.0, 7.0, 8.0, 46.0, 27.0, 2.0, 8.0, 15.0, 10.0, 2.0, 61.0, 16.0, 99.0, 2.0, 13.0, 27.0, 36.0, 2.0, 5.0, 5.0, 8.0]
r = 500
a_gpu = gpuarray.to_gpu(numpy.array(d))

# Run on CPU
start.record()
r_cpu = 2 * math.pi * r / sum(d)
c_cpu = numpy.array([r_cpu * x for x in d])
end.record()
end.synchronize()

print "-" * 80
print "Coefficient of CPU operation:",
print r_cpu
print "CPU time: %fs" %(start.time_till(end) * 1e-3)
print c_cpu

# Run on GPU
start.record()
r_gpu = 2 * math.pi * r / gpuarray.sum(a_gpu).get()
c_gpu = (r_gpu * a_gpu).get()
end.record()
end.synchronize()

print "-" * 80
print "Coefficient of GPU operation:",
print r_gpu
print "GPU time: %fs" %(start.time_till(end) * 1e-3)
print c_gpu

print "-" * 80
print "CPU-GPU difference:"
print c_cpu - c_gpu

numpy.allclose(c_cpu, c_gpu)
