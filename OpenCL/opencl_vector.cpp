#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define __CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

const char* kernelSource = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
)";

int main() {
    constexpr int N = 1024 * 2000000;
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem d_A, d_B, d_C;
    cl_int err;

    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, A.data(), &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, B.data(), &err);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "vector_add", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);

    // Calculation
    auto start = std::chrono::high_resolution_clock::now();

    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * N, C.data(), 0, nullptr, nullptr);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vector addition completed in " << elapsed.count() << " seconds." << std::endl;

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
