#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <chrono>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define __CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

const float k_e = 8.987551786214e9; // Value from Wikipedia: https://en.wikipedia.org/wiki/Coulomb%27s_law

int parse_args(int argc, char **argv, unsigned int &particle_count, unsigned int &grid_size) {
	if (argc == 1) return 0;
	std::string usage("Usage (positive particle_count, positive grid_size): <particle_count>, <grid_size>");

	if (argc > 3) {
		std::cout << usage << std::endl;
		exit(-1);
	}
    if (std::stoi(argv[1]) < 1 || std::stoi(argv[2]) < 1) {
        std::cout << usage << std::endl;
        exit(-1);
    }

	particle_count = std::stoi(argv[1]);
    grid_size = std::stoi(argv[2]);
    return 1;
}

std::string readKernelFile(const std::string &fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + fileName);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void generate_random_particles(std::vector<float> &particles, unsigned int grid_size, unsigned int particle_count) {
    std::random_device rd;
    std::mt19937 gen(11); // Fixed seed for predictable results
    std::uniform_int_distribution<> pos_dist(0, grid_size);
    std::uniform_real_distribution<float> charge_dist(-5e-6, 5e-6);
    
    for (int i = 0; i < particle_count; ++i) {
        particles.push_back(pos_dist(gen));
        particles.push_back(pos_dist(gen));
        particles.push_back(pos_dist(gen));
        particles.push_back(charge_dist(gen));
    }
}

int main(int argc, char* argv[]) {
	try {
        std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found.");
        }

        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        std::string kernelSource = readKernelFile("kernel.cl");

        // Create program and kernel
        cl::Program::Sources sources = {{kernelSource.c_str(), kernelSource.length()}};
        cl::Program program(context, sources);
        std::string build_options = "-cl-std=CL3.0";
        cl_int error = program.build(build_options.c_str());
        if (error != CL_SUCCESS) {
            std::string buildLog;
            program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &buildLog);
            std::cerr << "Build failed: " << buildLog << std::endl;
            return 2;
        }
        cl::Kernel kernel(program, "dcs");

        // Parsing args
        unsigned int particle_count = 1; //* Discuss random init
        unsigned int grid_size = 1; //* Discussion: grid is a box
        parse_args(argc, argv, particle_count, grid_size);
        printf("Particle count: %d, Grid size: %d\n", particle_count, grid_size);
        
        // Value init
        std::vector<int> grid(3, grid_size);

        std::vector<float> energygrid;
        energygrid.reserve(grid_size * grid_size * grid_size);

        std::vector<float> particles;
        particles.reserve(particle_count * 4);
        generate_random_particles(particles, grid_size, particle_count);

        std::vector<float> dcs_result(grid_size * grid_size * grid_size);

        // Buffers and kernel arguments
        cl::Buffer energygridBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * grid_size * grid_size);
        cl::Buffer particlesBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * particle_count * 4, particles.data());

        kernel.setArg(0, energygridBuffer);
        kernel.setArg(1, particlesBuffer);
        kernel.setArg(3, particle_count);
        kernel.setArg(4, k_e); // potencijalno u host izracunati

        // Calculation loop
        auto start = std::chrono::high_resolution_clock::now();

        for (int z = 0; z < grid_size; ++z) { //? jel ovo int
            kernel.setArg(2, z);
            
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size, grid_size), cl::NullRange);
            queue.enqueueReadBuffer(energygridBuffer, CL_TRUE, 0, sizeof(float) * grid_size * grid_size, dcs_result.data() + (z * grid_size * grid_size));
            queue.finish();
        }
        
        // Finalization
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Simulation completed in " << elapsed.count() << " seconds." << std::endl;

        float sum = 0.0f;
        for (int x = 0; x < dcs_result.size(); ++x) {
            sum += dcs_result[x];
        }
        std::cout << "Total sum: " << sum << std::endl;

	} catch (const std::exception &ex) {
		std::cerr << "Error: " << ex.what() << std::endl;
		return 1;
	} catch (...) {
        std::cerr << "An unknown error occured." << std::endl;
        return -1;
    }
	return 0;
}
