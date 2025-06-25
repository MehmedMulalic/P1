#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <cmath>
#include <sycl/sycl.hpp>

// Value from Wikipedia: https://en.wikipedia.org/wiki/Coulomb%27s_law [Access date: 28.05.2025]
const float k_e = 8.987551786214e9;

std::vector<float> dcs_usm(sycl::queue &q, const std::vector<float> &particles, sycl::range<2> global_size, int z) {
    const size_t size = global_size[0] * global_size[1];
    std::vector<float> result(size);

    float* device_particles = sycl::malloc_device<float>(particles.size(), q);
    float* device_result = sycl::malloc_device<float>(result.size(), q);

    q.memcpy(device_particles, particles.data(), sizeof(float) * particles.size()).wait();

    size_t particle_size = particles.size();
    q.parallel_for(global_size, [=](sycl::id<2> idx) {
        const size_t index = idx[0]*global_size[1] + idx[1];
        float energy = 0.0f;
        
        for (size_t p = 0; p < particle_size; p+=4) {
            float dx = static_cast<float>(idx[0]) - device_particles[p];
            float dy = static_cast<float>(idx[1]) - device_particles[p+1];
            float dz = static_cast<float>(z) - device_particles[p+2];

            float r = sycl::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r > 0.0f) energy += device_particles[p+3] / r;
        }

        device_result[index] = energy * k_e;
    });
    
    q.memcpy(result.data(), device_result, sizeof(float) * size).wait();

    sycl::free(device_particles, q);
    sycl::free(device_result, q);

    return result;
}

std::vector<float> dcs_buffer(sycl::queue &q, const std::vector<float> &particles, sycl::range<2> global_size, int z) {
    const size_t size = global_size[0] * global_size[1];
    std::vector<float> result(size);

    sycl::buffer<float, 1> buffer_particles(particles.data(), sycl::range<1>(particles.size()));
    sycl::buffer<float, 1> buffer_result(result.data(), sycl::range<1>(size));

    q.submit([&](sycl::handler &h) {
        auto acc_particles = buffer_particles.get_access<sycl::access::mode::read>(h);
        auto acc_result = buffer_result.get_access<sycl::access::mode::write>(h);

        h.parallel_for(global_size, [=](sycl::id<2> idx) {
            const size_t index = idx[0]*global_size[1] + idx[1];
            float energy = 0.0f;
            
            for (size_t p = 0; p < acc_particles.get_range()[0]; p+=4) {
                float dx = static_cast<float>(idx[0]) - acc_particles[p];
                float dy = static_cast<float>(idx[1]) - acc_particles[p+1];
                float dz = static_cast<float>(z) - acc_particles[p+2];

                float r = sycl::sqrt(dx*dx + dy*dy + dz*dz);

                if (r > 0.0f) energy += acc_particles[p+3] / r;
            }

            acc_result[index] = energy * k_e;
        });
    }).wait();

    return result;
}

void parse_args(int argc, char **argv, size_t &particle_count, size_t &grid_size, std::string &method) {
	if (argc == 1) return;
	std::string usage("Usage (positive values): <particle_count>, <grid_size>, <usm|buffer>");

	if (argc > 4) {
		std::cout << usage << std::endl;
		exit(-1);
	}
    if (std::stoi(argv[1]) < 1 || std::stoi(argv[2]) < 1) {
        std::cout << usage << std::endl;
        exit(-1);
    }

	particle_count = std::stoi(argv[1]);
    grid_size = std::stoi(argv[2]);
    if (argc > 3)
        method = argv[3];
}

void generate_random_particles(std::vector<float> &particles, unsigned int grid_size, unsigned int particle_count) {
    std::random_device rd;
    std::mt19937 gen(11);
    std::uniform_int_distribution<> pos_dist(0, grid_size); //? float ili int
    std::uniform_real_distribution<float> charge_dist(-5e-6, 5e-6);
    
    for (int i = 0; i < particle_count; ++i) {
        particles.push_back(pos_dist(gen));
        particles.push_back(pos_dist(gen));
        particles.push_back(pos_dist(gen));
        particles.push_back(charge_dist(gen));
    }
}

int main(int argc, char* argv[]) {
    // GPU device selection
    sycl::queue q;
    try {
        q = sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
        if (!q.get_device().has(sycl::aspect::gpu))
            throw std::runtime_error("No GPU device available");

        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    // Parsing args
    size_t particle_count = 1;
    size_t grid_size = 1;
    std::string sycl_method = "usm";
    parse_args(argc, argv, particle_count, grid_size, sycl_method);
    printf("Particle count: %zu, Grid size: %zu, Method: %s\n", particle_count, grid_size, sycl_method.c_str());

    // Value init
    std::vector<int> grid(3, grid_size);
    
    std::vector<float> particles;
    particles.reserve(particle_count * 4);
    generate_random_particles(particles, grid_size, particle_count);
    
    std::vector<float> energygrid(grid_size * grid_size * grid_size);

    // Calculation
    auto start = std::chrono::high_resolution_clock::now();
    for (int z = 0; z < grid_size; ++z) {
        std::vector<float> result;
        if (sycl_method == "usm") {
            result = dcs_usm(q, particles, sycl::range<2>{grid_size, grid_size}, z);
        } else if (sycl_method == "buffer") {
            result = dcs_buffer(q, particles, sycl::range<2>{grid_size, grid_size}, z);
        }
        std::copy(result.begin(), result.end(), energygrid.begin() + z*grid_size*grid_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds." << std::endl;

    float sum = 0.0f;
    for (int x = 0; x < energygrid.size(); ++x)
        sum += energygrid[x];
    std::cout << "Total sum: " << sum << std::endl;

    return 0;
}