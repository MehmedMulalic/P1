#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <chrono>

const float k_e = 8.987551786214e9; // Value from Wikipedia: https://en.wikipedia.org/wiki/Coulomb%27s_law

struct Particle {
    int x, y, z;
    float q;
};

// change to vector for kernel
struct Vector3D {
    int x, y, z;

    Vector3D(int x, int y, int z) : x(x), y(y), z(z) {}
};

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

std::vector<Particle> generate_random_particles(int num_particles, int grid_size) {
    std::vector<Particle> particles;
    particles.reserve(num_particles);
    
    std::random_device rd;
    std::mt19937 gen(11);
    std::uniform_int_distribution<> pos_dist(0, grid_size);
    std::uniform_real_distribution<float> charge_dist(-5e-6, 5e-6);
    
    for (int i = 0; i < num_particles; ++i) {
        particles.push_back({
            pos_dist(gen),
            pos_dist(gen),
            pos_dist(gen),
            charge_dist(gen)
        });
    }
    
    return particles;
}

float distance(const Particle a, const Vector3D p) {
    float dx = a.x - p.x;
    float dy = a.y - p.y;
    float dz = a.z - p.z;

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// U = k_e * (Ei qi/ri)
void coulomb_energy(const std::vector<Particle> &particles, const Vector3D grid, std::vector<float> &results) {
    size_t n = particles.size();

    for (size_t z = 0; z < grid.z; ++z) {
        for (size_t y = 0; y < grid.y; ++y) {
            for (size_t x = 0; x < grid.x; ++x) {
                float energy = 0.0f;
                for (size_t a = 0; a < n; ++a) {
                    float r = distance(particles[a], Vector3D(x, y, z));
                    if (r > 0) energy += particles[a].q / r;
                }

                energy *= k_e;
                results.push_back(energy);
            }
        }
    }
}

int main(int argc, char* argv[]) {
	// Parsing args
    unsigned int particle_count = 1;
    unsigned int grid_size = 1;
    parse_args(argc, argv, particle_count, grid_size);
    printf("Particle count: %d, Grid size: %d\n", particle_count, grid_size);
    
    Vector3D grid(grid_size, grid_size, grid_size);
    std::vector<Particle> particles = generate_random_particles(particle_count, grid_size);
    std::vector<float> results;
    results.reserve(grid_size * grid_size * grid_size);

    auto start = std::chrono::high_resolution_clock::now();

    coulomb_energy(particles, grid, results);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds." << std::endl;
    
    // float sum = 0.0f;
    // for (int x = 0; x < results.size(); ++x) {
    //     sum += results[x];
    // }
    // std::cout << "Total sum: " << sum << std::endl;
	return 0;
}
