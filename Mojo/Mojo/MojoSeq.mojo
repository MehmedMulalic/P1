import benchmark
import time
from memory import UnsafePointer
from sys import argv, exit
from random import rand, seed
from math import sqrt

alias K_E: Float32 = 8.987551786214e9
alias randomSeed = 12
alias qmin = SIMD[DType.float64, 1](Float32(-5e-6))
alias qmax = SIMD[DType.float64, 1](Float32(5e-6))
alias dtype = DType.float32


def main():
    fn generate_random_particles(grid_size: UInt32, pcount: UInt32, particle_matrix_size: Int) -> UnsafePointer[Float32]:
        seed(randomSeed)
        var ptr = UnsafePointer[Float32].alloc(particle_matrix_size)
        var max_position_value = SIMD[DType.float64, 1](grid_size)

        for i in range(pcount):
            rand[dtype](ptr+(i*4), 3, max=max_position_value)
            rand[dtype](ptr+(i*4)+3, 1, min=qmin, max=qmax)
        
        return ptr

    fn dcs() capturing -> None:
        var grid_size = 1000
        var particle_count = 800
        var particle_matrix_size = particle_count * 4
        
        var particles: UnsafePointer[Float32] = generate_random_particles(grid_size, particle_count, particle_matrix_size)
        var energygrid: UnsafePointer[Float32] = UnsafePointer[Float32].alloc(grid_size * grid_size * grid_size)

        for z in range(grid_size):
            for y in range(grid_size):
                for x in range(grid_size):
                    var energy: Float32 = 0.0
                    
                    for p in range(0, particle_matrix_size, 4):
                        var dx: Float32 = x - particles[p]
                        var dy: Float32 = y - particles[p+1]
                        var dz: Float32 = z - particles[p+2]

                        var r: Float32 = sqrt(dx*dx + dy*dy + dz*dz)
                        if (r > 0.0):
                            energy += particles[p+3] / r
                        
                    energy *= K_E
                    var index = z*grid_size*grid_size + y*grid_size + x
                    energygrid[index] = energy

    # Parsing args
    # var particle_count = 1
    # var grid_size = 1

    var args = argv()
    if len(args) > 1:
        if len(args) > 3 or len(args) == 2:
            print("Usage (positive values): <particle_count>, <grid_size>")
            exit(1)
        if atol(args[1]) < 1 or atol(args[2]) < 1:
            print("Usage (positive values): <particle_count>, <grid_size>")
            exit(1)
        particle_count = atol(args[1])
        grid_size = atol(args[2])

    # var particle_matrix_size = particle_count * 4
    # print("Particle count: ", particle_count, ", Grid size: " , grid_size, sep="")

    var report = benchmark.run[dcs](1, 3, 1, 50)
    var asda = time.time_function[func=dcs]()
    print(asda / 1e9)
    
    # var sum: Float32 = 0.0
    # for x in range(grid_size*grid_size*grid_size):
    #     sum += energygrid[x]
    
    # print(sum)
    report.print()

    # print("Simulation completed in", report.mean(), "seconds")