from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from memory import UnsafePointer
from layout import Layout, LayoutTensor
from time import monotonic
from sys import argv, exit
from random import rand, seed
from algorithm import parallel_memcpy
from math import sqrt

alias K_E: Float32 = 8.987551786214e9
alias randomSeed = 12
alias qmin = SIMD[DType.float64, 1](Float32(-5e-6))
alias qmax = SIMD[DType.float64, 1](Float32(5e-6))
alias dtype = DType.float32

# Naglasi u reportu da nisam rekreirao identicne random brojeve sto moze predstavljati malu deviaciju u poredenju
fn generate_random_particles(grid_size: UInt32, pcount: UInt32, particle_matrix_size: Int) -> UnsafePointer[Float32]:
    seed(randomSeed)
    var ptr = UnsafePointer[Float32].alloc(particle_matrix_size)
    var max_position_value = SIMD[DType.float64, 1](grid_size)

    for i in range(pcount):
        rand[dtype](ptr+(i*4), 3, max=max_position_value)
        rand[dtype](ptr+(i*4)+3, 1, min=qmin, max=qmax)
    
    return ptr

def main():
    fn dcs(output: UnsafePointer[Scalar[dtype]], particles: UnsafePointer[Scalar[dtype]], particle_matrix_size: Int, z: Int, grid_size: Int):
        var global_x = block_idx.x * block_dim.x + thread_idx.x
        var global_y = block_idx.y * block_dim.y + thread_idx.y

        if global_x >= grid_size or global_y >= grid_size:
            return

        var energy: Float32 = 0.0
        for p in range(0, particle_matrix_size, 4):
            var dx: Float32 = global_x - particles[p]
            var dy: Float32 = global_y - particles[p+1]
            var dz: Float32 = z - particles[p+2]

            var r: Float32 = sqrt(dx*dx + dy*dy + dz*dz)

            if (r > 0.0):
                energy += particles[p+3] / r

        energy *= K_E
        output[(global_y*grid_size)+global_x] = energy


    # Parsing args
    var particle_count = 1
    var grid_size = 1

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

    var particle_matrix_size = particle_count * 4
    print("Particle count: ", particle_count, ", Grid size: " , grid_size, sep="")

    # Value init
    var particles: UnsafePointer[Float32] = generate_random_particles(grid_size, particle_count, particle_matrix_size)
    var energygrid: UnsafePointer[Float32] = UnsafePointer[Float32].alloc(grid_size * grid_size * grid_size)

    # Buffer allocation
    var ctx: DeviceContext = DeviceContext()

    var host_buffer = ctx.enqueue_create_host_buffer[dtype](grid_size * grid_size)
    var device_buffer = ctx.enqueue_create_buffer[dtype](grid_size * grid_size)

    var particle_buffer = ctx.enqueue_create_buffer[dtype](particle_matrix_size)
    particle_buffer.enqueue_copy_from(particles)

    var block_dim = (16, 16)
    var grid_dim = (
        (grid_size + block_dim[0] - 1),
        (grid_size + block_dim[1] - 1)
    )

    # Calculation
    var start: UInt = monotonic()

    for z in range(grid_size):
        ctx.enqueue_function[dcs](device_buffer, particle_buffer, particle_matrix_size, z, grid_size, grid_dim=grid_dim, block_dim=block_dim)
        device_buffer.enqueue_copy_to(host_buffer)
        ctx.synchronize()
        parallel_memcpy[dtype](energygrid+(z*grid_size*grid_size), host_buffer.unsafe_ptr(), grid_size*grid_size)

    var end: UInt = monotonic()

    var elapsed_time: Float64 = (end-start) / 1e9
    print("Simulation completed in", elapsed_time, "seconds")