__kernel void dcs(__global float* restrict energygrid, __global float* restrict particles, const int z, const int atom_count, const float k_e) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t id = (i*get_global_size(1)) + j;

    const size_t atoms = atom_count * 4;
    float energy = 0.0f;

    for (size_t p = 0; p < atoms; p+=4) {
        float dx = (float)(i) - particles[p];
        float dy = (float)(j) - particles[p+1];
        float dz = (float)(z) - particles[p+2];

        float r = sqrt(dx*dx + dy*dy + dz*dz);

        if (r > 0.0f) energy += particles[p+3] / r;
    }

    energygrid[id] = energy * k_e;
}
