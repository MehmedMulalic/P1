__kernel void dcs(__global float *energygrid, __global float *particles, const int z, const int n, const float k_e) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int id = (i*get_global_size(1)) + j;

    int atoms = n * 4;
    float energy = 0.0f;

    for (int p = 0; p < atoms; p+=4) {
        float dx = i - particles[p];
        float dy = j - particles[p+1];
        float dz = z - particles[p+2];

        float r = sqrt(dx*dx + dy*dy + dz*dz);

        if (r > 0.0f) energy += particles[p+3] / r;
    }

    energy *= k_e;
    energygrid[id] = energy;
}
