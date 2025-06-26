#include <iostream>
#include <vector>
#include <chrono>

int main() {
    constexpr size_t N = 1024 * 2000000;
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

    // Calculation
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vector addition completed in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
