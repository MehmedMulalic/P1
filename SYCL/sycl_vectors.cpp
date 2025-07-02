#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

int main() {
    constexpr size_t N = 1024 * 2000000;

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0);

    sycl::queue q;

    sycl::buffer<float> bufA(A.data(), N);
    sycl::buffer<float> bufB(B.data(), N);
    sycl::buffer<float> bufC(C.data(), N);

    // Calculation
    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        auto a = bufA.get_access<sycl::access::mode::read>(h);
        auto b = bufB.get_access<sycl::access::mode::read>(h);
        auto c = bufC.get_access<sycl::access::mode::write>(h);

        h.parallel_for(N, [=](sycl::id<1> idx) {
            c[idx] = a[idx] + b[idx];
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vector addition completed in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
