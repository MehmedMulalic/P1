#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    constexpr size_t N = 1024;
    std::vector<int> A(N, 1);
    std::vector<int> B(N, 2);
    std::vector<int> C(N, 0);

    sycl::queue q;

    {
        sycl::buffer<int> bufA(A.data(), N);
        sycl::buffer<int> bufB(B.data(), N);
        sycl::buffer<int> bufC(C.data(), N);

        q.submit([&](sycl::handler& h) {
            auto a = bufA.get_access<sycl::access::mode::read>(h);
            auto b = bufB.get_access<sycl::access::mode::read>(h);
            auto c = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class vector_add>(sycl::range<1>(N), [=](sycl::id<1> i) {
                c[i] = a[i] + b[i];
            });
        });
    } // buffers go out of scope here, data is copied back to C

    // Print some results
    std::cout << "C[0] = " << C[0] << ", C[N-1] = " << C[N-1] << "\n";

    return 0;
}
