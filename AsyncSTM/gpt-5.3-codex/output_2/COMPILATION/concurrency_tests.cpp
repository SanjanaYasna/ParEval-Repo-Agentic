////////////////////////////////////////////////////////////////////////////////
//  Concurrency stress tests for ASTM
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "astm.hpp"

using namespace astm;

// A = A * A
void square_transaction(shared_var<int>& A)
{
    transaction t;
    do {
        auto A_ = A.get_local(t);
        A_ = A_ * A_;
    } while (!t.commit_transaction());
}

// A = A + 1
void increment_transaction(shared_var<int>& A)
{
    transaction t;
    do {
        auto A_ = A.get_local(t);
        A_ = A_ + 1;
    } while (!t.commit_transaction());
}

int main()
{
    std::ofstream outfile("concurrency_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
        return 1;
    }

    // ---------------------------------------------------------
    // TEST 1: High Concurrency Increment
    // Goal: Ensure 64 threads atomically increment a shared variable.
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        shared_var<int> A(0);

        std::vector<std::thread> threads;
        threads.reserve(64);
        for (int i = 0; i < 64; ++i)
            threads.emplace_back([&A]() { increment_transaction(A); });
        for (auto& th : threads)
            th.join();

        int result = A.read();

        outfile << "  Expected Result: 64\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";
    }

    // ---------------------------------------------------------
    // TEST 2: Exponential Growth (Squaring)
    // 2 -> 4 -> 16 -> 256 -> 65536
    // ---------------------------------------------------------
    {
        outfile << "Exponential Growth (Squaring) Test\n";
        outfile << "  Initial Value: 2\n";
        outfile << "  Threads: 4\n";
        outfile << "  Operation: A = A * A\n";

        shared_var<int> A(2);

        std::vector<std::thread> threads;
        threads.reserve(4);
        for (int i = 0; i < 4; ++i)
            threads.emplace_back([&A]() { square_transaction(A); });
        for (auto& th : threads)
            th.join();

        int result = A.read();

        outfile << "  Expected Result: 65536\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
    }

    outfile.close();
    return 0;
}
