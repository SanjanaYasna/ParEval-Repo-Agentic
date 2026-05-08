////////////////////////////////////////////////////////////////////////////////
// Concurrency tests using ASTM transactions (standalone C++17 executable)
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "astm.hpp"

using namespace astm;

static void increment_once(shared_var<int>& a)
{
  transaction t;
  do {
    auto A_ = a.get_local(t);
    const int cur = A_.get();
    A_ = cur + 1;
  } while (!t.commit_transaction());
}

static void square_once(shared_var<int>& a)
{
  transaction t;
  do {
    auto A_ = a.get_local(t);
    const int cur = A_.get();
    A_ = cur * cur;
  } while (!t.commit_transaction());
}

int main()
{
  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt\n";
    return 1;
  }

  // ---------------------------------------------------------
  // TEST 1: High Concurrency Increment
  // ---------------------------------------------------------
  {
    shared_var<int> A(0);

    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    std::vector<std::thread> workers;
    workers.reserve(64);

    for (unsigned i = 0; i < 64; ++i) {
      workers.emplace_back([&A]() { increment_once(A); });
    }
    for (auto& th : workers) th.join();

    const int result = A.read();

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";
  }

  // ---------------------------------------------------------
  // TEST 2: Exponential Growth (Squaring)
  // ---------------------------------------------------------
  {
    shared_var<int> A(2);

    outfile << "Exponential Growth (Squaring) Test\n";
    outfile << "  Initial Value: 2\n";
    outfile << "  Threads: 4\n";
    outfile << "  Operation: A = A * A\n";

    std::vector<std::thread> workers;
    workers.reserve(4);

    for (unsigned i = 0; i < 4; ++i) {
      workers.emplace_back([&A]() { square_once(A); });
    }
    for (auto& th : workers) th.join();

    const int result = A.read();

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
  }

  outfile.close();
  return 0;
}
