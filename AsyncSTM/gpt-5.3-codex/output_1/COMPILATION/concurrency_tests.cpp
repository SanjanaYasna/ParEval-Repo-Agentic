#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

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
    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    std::atomic<int> value{0};
    std::vector<std::thread> workers;
    workers.reserve(64);

    for (unsigned i = 0; i < 64; ++i) {
      workers.emplace_back([&value]() { value.fetch_add(1, std::memory_order_relaxed); });
    }

    for (auto& t : workers) t.join();

    int result = value.load(std::memory_order_relaxed);

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";
  }

  // ---------------------------------------------------------
  // TEST 2: Exponential Growth (Squaring)
  // ---------------------------------------------------------
  {
    outfile << "Exponential Growth (Squaring) Test\n";
    outfile << "  Initial Value: 2\n";
    outfile << "  Threads: 4\n";
    outfile << "  Operation: A = A * A\n";

    int value = 2;
    std::mutex mtx;
    std::vector<std::thread> workers;
    workers.reserve(4);

    for (unsigned i = 0; i < 4; ++i) {
      workers.emplace_back([&]() {
        std::lock_guard<std::mutex> lock(mtx);
        value = value * value;
      });
    }

    for (auto& t : workers) t.join();

    int result = value;

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
  }

  outfile.close();
  return 0;
}
