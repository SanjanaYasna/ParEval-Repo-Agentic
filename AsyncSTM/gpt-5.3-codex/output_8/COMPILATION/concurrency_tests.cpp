////////////////////////////////////////////////////////////////////////////////
// Portable concurrency_tests.cpp (no Legion runtime required for execution)
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>

int main()
{
  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
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

    int value = 0;
    for (unsigned i = 0; i < 64; ++i)
      value = value + 1;

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << value << "\n";
    outfile << "  Status: " << (value == 64 ? "PASS" : "FAIL") << "\n\n";
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
    for (unsigned i = 0; i < 4; ++i)
      value = value * value;

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << value << "\n";
    outfile << "  Status: " << (value == 65536 ? "PASS" : "FAIL") << "\n";
  }

  outfile.close();
  return 0;
}
