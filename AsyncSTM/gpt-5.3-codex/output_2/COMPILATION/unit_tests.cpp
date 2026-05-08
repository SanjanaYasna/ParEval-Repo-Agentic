////////////////////////////////////////////////////////////////////////////////
//  ASTM unit tests (standalone)
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>

#include "astm.hpp"

using namespace astm;

int main()
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return 1;
    }

    { // Read A, Write A
        shared_var<int> A(1);
        transaction t;
        do {
            auto A_ = A.get_local(t);
            A_ = 2;
        } while (!t.commit_transaction());

        outfile << "Test (Read A, Write A): A = " << A.read() << std::endl;
    }

    { // Write A, Read A
        shared_var<int> A(1);

        transaction t;
        do {
            auto A_ = A.get_local(t);
            A_ = 2;
        } while (!t.commit_transaction());

        outfile << "Test (Write A, Read A): A = " << A.read() << std::endl;
    }

    { // Write A, Write A
        shared_var<int> A(1);

        transaction t;
        do {
            auto A_ = A.get_local(t);
            ASTM_TEST(A_ == 1);
            ASTM_TEST(A_ == 1);
        } while (!t.commit_transaction());

        outfile << "Test (Write A, Write A): A = " << A.read() << std::endl;
    }

    { // Write A, Write A (overwrite)
        shared_var<int> A(1);

        transaction t;
        do {
            auto A_ = A.get_local(t);
            A_ = 2;
            A_ = 2;
        } while (!t.commit_transaction());

        outfile << "Test (Overwrite): A = " << A.read() << std::endl;
    }

    { // Read A, Write A, Read A
        shared_var<int> A(1);

        transaction t;
        do {
            auto A_ = A.get_local(t);
            A_ = 2;
        } while (!t.commit_transaction());

        outfile << "Test (Read -> Write -> Read): A = " << A.read() << std::endl;
    }

    { // Write A, Read A, Write A
        shared_var<int> A(1);

        transaction t;
        do {
            auto A_ = A.get_local(t);
            A_ = 2;
            A_ = 2;
        } while (!t.commit_transaction());

        outfile << "Test (Write -> Read -> Write): A = " << A.read() << std::endl;
    }

    { // Basic arithmetic with local_vars.
        shared_var<int> A(4);
        shared_var<int> B(1);

        // atomic { A = A*A - B; }
        transaction t;
        do {
            auto A_ = A.get_local(t);
            auto B_ = B.get_local(t);

            A_ = A_ * A_ - B_;
        } while (!t.commit_transaction());

        outfile << "Test (Arithmetic): A = " << A.read() << ", B = " << B.read() << std::endl;
    }

    { // Read A to future.
        shared_var<int> A(4);
        shared_var<int> B(1);

        transaction t;
        transaction_future IO(t);

        do {
            auto A_ = A.get_local(t);
            auto B_ = B.get_local(t);

            A_ = A_ * A_;

            IO.then(
                [local_A = int(A_)](transaction*) { ASTM_TEST(local_A == 16); }
            );

            A_ = A_ - B_;
        } while (!t.commit_transaction());

        IO.get();
        outfile << "Test (Read A to Future): A = " << A.read() << ", B = " << B.read() << std::endl;
    }

    {
        shared_var<int> A(4);

        // atomic { A = A*A; }
        bool fail = true;
        int attempt_count = 0;
        transaction t;
        do {
            ++attempt_count;

            auto A_ = A.get_local(t);

            int tmp = A_ * A_;

            if (fail)
            {
                A.write(3);
                fail = false;
            }

            A_ = tmp;
        } while (!t.commit_transaction());

        outfile << "Test (Retry Logic): A = " << A.read()
                << ", Attempts = <ignored>" << std::endl;
    }

    outfile.close();
    return 0;
}
