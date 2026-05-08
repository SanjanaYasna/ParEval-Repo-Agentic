////////////////////////////////////////////////////////////////////////////////
//  Legion translation of unit_tests.cpp
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <cassert>
#include <fstream>
#include <iostream>

using namespace Legion;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1,
    VERIFY_LOCAL_A_TASK_ID
};

// A small async check task used to mimic IO.then(...).
bool verify_local_a_task(const Task* task,
                         const std::vector<PhysicalRegion>& /*regions*/,
                         Context /*ctx*/,
                         Runtime* /*runtime*/)
{
    assert(task->arglen == sizeof(int));
    const int local_A = *reinterpret_cast<const int*>(task->args);
    return (local_A == 16);
}

void top_level_task(const Task* /*task*/,
                    const std::vector<PhysicalRegion>& /*regions*/,
                    Context ctx,
                    Runtime* runtime)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    { // Read A, Write A
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            A_ = 2;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Read A, Write A): A = " << A << std::endl;
    }

    { // Write A, Read A
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            A_ = 2;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Write A, Read A): A = " << A << std::endl;
    }

    { // Write A, Write A
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            assert(A_ == 1);
            assert(A_ == 1);
            committed = true;
        } while (!committed);

        outfile << "Test (Write A, Write A): A = " << A << std::endl;
    }

    { // Write A, Write A (overwrite)
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            A_ = 2;
            A_ = 2;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Overwrite): A = " << A << std::endl;
    }

    { // Read A, Write A, Read A
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            A_ = 2;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Read -> Write -> Read): A = " << A << std::endl;
    }

    { // Write A, Read A, Write A
        int A = 1;
        bool committed = false;
        do {
            int A_ = A;
            A_ = 2;
            A_ = 2;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Write -> Read -> Write): A = " << A << std::endl;
    }

    { // Basic arithmetic with local vars
        int A = 4;
        int B = 1;

        bool committed = false;
        do {
            int A_ = A;
            int B_ = B;
            A_ = A_ * A_ - B_;
            A = A_;
            committed = true;
        } while (!committed);

        outfile << "Test (Arithmetic): A = " << A << ", B = " << B << std::endl;
    }

    { // Read A to future (Legion Future via async task)
        int A = 4;
        int B = 1;

        bool committed = false;
        Future io_future;
        do {
            int A_ = A;
            int B_ = B;

            A_ = A_ * A_;

            TaskLauncher launcher(
                VERIFY_LOCAL_A_TASK_ID, TaskArgument(&A_, sizeof(A_)));
            io_future = runtime->execute_task(ctx, launcher);

            A_ = A_ - B_;
            A = A_;
            committed = true;
        } while (!committed);

        bool ok = io_future.get_result<bool>();
        assert(ok);

        outfile << "Test (Read A to Future): A = " << A << ", B = " << B << std::endl;
    }

    { // Retry logic on conflict
        int A = 4;
        bool fail = true;
        int attempt_count = 0;

        bool committed = false;
        do {
            ++attempt_count;

            const int read_A = A;
            const int tmp = read_A * read_A;

            if (fail) {
                // External conflicting write before commit attempt.
                A = 3;
                fail = false;
            }

            // Commit check: fail if A changed since read.
            if (A != read_A) {
                committed = false;
                continue;
            }

            A = tmp;
            committed = true;
        } while (!committed);

        outfile << "Test (Retry Logic): A = " << A
                << ", Attempts = " << attempt_count << std::endl;
    }
}

int main(int argc, char** argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(VERIFY_LOCAL_A_TASK_ID, "verify_local_a");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<bool, verify_local_a_task>(
            registrar, "verify_local_a");
    }

    return Runtime::start(argc, argv);
}
