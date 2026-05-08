////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Translated to Legion execution model (default mapper).
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>

#include "legion.h"
#include "astm.hpp"

using namespace Legion;
using namespace astm;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1,
    CHECK_VALUE_TASK_ID
};

struct CheckValueArgs
{
    int actual;
    int expected;
};

void check_value_task(
    const Task* task,
    const std::vector<PhysicalRegion>& /*regions*/,
    Context /*ctx*/,
    Runtime* /*runtime*/)
{
    const auto* args = static_cast<const CheckValueArgs*>(task->args);
    ASTM_TEST(args != nullptr);
    ASTM_TEST(args->actual == args->expected);
}

void top_level_task(
    const Task* /*task*/,
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

    { // Read A to a Legion task future.
        shared_var<int> A(4);
        shared_var<int> B(1);

        int local_A = 0;
        transaction t;
        do {
            auto A_ = A.get_local(t);
            auto B_ = B.get_local(t);

            A_ = A_ * A_;
            local_A = int(A_);

            A_ = A_ - B_;
        } while (!t.commit_transaction());

        CheckValueArgs args{local_A, 16};
        TaskLauncher check_launcher(CHECK_VALUE_TASK_ID, TaskArgument(&args, sizeof(args)));
        Future check_future = runtime->execute_task(ctx, check_launcher);
        check_future.get_void_result();

        outfile << "Test (Read A to Future): A = " << A.read() << ", B = " << B.read() << std::endl;
    }

    { // Retry logic
        shared_var<int> A(4);

        bool fail = true;
        int attempt_count = 0;
        transaction t;
        do {
            ++attempt_count;

            auto A_ = A.get_local(t);
            int tmp = A_ * A_;

            if (fail) {
                A.write(3);
                fail = false;
            }

            A_ = tmp;
        } while (!t.commit_transaction());

        outfile << "Test (Retry Logic): A = " << A.read()
                << ", Attempts = " << attempt_count << std::endl;
    }
}

int main(int argc, char** argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
    }

    {
        TaskVariantRegistrar registrar(CHECK_VALUE_TASK_ID, "check_value_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<check_value_task>(registrar, "check_value_task");
    }

    return Runtime::start(argc, argv);
}
