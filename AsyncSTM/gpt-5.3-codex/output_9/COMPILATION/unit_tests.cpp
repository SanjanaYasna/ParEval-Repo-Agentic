////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>

#include "legion.h"
#include "astm.hpp"

using namespace Legion;
using namespace astm;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1,
    VERIFY_LOCAL_A_TASK_ID
};

void verify_local_a_task(
    const Task* task,
    const std::vector<PhysicalRegion>&,
    Context,
    Runtime*)
{
    ASTM_TEST(task->arglen == sizeof(int));
    int local_A = *static_cast<const int*>(task->args);
    ASTM_TEST(local_A == 16);
}

void top_level_task(
    const Task*,
    const std::vector<PhysicalRegion>&,
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

        transaction t;
        do {
            auto A_ = A.get_local(t);
            auto B_ = B.get_local(t);

            A_ = A_ * A_ - B_;
        } while (!t.commit_transaction());

        outfile << "Test (Arithmetic): A = " << A.read()
                << ", B = " << B.read() << std::endl;
    }

    { // Read A to Legion future task.
        shared_var<int> A(4);
        shared_var<int> B(1);

        int local_A = 0;
        transaction t;
        do {
            auto A_ = A.get_local(t);
            auto B_ = B.get_local(t);

            A_ = A_ * A_;
            local_A = int(A_); // capture value equivalent to original continuation

            A_ = A_ - B_;
        } while (!t.commit_transaction());

        TaskLauncher verify_launcher(
            VERIFY_LOCAL_A_TASK_ID, TaskArgument(&local_A, sizeof(local_A)));
        Future io_future = runtime->execute_task(ctx, verify_launcher);
        io_future.get_void_result();

        outfile << "Test (Read A to Future): A = " << A.read()
                << ", B = " << B.read() << std::endl;
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

    outfile.close();
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
        Runtime::preregister_task_variant<verify_local_a_task>(
            registrar, "verify_local_a");
    }

    return Runtime::start(argc, argv);
}
