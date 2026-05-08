////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Legion translation of unit_tests.cpp (default mapper, no custom mapping)

#include <fstream>
#include <iostream>

#include "legion.h"
#include "astm.hpp"

using namespace Legion;
using namespace astm;

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 1,
    CHECK_LOCAL_A_TASK_ID
};

struct CheckValueArgs {
    int expected;
    int actual;
};

void check_local_a_task(
    const Task* task,
    const std::vector<PhysicalRegion>& /*regions*/,
    Context /*ctx*/,
    Runtime* /*runtime*/)
{
    ASTM_TEST(task->arglen == sizeof(CheckValueArgs));
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
            local_A = int(A_);

            A_ = A_ - B_;
        } while (!t.commit_transaction());

        // In Legion, launch a task after commit to validate the captured read.
        CheckValueArgs args{16, local_A};
        TaskLauncher launcher(
            CHECK_LOCAL_A_TASK_ID, TaskArgument(&args, sizeof(args)));
        Future f = runtime->execute_task(ctx, launcher);
        f.get_void_result();

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
        TaskVariantRegistrar registrar(CHECK_LOCAL_A_TASK_ID, "check_local_a");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<check_local_a_task>(
            registrar, "check_local_a");
    }

    return Runtime::start(argc, argv);
}