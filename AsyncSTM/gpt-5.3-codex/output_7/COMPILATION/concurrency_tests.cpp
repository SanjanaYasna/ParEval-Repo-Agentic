////////////////////////////////////////////////////////////////////////////////
//  Translated from HPX-style async usage to Legion task launches.
//  Assumes default mapper and a single shared address space for task args.
////////////////////////////////////////////////////////////////////////////////

#include <legion.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

#include "astm.hpp"

using namespace Legion;
using namespace astm;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1,
    INCREMENT_TASK_ID,
    SQUARE_TASK_ID
};

struct SharedVarArg
{
    uintptr_t var_ptr;
};

// A = A * A (transactional)
void square_transaction(shared_var<int>& A)
{
    transaction t;
    do {
        auto A_ = A.get_local(t);
        A_ = A_ * A_;
    } while (!t.commit_transaction());
}

// A = A + 1 (transactional)
void increment_transaction(shared_var<int>& A)
{
    transaction t;
    do {
        auto A_ = A.get_local(t);
        A_ = A_ + 1;
    } while (!t.commit_transaction());
}

void increment_task(const Task* task,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*)
{
    const auto* arg = static_cast<const SharedVarArg*>(task->args);
    auto* A = reinterpret_cast<shared_var<int>*>(arg->var_ptr);
    increment_transaction(*A);
}

void square_task(const Task* task,
                 const std::vector<PhysicalRegion>&,
                 Context,
                 Runtime*)
{
    const auto* arg = static_cast<const SharedVarArg*>(task->args);
    auto* A = reinterpret_cast<shared_var<int>*>(arg->var_ptr);
    square_transaction(*A);
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime)
{
    std::ofstream outfile("concurrency_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
        return;
    }

    // ---------------------------------------------------------
    // TEST 1: High Concurrency Increment
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        shared_var<int> A(0);
        SharedVarArg arg{reinterpret_cast<uintptr_t>(&A)};
        std::vector<Future> futures;
        futures.reserve(64);

        for (unsigned i = 0; i < 64; ++i) {
            TaskLauncher launcher(INCREMENT_TASK_ID, TaskArgument(&arg, sizeof(arg)));
            futures.push_back(runtime->execute_task(ctx, launcher));
        }

        for (auto& f : futures) {
            f.get_void_result();
        }

        int result = 0;
        {
            transaction t;
            result = A.get_local(t).get();
        }

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

        shared_var<int> A(2);
        SharedVarArg arg{reinterpret_cast<uintptr_t>(&A)};
        std::vector<Future> futures;
        futures.reserve(4);

        for (unsigned i = 0; i < 4; ++i) {
            TaskLauncher launcher(SQUARE_TASK_ID, TaskArgument(&arg, sizeof(arg)));
            futures.push_back(runtime->execute_task(ctx, launcher));
        }

        for (auto& f : futures) {
            f.get_void_result();
        }

        int result = 0;
        {
            transaction t;
            result = A.get_local(t).get();
        }

        outfile << "  Expected Result: 65536\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
    }

    outfile.close();
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
        TaskVariantRegistrar registrar(INCREMENT_TASK_ID, "increment_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<increment_task>(registrar, "increment_task");
    }

    {
        TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<square_task>(registrar, "square_task");
    }

    return Runtime::start(argc, argv);
}
