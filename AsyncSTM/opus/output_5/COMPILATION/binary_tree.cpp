////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include "legion.h"

using namespace Legion;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int NIL_NODE    = -1;
static const int MAX_NODES   = 64;
static const int MAX_VAL_LEN = 64;

// ---------------------------------------------------------------------------
// Task IDs
// ---------------------------------------------------------------------------
enum {
    TOP_LEVEL_TASK_ID,
    INSERT_TASK_ID,
    SEARCH_TASK_ID,
};

// ---------------------------------------------------------------------------
// Field IDs
// ---------------------------------------------------------------------------
enum {
    FID_KEY   = 100,
    FID_VALUE = 101,
    FID_LEFT  = 102,
    FID_RIGHT = 103,
};

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

// Fixed-size string wrapper so it can live inside a Legion region field.
struct NodeValue {
    char str[MAX_VAL_LEN];
};

// Metadata that tracks the root index and next free slot in the node array.
struct TreeState {
    int root_idx;
    int next_free;
};

// Serialisable arguments for insert / search tasks.
struct InsertArgs {
    int       key;
    NodeValue value;
    TreeState state;
};

struct SearchArgs {
    int       key;
    TreeState state;
};

struct SearchResult {
    int       found;          // 1 = found, 0 = not found
    NodeValue value;
};

// ---------------------------------------------------------------------------
// Insert Task
// ---------------------------------------------------------------------------
TreeState insert_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(InsertArgs));
    InsertArgs args;
    memcpy(&args, task->args, sizeof(InsertArgs));

    const FieldAccessor<READ_WRITE, int,       1> fa_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> fa_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_WRITE, int,       1> fa_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_WRITE, int,       1> fa_right(regions[0], FID_RIGHT);

    TreeState state = args.state;
    int new_idx = state.next_free++;

    // Initialise the new node.
    fa_key[new_idx]   = args.key;
    fa_val[new_idx]   = args.value;
    fa_left[new_idx]  = NIL_NODE;
    fa_right[new_idx] = NIL_NODE;

    if (state.root_idx == NIL_NODE) {
        // Tree was empty — new node becomes the root.
        state.root_idx = new_idx;
    } else {
        // Walk the tree to find the correct insertion point.
        int cur = state.root_idx;
        while (true) {
            if (args.key < fa_key[cur]) {
                if (fa_left[cur] != NIL_NODE) {
                    cur = fa_left[cur];
                } else {
                    fa_left[cur] = new_idx;
                    break;
                }
            } else {                        // key >= current  → go right
                if (fa_right[cur] != NIL_NODE) {
                    cur = fa_right[cur];
                } else {
                    fa_right[cur] = new_idx;
                    break;
                }
            }
        }
    }

    return state;
}

// ---------------------------------------------------------------------------
// Search Task
// ---------------------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(SearchArgs));
    SearchArgs args;
    memcpy(&args, task->args, sizeof(SearchArgs));

    const FieldAccessor<READ_ONLY, int,       1> fa_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_ONLY, NodeValue, 1> fa_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_ONLY, int,       1> fa_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_ONLY, int,       1> fa_right(regions[0], FID_RIGHT);

    SearchResult result;
    result.found = 0;
    memset(&result.value, 0, sizeof(NodeValue));

    int cur = args.state.root_idx;
    while (cur != NIL_NODE) {
        int cur_key = fa_key[cur];
        if (args.key == cur_key) {
            result.found = 1;
            result.value = fa_val[cur];
            return result;
        } else if (args.key < cur_key) {
            cur = fa_left[cur];
        } else {
            cur = fa_right[cur];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Top-Level Task
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    // ---- Create the logical region that backs the tree ----
    Rect<1> bounds(0, MAX_NODES - 1);
    IndexSpace is = runtime->create_index_space(ctx, bounds);
    runtime->attach_name(is, "tree_index_space");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),       FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int),       FID_LEFT);
        fa.allocate_field(sizeof(int),       FID_RIGHT);
    }
    runtime->attach_name(fs, "tree_field_space");

    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(tree_lr, "tree_region");

    // Tree metadata (root index + next-free slot)
    TreeState state;
    state.root_idx  = NIL_NODE;
    state.next_free = 0;

    // ---- 1. Data Insertion Phase ----
    struct DataEntry { int id; const char *val; };

    DataEntry data[] = {
        {50, "Root Node"},
        {30, "Left Child"},
        {70, "Right Child"},
        {20, "Left-Left Grandchild"},
        {40, "Left-Right Grandchild"},
        {60, "Right-Left Grandchild"},
        {80, "Right-Right Grandchild"},
    };
    const int num_entries = sizeof(data) / sizeof(data[0]);

    outfile << "Inserting Data:\n";
    for (int i = 0; i < num_entries; i++) {
        outfile << "  Insert(Key: " << data[i].id
                << ", Value: \"" << data[i].val << "\")" << std::endl;

        InsertArgs args;
        args.key   = data[i].id;
        memset(&args.value, 0, sizeof(NodeValue));
        strncpy(args.value.str, data[i].val, MAX_VAL_LEN - 1);
        args.state = state;

        TaskLauncher launcher(INSERT_TASK_ID,
                              TaskArgument(&args, sizeof(args)));
        launcher.add_region_requirement(
            RegionRequirement(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr));
        launcher.region_requirements[0].add_field(FID_KEY);
        launcher.region_requirements[0].add_field(FID_VALUE);
        launcher.region_requirements[0].add_field(FID_LEFT);
        launcher.region_requirements[0].add_field(FID_RIGHT);

        Future f = runtime->execute_task(ctx, launcher);
        state = f.get_result<TreeState>();      // block — next insert needs updated state
    }
    outfile << "\n";

    // ---- 2. Search Test Phase ----
    outfile << "Running Search Tests:\n";

    int search_keys[] = {99, 10, 50, 20, 60, 80, 45};
    const int num_searches = sizeof(search_keys) / sizeof(search_keys[0]);

    for (int i = 0; i < num_searches; i++) {
        int search_key = search_keys[i];
        outfile << "  Search(Key: " << search_key << ") -> ";

        SearchArgs sargs;
        sargs.key   = search_key;
        sargs.state = state;

        TaskLauncher launcher(SEARCH_TASK_ID,
                              TaskArgument(&sargs, sizeof(sargs)));
        launcher.add_region_requirement(
            RegionRequirement(tree_lr, READ_ONLY, EXCLUSIVE, tree_lr));
        launcher.region_requirements[0].add_field(FID_KEY);
        launcher.region_requirements[0].add_field(FID_VALUE);
        launcher.region_requirements[0].add_field(FID_LEFT);
        launcher.region_requirements[0].add_field(FID_RIGHT);

        Future f = runtime->execute_task(ctx, launcher);
        SearchResult result = f.get_result<SearchResult>();

        if (result.found) {
            outfile << "Found! Result Value: \"" << result.value.str << "\"\n";
        } else {
            outfile << "Not Found.\n";
        }
    }

    outfile.close();

    // ---- Clean up ----
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// Main — register tasks and start the Legion runtime
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(INSERT_TASK_ID, "insert");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<TreeState, insert_task>(registrar, "insert");
    }

    {
        TaskVariantRegistrar registrar(SEARCH_TASK_ID, "search");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<SearchResult, search_task>(registrar, "search");
    }

    return Runtime::start(argc, argv);
}
