////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include "legion.h"

using namespace Legion;

// Task IDs
enum {
    TOP_LEVEL_TASK_ID,
    INSERT_TASK_ID,
    SEARCH_TASK_ID,
};

// Field IDs for the tree node region
enum {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int MAX_NODES = 128;
static const int MAX_VALUE_LEN = 64;
static const int NULL_NODE = -1;

// Fixed-size string wrapper for storing values in a Legion region field
struct NodeValue {
    char str[MAX_VALUE_LEN];
};

// Arguments passed to the insert task
struct InsertArgs {
    int key;
    NodeValue value;
    int root_idx;
    int next_free;
};

// Result returned from the insert task
struct InsertResult {
    int root_idx;
    int next_free;
};

// Arguments passed to the search task
struct SearchArgs {
    int search_key;
    int root_idx;
};

// Result returned from the search task
struct SearchResult {
    bool found;
    int node_idx;
    NodeValue value;
};

//--------------------------------------------------------------------------
// Insert Task
// Allocates a new node in the region and links it into the tree.
// Mirrors the transactional insert of the original ASTM binary_tree.
//--------------------------------------------------------------------------
InsertResult insert_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    InsertArgs args = *(const InsertArgs *)task->args;

    const FieldAccessor<READ_WRITE, int, 1> acc_key(regions[0], FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> acc_value(regions[0], FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1> acc_left(regions[0], FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1> acc_right(regions[0], FID_RIGHT);

    // Allocate new node at next_free slot
    int new_idx = args.next_free;
    acc_key[new_idx] = args.key;
    acc_value[new_idx] = args.value;
    acc_left[new_idx] = NULL_NODE;
    acc_right[new_idx] = NULL_NODE;

    InsertResult result;
    result.next_free = args.next_free + 1;

    // If tree is empty, new node becomes root
    if (args.root_idx == NULL_NODE) {
        result.root_idx = new_idx;
        return result;
    }

    result.root_idx = args.root_idx;

    // Traverse tree to find the correct insertion point
    // Mirrors the recursive insert logic: key < node.key goes left,
    // key >= node.key goes right
    int current = args.root_idx;
    while (true) {
        if (args.key < acc_key[current]) {
            if (acc_left[current] == NULL_NODE) {
                acc_left[current] = new_idx;
                return result;
            }
            current = acc_left[current];
        } else { // key >= current key
            if (acc_right[current] == NULL_NODE) {
                acc_right[current] = new_idx;
                return result;
            }
            current = acc_right[current];
        }
    }
}

//--------------------------------------------------------------------------
// Search Task
// Traverses the tree in the region to find a node by key.
// Mirrors the transactional search of the original ASTM binary_tree.
//--------------------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    SearchArgs args = *(const SearchArgs *)task->args;

    const FieldAccessor<READ_ONLY, int, 1> acc_key(regions[0], FID_KEY);
    const FieldAccessor<READ_ONLY, NodeValue, 1> acc_value(regions[0], FID_VALUE);
    const FieldAccessor<READ_ONLY, int, 1> acc_left(regions[0], FID_LEFT);
    const FieldAccessor<READ_ONLY, int, 1> acc_right(regions[0], FID_RIGHT);

    SearchResult result;
    result.found = false;
    result.node_idx = NULL_NODE;
    memset(result.value.str, 0, MAX_VALUE_LEN);

    int current = args.root_idx;
    while (current != NULL_NODE) {
        int cur_key = acc_key[current];
        if (args.search_key == cur_key) {
            result.found = true;
            result.node_idx = current;
            result.value = acc_value[current];
            return result;
        } else if (args.search_key < cur_key) {
            current = acc_left[current];
        } else {
            current = acc_right[current];
        }
    }

    return result;
}

//--------------------------------------------------------------------------
// Top-Level Task
// Orchestrates tree creation, data insertion, and search tests.
// Output is written to binary_tree.txt to match the original program.
//--------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    {
        // Create index space for tree node storage
        Rect<1> bounds(0, MAX_NODES - 1);
        IndexSpaceT<1> is = runtime->create_index_space(ctx, bounds);
        runtime->attach_name(is, "tree_node_is");

        // Create field space with tree node fields
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(int), FID_KEY);
            fa.allocate_field(sizeof(NodeValue), FID_VALUE);
            fa.allocate_field(sizeof(int), FID_LEFT);
            fa.allocate_field(sizeof(int), FID_RIGHT);
        }
        runtime->attach_name(fs, "tree_node_fs");

        // Create logical region for the tree
        LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
        runtime->attach_name(tree_lr, "tree_lr");

        // Initialize the region: set all child pointers to NULL_NODE
        {
            InlineLauncher il(RegionRequirement(tree_lr, WRITE_DISCARD,
                                                EXCLUSIVE, tree_lr));
            il.add_field(FID_KEY);
            il.add_field(FID_VALUE);
            il.add_field(FID_LEFT);
            il.add_field(FID_RIGHT);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<WRITE_DISCARD, int, 1> acc_key(pr, FID_KEY);
            const FieldAccessor<WRITE_DISCARD, NodeValue, 1> acc_value(pr, FID_VALUE);
            const FieldAccessor<WRITE_DISCARD, int, 1> acc_left(pr, FID_LEFT);
            const FieldAccessor<WRITE_DISCARD, int, 1> acc_right(pr, FID_RIGHT);

            NodeValue empty_val;
            memset(empty_val.str, 0, MAX_VALUE_LEN);
            for (int i = 0; i < MAX_NODES; i++) {
                acc_key[i] = 0;
                acc_value[i] = empty_val;
                acc_left[i] = NULL_NODE;
                acc_right[i] = NULL_NODE;
            }

            runtime->unmap_region(ctx, pr);
        }

        int root_idx = NULL_NODE;
        int next_free = 0;

        // 1. Data Insertion Phase
        struct DataEntry {
            int id;
            const char *val;
        };

        std::vector<DataEntry> data = {
            {50, "Root Node"},
            {30, "Left Child"},
            {70, "Right Child"},
            {20, "Left-Left Grandchild"},
            {40, "Left-Right Grandchild"},
            {60, "Right-Left Grandchild"},
            {80, "Right-Right Grandchild"}
        };

        outfile << "Inserting Data:\n";
        for (const auto &entry : data) {
            outfile << "  Insert(Key: " << entry.id
                    << ", Value: \"" << entry.val << "\")" << std::endl;

            InsertArgs args;
            args.key = entry.id;
            memset(args.value.str, 0, MAX_VALUE_LEN);
            strncpy(args.value.str, entry.val, MAX_VALUE_LEN - 1);
            args.root_idx = root_idx;
            args.next_free = next_free;

            TaskLauncher launcher(INSERT_TASK_ID,
                                  TaskArgument(&args, sizeof(args)));
            launcher.add_region_requirement(
                RegionRequirement(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr));
            launcher.region_requirements[0].add_field(FID_KEY);
            launcher.region_requirements[0].add_field(FID_VALUE);
            launcher.region_requirements[0].add_field(FID_LEFT);
            launcher.region_requirements[0].add_field(FID_RIGHT);

            Future f = runtime->execute_task(ctx, launcher);
            InsertResult res = f.get_result<InsertResult>();
            root_idx = res.root_idx;
            next_free = res.next_free;
        }
        outfile << "\n";

        // 2. Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            SearchArgs args;
            args.search_key = search_key;
            args.root_idx = root_idx;

            TaskLauncher launcher(SEARCH_TASK_ID,
                                  TaskArgument(&args, sizeof(args)));
            launcher.add_region_requirement(
                RegionRequirement(tree_lr, READ_ONLY, EXCLUSIVE, tree_lr));
            launcher.region_requirements[0].add_field(FID_KEY);
            launcher.region_requirements[0].add_field(FID_VALUE);
            launcher.region_requirements[0].add_field(FID_LEFT);
            launcher.region_requirements[0].add_field(FID_RIGHT);

            Future f = runtime->execute_task(ctx, launcher);
            SearchResult res = f.get_result<SearchResult>();

            if (res.found) {
                outfile << "Found! Result Value: \"" << res.value.str << "\"\n";
            } else {
                outfile << "Not Found.\n";
            }
        };

        perform_search(99);
        perform_search(10);
        perform_search(50);
        perform_search(20);
        perform_search(60);
        perform_search(80);
        perform_search(45);

        // Cleanup Legion resources
        runtime->destroy_logical_region(ctx, tree_lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    outfile.close();
}

//--------------------------------------------------------------------------
// Main: register tasks and start the Legion runtime
//--------------------------------------------------------------------------
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
        Runtime::preregister_task_variant<InsertResult, insert_task>(
            registrar, "insert");
    }
    {
        TaskVariantRegistrar registrar(SEARCH_TASK_ID, "search");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<SearchResult, search_task>(
            registrar, "search");
    }

    return Runtime::start(argc, argv);
}
