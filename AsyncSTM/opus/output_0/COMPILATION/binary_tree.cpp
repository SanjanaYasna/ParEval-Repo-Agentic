////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated from HPX/ASTM to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cassert>
#include "legion.h"

using namespace Legion;

// ---------------------------------------------------------------------------
// Task IDs
// ---------------------------------------------------------------------------
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INSERT_TASK_ID,
    SEARCH_TASK_ID,
};

// ---------------------------------------------------------------------------
// Field IDs
// ---------------------------------------------------------------------------
enum FieldIDs {
    // Tree-node fields (stored in the nodes region)
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
    // Metadata fields (stored in a one-element metadata region)
    FID_ROOT_IDX,
    FID_NUM_NODES,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int MAX_NODES     = 128;
static const int VALUE_MAX_LEN = 64;
static const int NULL_IDX      = -1;

// ---------------------------------------------------------------------------
// POD helpers (must be trivially copyable for Legion task arguments / fields)
// ---------------------------------------------------------------------------
struct ValueString {
    char data[VALUE_MAX_LEN];
};

struct InsertArgs {
    int  key;
    char value[VALUE_MAX_LEN];
};

struct SearchArgs {
    int key;
};

struct SearchResult {
    bool found;
    char value[VALUE_MAX_LEN];
};

// ---------------------------------------------------------------------------
// INSERT task
//   Regions[0] : tree nodes  (READ_WRITE) – FID_KEY, FID_VALUE, FID_LEFT, FID_RIGHT
//   Regions[1] : metadata    (READ_WRITE) – FID_ROOT_IDX, FID_NUM_NODES
// ---------------------------------------------------------------------------
void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(InsertArgs));
    InsertArgs args;
    memcpy(&args, task->args, sizeof(InsertArgs));

    // Accessors for tree-node fields
    const FieldAccessor<READ_WRITE, int, 1>         fa_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_WRITE, ValueString, 1>  fa_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1>         fa_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1>         fa_right(regions[0], FID_RIGHT);

    // Accessors for metadata
    const FieldAccessor<READ_WRITE, int, 1> fa_root(regions[1], FID_ROOT_IDX);
    const FieldAccessor<READ_WRITE, int, 1> fa_num (regions[1], FID_NUM_NODES);

    int root_idx  = fa_root[0];
    int num_nodes = fa_num[0];

    // Allocate a new node at index == num_nodes
    int new_idx = num_nodes;
    fa_key[new_idx] = args.key;
    {
        ValueString vs;
        memset(vs.data, 0, VALUE_MAX_LEN);
        strncpy(vs.data, args.value, VALUE_MAX_LEN - 1);
        fa_val[new_idx] = vs;
    }
    fa_left[new_idx]  = NULL_IDX;
    fa_right[new_idx] = NULL_IDX;
    num_nodes++;

    if (root_idx == NULL_IDX) {
        // Tree was empty – the new node becomes the root
        root_idx = new_idx;
    } else {
        // Walk down from the root to find the correct leaf position
        int current = root_idx;
        while (true) {
            int cur_key = fa_key[current];
            if (args.key < cur_key) {
                int left = fa_left[current];
                if (left == NULL_IDX) {
                    fa_left[current] = new_idx;
                    break;
                }
                current = left;
            } else {
                // key >= cur_key  (matches original code's >=  branch)
                int right = fa_right[current];
                if (right == NULL_IDX) {
                    fa_right[current] = new_idx;
                    break;
                }
                current = right;
            }
        }
    }

    // Write back metadata
    fa_root[0] = root_idx;
    fa_num[0]  = num_nodes;
}

// ---------------------------------------------------------------------------
// SEARCH task
//   Regions[0] : tree nodes  (READ_ONLY) – FID_KEY, FID_VALUE, FID_LEFT, FID_RIGHT
//   Regions[1] : metadata    (READ_ONLY) – FID_ROOT_IDX
// Returns: SearchResult
// ---------------------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(SearchArgs));
    SearchArgs args;
    memcpy(&args, task->args, sizeof(SearchArgs));

    SearchResult result;
    result.found = false;
    memset(result.value, 0, VALUE_MAX_LEN);

    const FieldAccessor<READ_ONLY, int, 1>         fa_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_ONLY, ValueString, 1>  fa_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_ONLY, int, 1>         fa_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_ONLY, int, 1>         fa_right(regions[0], FID_RIGHT);

    const FieldAccessor<READ_ONLY, int, 1> fa_root(regions[1], FID_ROOT_IDX);

    int current = fa_root[0];

    while (current != NULL_IDX) {
        int cur_key = fa_key[current];
        if (args.key == cur_key) {
            result.found = true;
            ValueString vs = fa_val[current];
            strncpy(result.value, vs.data, VALUE_MAX_LEN);
            result.value[VALUE_MAX_LEN - 1] = '\0';
            return result;
        } else if (args.key < cur_key) {
            current = fa_left[current];
        } else {
            current = fa_right[current];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// TOP-LEVEL task – orchestrates the whole program
// ---------------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Create logical region for tree nodes
    // ------------------------------------------------------------------
    Rect<1> node_rect(0, MAX_NODES - 1);
    IndexSpace node_is = runtime->create_index_space(ctx, node_rect);
    runtime->attach_name(node_is, "node_index_space");

    FieldSpace node_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator alloc = runtime->create_field_allocator(ctx, node_fs);
        alloc.allocate_field(sizeof(int),         FID_KEY);
        alloc.allocate_field(sizeof(ValueString), FID_VALUE);
        alloc.allocate_field(sizeof(int),         FID_LEFT);
        alloc.allocate_field(sizeof(int),         FID_RIGHT);
    }
    runtime->attach_name(node_fs, "node_field_space");

    LogicalRegion tree_lr = runtime->create_logical_region(ctx, node_is, node_fs);
    runtime->attach_name(tree_lr, "tree_region");

    // ------------------------------------------------------------------
    // Create logical region for metadata (root_idx, num_nodes)
    // ------------------------------------------------------------------
    Rect<1> meta_rect(0, 0);
    IndexSpace meta_is = runtime->create_index_space(ctx, meta_rect);
    runtime->attach_name(meta_is, "meta_index_space");

    FieldSpace meta_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator alloc = runtime->create_field_allocator(ctx, meta_fs);
        alloc.allocate_field(sizeof(int), FID_ROOT_IDX);
        alloc.allocate_field(sizeof(int), FID_NUM_NODES);
    }
    runtime->attach_name(meta_fs, "meta_field_space");

    LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);
    runtime->attach_name(meta_lr, "meta_region");

    // ------------------------------------------------------------------
    // Initialise metadata (empty tree)
    // ------------------------------------------------------------------
    {
        RegionRequirement req(meta_lr, WRITE_DISCARD, EXCLUSIVE, meta_lr);
        req.add_field(FID_ROOT_IDX);
        req.add_field(FID_NUM_NODES);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, launcher);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1> fa_root(pr, FID_ROOT_IDX);
        const FieldAccessor<WRITE_DISCARD, int, 1> fa_num (pr, FID_NUM_NODES);
        fa_root[0] = NULL_IDX;
        fa_num[0]  = 0;

        runtime->unmap_region(ctx, pr);
    }

    // ------------------------------------------------------------------
    // 1.  Data Insertion Phase
    // ------------------------------------------------------------------
    struct DataEntry { int id; const char* val; };

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
        args.key = data[i].id;
        memset(args.value, 0, VALUE_MAX_LEN);
        strncpy(args.value, data[i].val, VALUE_MAX_LEN - 1);

        TaskLauncher launcher(INSERT_TASK_ID,
                              TaskArgument(&args, sizeof(InsertArgs)));

        // Tree-node region: READ_WRITE
        RegionRequirement tree_req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
        tree_req.add_field(FID_KEY);
        tree_req.add_field(FID_VALUE);
        tree_req.add_field(FID_LEFT);
        tree_req.add_field(FID_RIGHT);
        launcher.add_region_requirement(tree_req);

        // Metadata region: READ_WRITE
        RegionRequirement meta_req(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr);
        meta_req.add_field(FID_ROOT_IDX);
        meta_req.add_field(FID_NUM_NODES);
        launcher.add_region_requirement(meta_req);

        // Legion serialises these tasks because they all request READ_WRITE
        // on the same regions – equivalent to the STM commit ordering.
        runtime->execute_task(ctx, launcher);
    }
    outfile << "\n";

    // ------------------------------------------------------------------
    // 2.  Search Test Phase
    // ------------------------------------------------------------------
    outfile << "Running Search Tests:\n";

    int search_keys[] = {99, 10, 50, 20, 60, 80, 45};
    const int num_searches = sizeof(search_keys) / sizeof(search_keys[0]);

    // Launch all search tasks (READ_ONLY – may execute in parallel)
    std::vector<Future> search_futures;
    for (int i = 0; i < num_searches; i++) {
        SearchArgs args;
        args.key = search_keys[i];

        TaskLauncher launcher(SEARCH_TASK_ID,
                              TaskArgument(&args, sizeof(SearchArgs)));

        RegionRequirement tree_req(tree_lr, READ_ONLY, EXCLUSIVE, tree_lr);
        tree_req.add_field(FID_KEY);
        tree_req.add_field(FID_VALUE);
        tree_req.add_field(FID_LEFT);
        tree_req.add_field(FID_RIGHT);
        launcher.add_region_requirement(tree_req);

        RegionRequirement meta_req(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr);
        meta_req.add_field(FID_ROOT_IDX);
        launcher.add_region_requirement(meta_req);

        search_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Collect results in launch order to match the original output
    for (int i = 0; i < num_searches; i++) {
        SearchResult result = search_futures[i].get_result<SearchResult>();
        outfile << "  Search(Key: " << search_keys[i] << ") -> ";
        if (result.found) {
            outfile << "Found! Result Value: \"" << result.value << "\"\n";
        } else {
            outfile << "Not Found.\n";
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_logical_region(ctx, meta_lr);
    runtime->destroy_field_space(ctx, node_fs);
    runtime->destroy_field_space(ctx, meta_fs);
    runtime->destroy_index_space(ctx, node_is);
    runtime->destroy_index_space(ctx, meta_is);

    outfile.close();
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
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
        Runtime::preregister_task_variant<insert_task>(registrar, "insert");
    }
    {
        TaskVariantRegistrar registrar(SEARCH_TASK_ID, "search");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<SearchResult, search_task>(registrar, "search");
    }

    return Runtime::start(argc, argv);
}
