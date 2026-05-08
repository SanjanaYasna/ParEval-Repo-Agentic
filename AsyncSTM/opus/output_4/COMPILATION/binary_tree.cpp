////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

using namespace Legion;

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------
static const int MAX_NODES   = 128;
static const int MAX_STR_LEN = 64;

// ---------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INSERT_TASK_ID,
    SEARCH_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
    FID_ROOT_IDX,
    FID_NODE_COUNT,
};

// ---------------------------------------------------------------
// Helper types (must be trivially copyable for Legion arguments/futures)
// ---------------------------------------------------------------
struct StrField {
    char data[MAX_STR_LEN];
};

struct InsertArgs {
    int key;
    StrField value;
};

struct SearchResult {
    bool found;
    StrField value;
};

// ---------------------------------------------------------------
// Recursive BST insert (operates on accessors)
// ---------------------------------------------------------------
static void bst_insert_recursive(
    int key, int node_idx, int new_idx,
    const FieldAccessor<READ_WRITE, int, 1> &acc_key,
    const FieldAccessor<READ_WRITE, int, 1> &acc_left,
    const FieldAccessor<READ_WRITE, int, 1> &acc_right)
{
    int nk = acc_key[node_idx];
    if (key < nk) {
        int l = acc_left[node_idx];
        if (l != -1)
            bst_insert_recursive(key, l, new_idx, acc_key, acc_left, acc_right);
        else
            acc_left[node_idx] = new_idx;
    } else {
        int r = acc_right[node_idx];
        if (r != -1)
            bst_insert_recursive(key, r, new_idx, acc_key, acc_left, acc_right);
        else
            acc_right[node_idx] = new_idx;
    }
}

// ---------------------------------------------------------------
// Insert task  –  corresponds to an ASTM transaction with writes
//   regions[0] = node_lr  (READ_WRITE)
//   regions[1] = meta_lr  (READ_WRITE)
// ---------------------------------------------------------------
void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    InsertArgs args;
    memcpy(&args, task->args, sizeof(InsertArgs));

    const FieldAccessor<READ_WRITE, int, 1>      acc_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_WRITE, StrField, 1>  acc_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1>       acc_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1>       acc_right(regions[0], FID_RIGHT);

    const FieldAccessor<READ_WRITE, int, 1> acc_root (regions[1], FID_ROOT_IDX);
    const FieldAccessor<READ_WRITE, int, 1> acc_count(regions[1], FID_NODE_COUNT);

    int root_idx = acc_root[0];
    int count    = acc_count[0];
    int new_idx  = count;

    // Initialise the new node
    acc_key  [new_idx] = args.key;
    acc_val  [new_idx] = args.value;
    acc_left [new_idx] = -1;
    acc_right[new_idx] = -1;

    if (root_idx == -1) {
        // Tree was empty – new node becomes root
        acc_root[0] = new_idx;
    } else {
        bst_insert_recursive(args.key, root_idx, new_idx,
                             acc_key, acc_left, acc_right);
    }

    acc_count[0] = count + 1;
}

// ---------------------------------------------------------------
// Recursive BST search (operates on read-only accessors)
// ---------------------------------------------------------------
static int bst_search_recursive(
    int key, int node_idx,
    const FieldAccessor<READ_ONLY, int, 1> &acc_key,
    const FieldAccessor<READ_ONLY, int, 1> &acc_left,
    const FieldAccessor<READ_ONLY, int, 1> &acc_right)
{
    if (node_idx == -1) return -1;
    int nk = acc_key[node_idx];
    if (key == nk) return node_idx;
    if (key < nk)
        return bst_search_recursive(key, acc_left[node_idx],
                                    acc_key, acc_left, acc_right);
    return bst_search_recursive(key, acc_right[node_idx],
                                acc_key, acc_left, acc_right);
}

// ---------------------------------------------------------------
// Search task  –  corresponds to an ASTM read-only transaction
//   regions[0] = node_lr  (READ_ONLY)
//   regions[1] = meta_lr  (READ_ONLY)
// ---------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    int search_key;
    memcpy(&search_key, task->args, sizeof(int));

    const FieldAccessor<READ_ONLY, int, 1>      acc_key  (regions[0], FID_KEY);
    const FieldAccessor<READ_ONLY, StrField, 1>  acc_val  (regions[0], FID_VALUE);
    const FieldAccessor<READ_ONLY, int, 1>       acc_left (regions[0], FID_LEFT);
    const FieldAccessor<READ_ONLY, int, 1>       acc_right(regions[0], FID_RIGHT);

    const FieldAccessor<READ_ONLY, int, 1> acc_root(regions[1], FID_ROOT_IDX);

    int root_idx = acc_root[0];

    SearchResult result;
    memset(&result, 0, sizeof(result));
    result.found = false;

    int found_idx = bst_search_recursive(search_key, root_idx,
                                         acc_key, acc_left, acc_right);
    if (found_idx != -1) {
        result.found = true;
        result.value = acc_val[found_idx];
    }

    return result;
}

// ---------------------------------------------------------------
// Top-level task  –  orchestrates the whole programme
// ---------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ----- create the node region (array-based BST storage) -----
    Rect<1> node_rect(0, MAX_NODES - 1);
    IndexSpace node_is = runtime->create_index_space(ctx, node_rect);
    runtime->attach_name(node_is, "node_is");

    FieldSpace node_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, node_fs);
        fa.allocate_field(sizeof(int),      FID_KEY);
        fa.allocate_field(sizeof(StrField), FID_VALUE);
        fa.allocate_field(sizeof(int),      FID_LEFT);
        fa.allocate_field(sizeof(int),      FID_RIGHT);
    }
    LogicalRegion node_lr = runtime->create_logical_region(ctx, node_is, node_fs);

    // ----- create the metadata region (root index + count) -----
    Rect<1> meta_rect(0, 0);
    IndexSpace meta_is = runtime->create_index_space(ctx, meta_rect);
    runtime->attach_name(meta_is, "meta_is");

    FieldSpace meta_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, meta_fs);
        fa.allocate_field(sizeof(int), FID_ROOT_IDX);
        fa.allocate_field(sizeof(int), FID_NODE_COUNT);
    }
    LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

    // ----- initialise node region -----
    {
        InlineLauncher il(RegionRequirement(node_lr, WRITE_DISCARD,
                                            EXCLUSIVE, node_lr));
        il.requirement.add_field(FID_KEY);
        il.requirement.add_field(FID_VALUE);
        il.requirement.add_field(FID_LEFT);
        il.requirement.add_field(FID_RIGHT);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1>      a_key  (pr, FID_KEY);
        const FieldAccessor<WRITE_DISCARD, StrField, 1>  a_val  (pr, FID_VALUE);
        const FieldAccessor<WRITE_DISCARD, int, 1>       a_left (pr, FID_LEFT);
        const FieldAccessor<WRITE_DISCARD, int, 1>       a_right(pr, FID_RIGHT);

        StrField empty;
        memset(&empty, 0, sizeof(empty));
        for (int i = 0; i < MAX_NODES; i++) {
            a_key[i]   = 0;
            a_val[i]   = empty;
            a_left[i]  = -1;
            a_right[i] = -1;
        }
        runtime->unmap_region(ctx, pr);
    }

    // ----- initialise metadata region -----
    {
        InlineLauncher il(RegionRequirement(meta_lr, WRITE_DISCARD,
                                            EXCLUSIVE, meta_lr));
        il.requirement.add_field(FID_ROOT_IDX);
        il.requirement.add_field(FID_NODE_COUNT);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1> a_root (pr, FID_ROOT_IDX);
        const FieldAccessor<WRITE_DISCARD, int, 1> a_count(pr, FID_NODE_COUNT);
        a_root[0]  = -1;   // empty tree
        a_count[0] = 0;
        runtime->unmap_region(ctx, pr);
    }

    // ----- open output file -----
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        runtime->destroy_logical_region(ctx, node_lr);
        runtime->destroy_logical_region(ctx, meta_lr);
        runtime->destroy_field_space(ctx, node_fs);
        runtime->destroy_field_space(ctx, meta_fs);
        runtime->destroy_index_space(ctx, node_is);
        runtime->destroy_index_space(ctx, meta_is);
        return;
    }

    // =====================================================================
    // 1.  Data Insertion Phase
    //     Each insert is a separate Legion task (≈ ASTM transaction).
    //     READ_WRITE requirements on both regions serialise the inserts
    //     automatically through Legion's dependence analysis.
    // =====================================================================
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
    const int num_inserts = 7;

    outfile << "Inserting Data:\n";
    for (int i = 0; i < num_inserts; i++) {
        outfile << "  Insert(Key: " << data[i].id
                << ", Value: \"" << data[i].val << "\")" << std::endl;

        InsertArgs args;
        args.key = data[i].id;
        memset(&args.value, 0, sizeof(StrField));
        strncpy(args.value.data, data[i].val, MAX_STR_LEN - 1);

        TaskLauncher launcher(INSERT_TASK_ID,
                              TaskArgument(&args, sizeof(InsertArgs)));
        launcher.add_region_requirement(
            RegionRequirement(node_lr, READ_WRITE, EXCLUSIVE, node_lr));
        launcher.region_requirements[0].add_field(FID_KEY);
        launcher.region_requirements[0].add_field(FID_VALUE);
        launcher.region_requirements[0].add_field(FID_LEFT);
        launcher.region_requirements[0].add_field(FID_RIGHT);

        launcher.add_region_requirement(
            RegionRequirement(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr));
        launcher.region_requirements[1].add_field(FID_ROOT_IDX);
        launcher.region_requirements[1].add_field(FID_NODE_COUNT);

        // Launch – Legion serialises inserts via the RW dependence chain
        runtime->execute_task(ctx, launcher);
    }
    outfile << "\n";

    // =====================================================================
    // 2.  Search Test Phase
    //     Each search is a READ_ONLY task (≈ ASTM read-only transaction).
    //     All searches depend on the last insert (RW→RO) but may run in
    //     parallel with each other.  Futures are collected in launch order
    //     so the output is deterministic.
    // =====================================================================
    outfile << "Running Search Tests:\n";

    int search_keys[] = {99, 10, 50, 20, 60, 80, 45};
    const int num_searches = 7;
    std::vector<Future> search_futures;

    for (int i = 0; i < num_searches; i++) {
        int key = search_keys[i];

        TaskLauncher launcher(SEARCH_TASK_ID,
                              TaskArgument(&key, sizeof(int)));
        launcher.add_region_requirement(
            RegionRequirement(node_lr, READ_ONLY, EXCLUSIVE, node_lr));
        launcher.region_requirements[0].add_field(FID_KEY);
        launcher.region_requirements[0].add_field(FID_VALUE);
        launcher.region_requirements[0].add_field(FID_LEFT);
        launcher.region_requirements[0].add_field(FID_RIGHT);

        launcher.add_region_requirement(
            RegionRequirement(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr));
        launcher.region_requirements[1].add_field(FID_ROOT_IDX);

        search_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Collect results in launch order (preserves original output ordering)
    for (int i = 0; i < num_searches; i++) {
        SearchResult result = search_futures[i].get_result<SearchResult>();
        outfile << "  Search(Key: " << search_keys[i] << ") -> ";
        if (result.found) {
            outfile << "Found! Result Value: \"" << result.value.data << "\"\n";
        } else {
            outfile << "Not Found.\n";
        }
    }

    outfile.close();

    // ----- cleanup -----
    runtime->destroy_logical_region(ctx, node_lr);
    runtime->destroy_logical_region(ctx, meta_lr);
    runtime->destroy_field_space(ctx, node_fs);
    runtime->destroy_field_space(ctx, meta_fs);
    runtime->destroy_index_space(ctx, node_is);
    runtime->destroy_index_space(ctx, meta_is);
}

// ---------------------------------------------------------------
// Main – register tasks and start the Legion runtime
// ---------------------------------------------------------------
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
        Runtime::preregister_task_variant<SearchResult, search_task>(
            registrar, "search");
    }

    return Runtime::start(argc, argv);
}
