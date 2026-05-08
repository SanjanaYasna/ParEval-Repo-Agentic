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
#include <string>
#include <vector>
#include <cstring>
#include "legion.h"

using namespace Legion;

// Constants
static const int NIL_NODE = -1;
static const size_t MAX_VALUE_LEN = 64;
static const size_t MAX_NODES = 128;

// Field IDs
enum {
    FID_KEY = 100,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
    FID_META,
};

// Task IDs
enum {
    TOP_LEVEL_TASK_ID = 0,
    INSERT_TASK_ID,
    SEARCH_TASK_ID,
};

// Fixed-size string type suitable for storage in Legion regions and task arguments
struct FixedString {
    char data[MAX_VALUE_LEN];
    FixedString() { memset(data, 0, MAX_VALUE_LEN); }
    FixedString(const std::string& s) {
        memset(data, 0, MAX_VALUE_LEN);
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }
    std::string str() const { return std::string(data); }
};

// Task argument for insert operations
struct InsertArgs {
    int key;
    FixedString value;
};

// Task return type for search operations
struct SearchResult {
    bool found;
    int key;
    FixedString value;
};

// ---- Tree helper functions operating on region accessors ----

// Allocate a new node in the node pool region, returning its index.
// meta_acc[0] = root_idx, meta_acc[1] = next_free index
static int alloc_node(
    const FieldAccessor<READ_WRITE, int, 1>& meta_acc,
    const FieldAccessor<READ_WRITE, int, 1>& key_acc,
    const FieldAccessor<READ_WRITE, FixedString, 1>& val_acc,
    const FieldAccessor<READ_WRITE, int, 1>& left_acc,
    const FieldAccessor<READ_WRITE, int, 1>& right_acc,
    int key, const FixedString& value)
{
    int idx = meta_acc[1]; // next_free
    meta_acc[1] = idx + 1;
    key_acc[idx] = key;
    val_acc[idx] = value;
    left_acc[idx] = NIL_NODE;
    right_acc[idx] = NIL_NODE;
    return idx;
}

// Recursively insert into the subtree rooted at node_idx
static void insert_at(
    int node_idx, int key, const FixedString& value,
    const FieldAccessor<READ_WRITE, int, 1>& meta_acc,
    const FieldAccessor<READ_WRITE, int, 1>& key_acc,
    const FieldAccessor<READ_WRITE, FixedString, 1>& val_acc,
    const FieldAccessor<READ_WRITE, int, 1>& left_acc,
    const FieldAccessor<READ_WRITE, int, 1>& right_acc)
{
    int node_key = key_acc[node_idx];
    if (key < node_key) {
        int left = left_acc[node_idx];
        if (left != NIL_NODE)
            insert_at(left, key, value, meta_acc, key_acc, val_acc, left_acc, right_acc);
        else
            left_acc[node_idx] = alloc_node(meta_acc, key_acc, val_acc, left_acc, right_acc, key, value);
    } else {
        // key >= node_key: go right (matches original code's else-if branch)
        int right = right_acc[node_idx];
        if (right != NIL_NODE)
            insert_at(right, key, value, meta_acc, key_acc, val_acc, left_acc, right_acc);
        else
            right_acc[node_idx] = alloc_node(meta_acc, key_acc, val_acc, left_acc, right_acc, key, value);
    }
}

// Recursively search the subtree rooted at node_idx; returns node index or NIL_NODE
static int search_node(
    int node_idx, int key,
    const FieldAccessor<READ_ONLY, int, 1>& key_acc,
    const FieldAccessor<READ_ONLY, int, 1>& left_acc,
    const FieldAccessor<READ_ONLY, int, 1>& right_acc)
{
    if (node_idx == NIL_NODE) return NIL_NODE;
    int node_key = key_acc[node_idx];
    if (key == node_key) return node_idx;
    if (key < node_key)
        return search_node(left_acc[node_idx], key, key_acc, left_acc, right_acc);
    return search_node(right_acc[node_idx], key, key_acc, left_acc, right_acc);
}

// ---- Legion task implementations ----

// Insert task: corresponds to one ASTM transaction that inserts a key-value pair
void insert_task(const Task *task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx, Runtime *runtime)
{
    const InsertArgs& args = *reinterpret_cast<const InsertArgs*>(task->args);

    const FieldAccessor<READ_WRITE, int, 1> meta_acc(regions[0], FID_META);
    const FieldAccessor<READ_WRITE, int, 1> key_acc(regions[1], FID_KEY);
    const FieldAccessor<READ_WRITE, FixedString, 1> val_acc(regions[1], FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1> left_acc(regions[1], FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1> right_acc(regions[1], FID_RIGHT);

    int root_idx = meta_acc[0];

    if (root_idx == NIL_NODE) {
        int new_root = alloc_node(meta_acc, key_acc, val_acc, left_acc, right_acc,
                                  args.key, args.value);
        meta_acc[0] = new_root;
    } else {
        insert_at(root_idx, args.key, args.value,
                  meta_acc, key_acc, val_acc, left_acc, right_acc);
    }
}

// Search task: corresponds to one ASTM transaction that searches for a key
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime *runtime)
{
    int search_key = *reinterpret_cast<const int*>(task->args);

    const FieldAccessor<READ_ONLY, int, 1> meta_acc(regions[0], FID_META);
    const FieldAccessor<READ_ONLY, int, 1> key_acc(regions[1], FID_KEY);
    const FieldAccessor<READ_ONLY, FixedString, 1> val_acc(regions[1], FID_VALUE);
    const FieldAccessor<READ_ONLY, int, 1> left_acc(regions[1], FID_LEFT);
    const FieldAccessor<READ_ONLY, int, 1> right_acc(regions[1], FID_RIGHT);

    int root_idx = meta_acc[0];
    int found_idx = search_node(root_idx, search_key, key_acc, left_acc, right_acc);

    SearchResult result;
    result.key = search_key;
    if (found_idx != NIL_NODE) {
        result.found = true;
        result.value = val_acc[found_idx];
    } else {
        result.found = false;
        result.value = FixedString();
    }
    return result;
}

// ---- Top-level task ----

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime *runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    // Create metadata region (2 entries: [0]=root_idx, [1]=next_free)
    IndexSpace meta_is = runtime->create_index_space(ctx, Rect<1>(0, 1));
    FieldSpace meta_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, meta_fs);
        fa.allocate_field(sizeof(int), FID_META);
    }
    LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

    // Create node pool region (MAX_NODES entries)
    IndexSpace node_is = runtime->create_index_space(ctx,
                            Rect<1>(0, static_cast<coord_t>(MAX_NODES - 1)));
    FieldSpace node_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, node_fs);
        fa.allocate_field(sizeof(int), FID_KEY);
        fa.allocate_field(sizeof(FixedString), FID_VALUE);
        fa.allocate_field(sizeof(int), FID_LEFT);
        fa.allocate_field(sizeof(int), FID_RIGHT);
    }
    LogicalRegion node_lr = runtime->create_logical_region(ctx, node_is, node_fs);

    // Initialize metadata: root_idx = NIL_NODE (-1), next_free = 0
    {
        InlineLauncher il(RegionRequirement(meta_lr, WRITE_DISCARD, EXCLUSIVE, meta_lr));
        il.add_field(FID_META);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_META);
        acc[0] = NIL_NODE;  // root_idx: empty tree
        acc[1] = 0;         // next_free: first available slot
        runtime->unmap_region(ctx, pr);
    }

    // Data to insert (matches original program)
    struct DataEntry {
        int id;
        std::string val;
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

    // 1. Data Insertion Phase
    // Each insert is launched as a child task with READ_WRITE on both regions.
    // Legion serializes these in launch order due to conflicting region requirements,
    // which mirrors the ASTM transaction semantics.
    outfile << "Inserting Data:\n";
    for (const auto& entry : data) {
        outfile << "  Insert(Key: " << entry.id
                << ", Value: \"" << entry.val << "\")" << std::endl;

        InsertArgs args;
        args.key = entry.id;
        args.value = FixedString(entry.val);

        TaskLauncher launcher(INSERT_TASK_ID, TaskArgument(&args, sizeof(args)));
        launcher.add_region_requirement(
            RegionRequirement(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr));
        launcher.add_field(0, FID_META);
        launcher.add_region_requirement(
            RegionRequirement(node_lr, READ_WRITE, EXCLUSIVE, node_lr));
        launcher.add_field(1, FID_KEY);
        launcher.add_field(1, FID_VALUE);
        launcher.add_field(1, FID_LEFT);
        launcher.add_field(1, FID_RIGHT);

        // Launch (non-blocking); Legion orders tasks by region dependencies
        runtime->execute_task(ctx, launcher);
    }
    outfile << "\n";

    // 2. Search Test Phase
    // Each search is a child task with READ_ONLY on both regions.
    // Legion ensures all prior READ_WRITE insert tasks complete first.
    outfile << "Running Search Tests:\n";

    std::vector<int> search_keys = {99, 10, 50, 20, 60, 80, 45};

    for (int search_key : search_keys) {
        outfile << "  Search(Key: " << search_key << ") -> ";

        TaskLauncher launcher(SEARCH_TASK_ID,
                              TaskArgument(&search_key, sizeof(int)));
        launcher.add_region_requirement(
            RegionRequirement(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr));
        launcher.add_field(0, FID_META);
        launcher.add_region_requirement(
            RegionRequirement(node_lr, READ_ONLY, EXCLUSIVE, node_lr));
        launcher.add_field(1, FID_KEY);
        launcher.add_field(1, FID_VALUE);
        launcher.add_field(1, FID_LEFT);
        launcher.add_field(1, FID_RIGHT);

        Future f = runtime->execute_task(ctx, launcher);
        SearchResult result = f.get_result<SearchResult>();

        if (result.found) {
            outfile << "Found! Result Value: \"" << result.value.str() << "\"\n";
        } else {
            outfile << "Not Found.\n";
        }
    }

    outfile.close();

    // Cleanup
    runtime->destroy_logical_region(ctx, node_lr);
    runtime->destroy_logical_region(ctx, meta_lr);
    runtime->destroy_field_space(ctx, node_fs);
    runtime->destroy_field_space(ctx, meta_fs);
    runtime->destroy_index_space(ctx, node_is);
    runtime->destroy_index_space(ctx, meta_is);
}

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
