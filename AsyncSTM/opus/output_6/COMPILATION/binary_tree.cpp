////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstring>
#include <functional>
#include <vector>
#include "legion.h"

using namespace Legion;

enum TaskID {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int NULL_IDX = -1;
static const int MAX_VALUE_LEN = 64;
static const int MAX_NODES = 128;

struct NodeValue {
    char data[MAX_VALUE_LEN];
};

// Binary tree backed by a Legion region.
// Nodes are stored in a flat region; integer indices replace pointers.
// The STM transaction / commit_transaction retry loop is replaced by
// Legion's coherence guarantees on the inline-mapped region.
template <typename Key>
struct binary_tree
{
    const FieldAccessor<READ_WRITE, Key, 1>       key_acc;
    const FieldAccessor<READ_WRITE, NodeValue, 1>  value_acc;
    const FieldAccessor<READ_WRITE, int, 1>        left_acc;
    const FieldAccessor<READ_WRITE, int, 1>        right_acc;

    int root_idx;
    int next_free;

    binary_tree(PhysicalRegion pr)
      : key_acc(pr, FID_KEY),
        value_acc(pr, FID_VALUE),
        left_acc(pr, FID_LEFT),
        right_acc(pr, FID_RIGHT),
        root_idx(NULL_IDX),
        next_free(0)
    {
        // Initialise every slot to empty
        for (int i = 0; i < MAX_NODES; i++) {
            key_acc[i]   = Key();
            NodeValue nv;
            memset(nv.data, 0, MAX_VALUE_LEN);
            value_acc[i] = nv;
            left_acc[i]  = NULL_IDX;
            right_acc[i] = NULL_IDX;
        }
    }

    // Allocate a new node in the region, returning its index
    int allocate_node(Key k, const std::string& v)
    {
        int idx = next_free++;
        key_acc[idx] = k;
        NodeValue nv;
        memset(nv.data, 0, MAX_VALUE_LEN);
        strncpy(nv.data, v.c_str(), MAX_VALUE_LEN - 1);
        value_acc[idx] = nv;
        left_acc[idx]  = NULL_IDX;
        right_acc[idx] = NULL_IDX;
        return idx;
    }

    // Public insert – mirrors the original STM transaction that
    // reads root, then either sets it or recurses.
    void insert(Key key, const std::string& value)
    {
        if (root_idx != NULL_IDX)
            insert(key, value, root_idx);
        else
            root_idx = allocate_node(key, value);
    }

    // Public search – returns node index (NULL_IDX if not found)
    int search(Key key)
    {
        return search(key, root_idx);
    }

    // Retrieve the stored string value for a node index
    std::string get_value(int node_idx) const
    {
        NodeValue nv = value_acc[node_idx];
        return std::string(nv.data);
    }

private:
    void insert(Key key, const std::string& value, int leaf_idx)
    {
        Key leaf_key = key_acc[leaf_idx];

        if (key < leaf_key) {
            int left = left_acc[leaf_idx];
            if (left != NULL_IDX)
                insert(key, value, left);
            else
                left_acc[leaf_idx] = allocate_node(key, value);
        } else {                            // key >= leaf_key
            int right = right_acc[leaf_idx];
            if (right != NULL_IDX)
                insert(key, value, right);
            else
                right_acc[leaf_idx] = allocate_node(key, value);
        }
    }

    int search(Key key, int leaf_idx)
    {
        if (leaf_idx == NULL_IDX)
            return NULL_IDX;

        Key leaf_key = key_acc[leaf_idx];

        if (key == leaf_key)
            return leaf_idx;
        if (key < leaf_key)
            return search(key, left_acc[leaf_idx]);
        else
            return search(key, right_acc[leaf_idx]);
    }
};

// ----------------------------------------------------------------
// Top-level Legion task – replaces the original main / hpx_main
// ----------------------------------------------------------------
void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    {
        // Create index space and field space for tree nodes
        Rect<1> rect(0, MAX_NODES - 1);
        IndexSpace is = runtime->create_index_space(ctx, rect);
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(int),       FID_KEY);
            fa.allocate_field(sizeof(NodeValue),  FID_VALUE);
            fa.allocate_field(sizeof(int),       FID_LEFT);
            fa.allocate_field(sizeof(int),       FID_RIGHT);
        }
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Inline-map the region with READ_WRITE / EXCLUSIVE privilege.
        // This gives the top-level task exclusive coherent access –
        // the Legion equivalent of the STM transactional guarantees.
        InlineLauncher il(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        il.add_field(FID_KEY);
        il.add_field(FID_VALUE);
        il.add_field(FID_LEFT);
        il.add_field(FID_RIGHT);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        // Construct tree backed by the physical region
        binary_tree<int> tree(pr);

        // 1. Data Insertion Phase
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

        outfile << "Inserting Data:\n";
        for (const auto& entry : data) {
            outfile << "  Insert(Key: " << entry.id
                    << ", Value: \"" << entry.val << "\")" << std::endl;
            tree.insert(entry.id, entry.val);
        }
        outfile << "\n";

        // 2. Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            int idx = tree.search(search_key);

            if (idx != NULL_IDX) {
                outfile << "Found! Result Value: \""
                        << tree.get_value(idx) << "\"\n";
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

        // Clean up Legion resources
        runtime->unmap_region(ctx, pr);
        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    outfile.close();
}

// ----------------------------------------------------------------
// main – register the top-level task and start the Legion runtime
// ----------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar,
                                                          "top_level_task");
    }

    return Runtime::start(argc, argv);
}
