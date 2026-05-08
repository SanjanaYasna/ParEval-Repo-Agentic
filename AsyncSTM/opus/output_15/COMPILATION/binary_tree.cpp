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
#include "legion.h"

using namespace Legion;

// ---------------------------------------------------------------------------
// Constants and enumerations
// ---------------------------------------------------------------------------
enum {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int    NIL_NODE    = -1;
static const size_t MAX_VAL_LEN = 64;
static const size_t MAX_NODES   = 128;

// Fixed-size value type that can live inside a Legion region field.
struct NodeValue {
    char str[MAX_VAL_LEN];
};

// ---------------------------------------------------------------------------
// BinaryTree – operates directly on Legion field accessors
// ---------------------------------------------------------------------------
template <typename Key, typename Value>
class BinaryTree {
public:
    BinaryTree(const FieldAccessor<READ_WRITE, Key, 1>&       key_acc,
               const FieldAccessor<READ_WRITE, NodeValue, 1>& val_acc,
               const FieldAccessor<READ_WRITE, int, 1>&       left_acc,
               const FieldAccessor<READ_WRITE, int, 1>&       right_acc)
        : key_acc_(key_acc), val_acc_(val_acc),
          left_acc_(left_acc), right_acc_(right_acc),
          root_(NIL_NODE), node_count_(0)
    {}

    // ------------------------------------------------------------------
    // Insert – mirrors the original transactional insert logic
    // ------------------------------------------------------------------
    void insert(Key key, const Value& value)
    {
        if (root_ == NIL_NODE) {
            root_ = allocate_node(key, value);
        } else {
            insert_at(key, value, root_);
        }
    }

    // ------------------------------------------------------------------
    // Search – returns the region index of the found node, or NIL_NODE
    // ------------------------------------------------------------------
    int search(Key key)
    {
        return search_at(key, root_);
    }

    // Accessors for the caller to retrieve data from a found node index
    Key get_key(int idx)
    {
        return key_acc_[idx];
    }

    std::string get_value(int idx)
    {
        NodeValue nv = val_acc_[idx];
        return std::string(nv.str);
    }

private:
    // Allocate a new node in the next free slot of the region
    int allocate_node(Key key, const Value& value)
    {
        int idx = node_count_++;
        key_acc_[idx] = key;

        NodeValue nv;
        std::memset(nv.str, 0, MAX_VAL_LEN);
        std::strncpy(nv.str, value.c_str(), MAX_VAL_LEN - 1);
        val_acc_[idx] = nv;

        left_acc_[idx]  = NIL_NODE;
        right_acc_[idx] = NIL_NODE;
        return idx;
    }

    // Recursive insert – same traversal logic as the original:
    //   key <  current  →  go left
    //   key >= current  →  go right
    void insert_at(Key key, const Value& value, int node_idx)
    {
        Key node_key = key_acc_[node_idx];

        if (key < node_key) {
            int left = left_acc_[node_idx];
            if (left != NIL_NODE)
                insert_at(key, value, left);
            else
                left_acc_[node_idx] = allocate_node(key, value);
        } else {  // key >= node_key
            int right = right_acc_[node_idx];
            if (right != NIL_NODE)
                insert_at(key, value, right);
            else
                right_acc_[node_idx] = allocate_node(key, value);
        }
    }

    // Recursive search
    int search_at(Key key, int node_idx)
    {
        if (node_idx == NIL_NODE)
            return NIL_NODE;

        Key node_key = key_acc_[node_idx];
        if (key == node_key)
            return node_idx;
        if (key < node_key)
            return search_at(key, (int)left_acc_[node_idx]);
        else
            return search_at(key, (int)right_acc_[node_idx]);
    }

    FieldAccessor<READ_WRITE, Key, 1>       key_acc_;
    FieldAccessor<READ_WRITE, NodeValue, 1>  val_acc_;
    FieldAccessor<READ_WRITE, int, 1>        left_acc_;
    FieldAccessor<READ_WRITE, int, 1>        right_acc_;
    int root_;
    int node_count_;
};

// ---------------------------------------------------------------------------
// Top-level Legion task – replaces the original main()
// ---------------------------------------------------------------------------
void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    // Create a logical region to hold tree nodes
    IndexSpace is = runtime->create_index_space(ctx,
                        Rect<1>(0, static_cast<coord_t>(MAX_NODES - 1)));
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

    // Inline-map the region with READ_WRITE/EXCLUSIVE
    RegionRequirement req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);

    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    {
        // Build typed accessors over the mapped region
        const FieldAccessor<READ_WRITE, int, 1>       key_acc(pr,  FID_KEY);
        const FieldAccessor<READ_WRITE, NodeValue, 1>  val_acc(pr,  FID_VALUE);
        const FieldAccessor<READ_WRITE, int, 1>        left_acc(pr, FID_LEFT);
        const FieldAccessor<READ_WRITE, int, 1>        right_acc(pr,FID_RIGHT);

        BinaryTree<int, std::string> tree(key_acc, val_acc, left_acc, right_acc);

        // -------------------------------------------------------
        // 1. Data Insertion Phase
        // -------------------------------------------------------
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

        // -------------------------------------------------------
        // 2. Search Test Phase
        // -------------------------------------------------------
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            int idx = tree.search(search_key);

            if (idx != NIL_NODE) {
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
    }

    outfile.close();

    // Clean up Legion resources
    runtime->unmap_region(ctx, pr);
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    return Runtime::start(argc, argv);
}
