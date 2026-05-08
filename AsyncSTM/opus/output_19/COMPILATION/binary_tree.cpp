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
// Constants and enums
// ---------------------------------------------------------------------------
enum {
    TOP_LEVEL_TASK_ID = 0,
};

enum {
    FID_KEY = 0,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int NULL_IDX      = -1;
static const int MAX_NODES     = 64;
static const int MAX_VALUE_LEN = 64;

// ---------------------------------------------------------------------------
// Fixed-size string that can live inside a Legion field (POD type).
// ---------------------------------------------------------------------------
struct NodeValue {
    char data[MAX_VALUE_LEN];

    NodeValue() { memset(data, 0, sizeof(data)); }

    explicit NodeValue(const std::string& s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }

    std::string str() const { return std::string(data); }
};

// ---------------------------------------------------------------------------
// Search helper
// ---------------------------------------------------------------------------
struct SearchResult {
    bool        found;
    std::string value;
};

// ---------------------------------------------------------------------------
// Binary tree backed by a Legion LogicalRegion used as a node pool.
//
// Replaces the original ASTM shared_var / transaction design:
//   shared_var<node*>  →  int index into the node-pool region
//   transaction + get_local + commit_transaction
//       →  inline-mapped PhysicalRegion with READ_WRITE privilege
// ---------------------------------------------------------------------------
template <typename Key, typename Value>
class binary_tree {
public:
    binary_tree(Runtime* runtime, Context ctx)
        : runtime_(runtime), ctx_(ctx), next_node_(0), root_idx_(NULL_IDX)
    {
        // Create an index space large enough for all nodes we will ever need.
        Rect<1> bounds(0, MAX_NODES - 1);
        is_ = runtime_->create_index_space(ctx_, bounds);
        runtime_->attach_name(is_, "node_pool_is");

        // Create fields: key, value, left-child index, right-child index.
        fs_ = runtime_->create_field_space(ctx_);
        {
            FieldAllocator fa = runtime_->create_field_allocator(ctx_, fs_);
            fa.allocate_field(sizeof(Key),       FID_KEY);
            fa.allocate_field(sizeof(NodeValue),  FID_VALUE);
            fa.allocate_field(sizeof(int),        FID_LEFT);
            fa.allocate_field(sizeof(int),        FID_RIGHT);
        }
        runtime_->attach_name(fs_, "node_pool_fs");

        lr_ = runtime_->create_logical_region(ctx_, is_, fs_);
        runtime_->attach_name(lr_, "node_pool_lr");

        // --- Initialise child pointers to NULL_IDX (WRITE_DISCARD) ----------
        {
            RegionRequirement req(lr_, WRITE_DISCARD, EXCLUSIVE, lr_);
            req.add_field(FID_KEY);
            req.add_field(FID_VALUE);
            req.add_field(FID_LEFT);
            req.add_field(FID_RIGHT);
            InlineLauncher launcher(req);
            PhysicalRegion pr = runtime_->map_region(ctx_, launcher);
            pr.wait_until_valid();

            const FieldAccessor<WRITE_DISCARD, int, 1> acc_left(pr,  FID_LEFT);
            const FieldAccessor<WRITE_DISCARD, int, 1> acc_right(pr, FID_RIGHT);
            for (int i = 0; i < MAX_NODES; ++i) {
                acc_left[i]  = NULL_IDX;
                acc_right[i] = NULL_IDX;
            }
            runtime_->unmap_region(ctx_, pr);
        }

        // --- Keep the region mapped READ_WRITE for all future operations ----
        {
            RegionRequirement req(lr_, READ_WRITE, EXCLUSIVE, lr_);
            req.add_field(FID_KEY);
            req.add_field(FID_VALUE);
            req.add_field(FID_LEFT);
            req.add_field(FID_RIGHT);
            InlineLauncher launcher(req);
            pr_ = runtime_->map_region(ctx_, launcher);
            pr_.wait_until_valid();
        }
    }

    ~binary_tree() {
        runtime_->unmap_region(ctx_, pr_);
        runtime_->destroy_logical_region(ctx_, lr_);
        runtime_->destroy_field_space(ctx_, fs_);
        runtime_->destroy_index_space(ctx_, is_);
    }

    // -----------------------------------------------------------------------
    // insert – mirrors the original transactional insert.
    //
    // The original wrapped every access in a transaction that could retry;
    // here the inline-mapped READ_WRITE region provides exclusive access.
    // -----------------------------------------------------------------------
    void insert(Key key, const Value& value)
    {
        const FieldAccessor<READ_WRITE, Key,       1> acc_key(pr_,   FID_KEY);
        const FieldAccessor<READ_WRITE, NodeValue,  1> acc_val(pr_,  FID_VALUE);
        const FieldAccessor<READ_WRITE, int,        1> acc_left(pr_, FID_LEFT);
        const FieldAccessor<READ_WRITE, int,        1> acc_right(pr_,FID_RIGHT);

        // Allocate a new node in the pool (replaces `new node(key, value)`)
        int new_idx = next_node_++;
        acc_key[new_idx]   = key;
        acc_val[new_idx]   = NodeValue(value);
        acc_left[new_idx]  = NULL_IDX;
        acc_right[new_idx] = NULL_IDX;

        if (root_idx_ == NULL_IDX) {
            root_idx_ = new_idx;
        } else {
            int cur = root_idx_;
            while (true) {
                if (key < acc_key[cur]) {
                    if (acc_left[cur] != NULL_IDX) {
                        cur = acc_left[cur];
                    } else {
                        acc_left[cur] = new_idx;
                        break;
                    }
                } else {                       // key >= acc_key[cur]
                    if (acc_right[cur] != NULL_IDX) {
                        cur = acc_right[cur];
                    } else {
                        acc_right[cur] = new_idx;
                        break;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // search – mirrors the original transactional search.
    // -----------------------------------------------------------------------
    SearchResult search(Key key)
    {
        const FieldAccessor<READ_WRITE, Key,       1> acc_key(pr_,   FID_KEY);
        const FieldAccessor<READ_WRITE, NodeValue,  1> acc_val(pr_,  FID_VALUE);
        const FieldAccessor<READ_WRITE, int,        1> acc_left(pr_, FID_LEFT);
        const FieldAccessor<READ_WRITE, int,        1> acc_right(pr_,FID_RIGHT);

        SearchResult result{false, ""};
        int cur = root_idx_;

        while (cur != NULL_IDX) {
            Key cur_key = acc_key[cur];
            if (key == cur_key) {
                result.found = true;
                NodeValue nv = acc_val[cur];
                result.value = nv.str();
                return result;
            } else if (key < cur_key) {
                cur = acc_left[cur];
            } else {
                cur = acc_right[cur];
            }
        }
        return result;
    }

private:
    Runtime*       runtime_;
    Context        ctx_;
    IndexSpace     is_;
    FieldSpace     fs_;
    LogicalRegion  lr_;
    PhysicalRegion pr_;
    int            next_node_;
    int            root_idx_;
};

// ---------------------------------------------------------------------------
// Top-level Legion task  (replaces hpx_main / main)
// ---------------------------------------------------------------------------
void top_level_task(const Task*                      /*task*/,
                    const std::vector<PhysicalRegion>& /*regions*/,
                    Context ctx,
                    Runtime* runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    {
        binary_tree<int, std::string> tree(runtime, ctx);

        // 1. Data Insertion Phase
        struct DataEntry {
            int         id;
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

            auto result = tree.search(search_key);

            if (result.found) {
                outfile << "Found! Result Value: \"" << result.value << "\"\n";
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
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime.
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
