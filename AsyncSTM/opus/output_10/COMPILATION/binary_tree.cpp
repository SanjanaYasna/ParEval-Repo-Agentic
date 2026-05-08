////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <legion.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string>

using namespace Legion;

static const int NIL_NODE = -1;
static const size_t MAX_VALUE_LEN = 64;
static const size_t MAX_NODES = 128;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

// Fixed-size string suitable for storage in a Legion logical region field
struct FixedString {
    char data[MAX_VALUE_LEN];
    FixedString() { memset(data, 0, sizeof(data)); }
    FixedString(const std::string& s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }
    std::string str() const { return std::string(data); }
};

// Binary tree that stores node data in a Legion logical region.
// Nodes are allocated from a flat pool; left/right children are
// represented as integer indices into that pool (NIL_NODE = null).
template <typename Key>
class BinaryTree {
public:
    BinaryTree(Context ctx, Runtime* runtime)
        : ctx_(ctx), runtime_(runtime), root_idx_(NIL_NODE), node_count_(0)
    {
        is_ = runtime->create_index_space(ctx,
                  Rect<1>(Point<1>(0), Point<1>(MAX_NODES - 1)));
        fs_ = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs_);
            fa.allocate_field(sizeof(Key),         FID_KEY);
            fa.allocate_field(sizeof(FixedString), FID_VALUE);
            fa.allocate_field(sizeof(int),         FID_LEFT);
            fa.allocate_field(sizeof(int),         FID_RIGHT);
        }
        lr_ = runtime->create_logical_region(ctx, is_, fs_);
    }

    ~BinaryTree() {
        runtime_->destroy_logical_region(ctx_, lr_);
        runtime_->destroy_field_space(ctx_, fs_);
        runtime_->destroy_index_space(ctx_, is_);
    }

    // ---- insert ---------------------------------------------------------
    void insert(Key key, const std::string& value)
    {
        RegionRequirement req(lr_, READ_WRITE, EXCLUSIVE, lr_);
        req.add_field(FID_KEY);
        req.add_field(FID_VALUE);
        req.add_field(FID_LEFT);
        req.add_field(FID_RIGHT);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime_->map_region(ctx_, launcher);
        pr.wait_until_valid();

        const FieldAccessor<READ_WRITE, Key,         1> acc_key(pr, FID_KEY);
        const FieldAccessor<READ_WRITE, FixedString,  1> acc_val(pr, FID_VALUE);
        const FieldAccessor<READ_WRITE, int,          1> acc_left(pr, FID_LEFT);
        const FieldAccessor<READ_WRITE, int,          1> acc_right(pr, FID_RIGHT);

        // Allocate a new node at the end of the pool
        int new_idx = node_count_++;
        acc_key[new_idx]   = key;
        acc_val[new_idx]   = FixedString(value);
        acc_left[new_idx]  = NIL_NODE;
        acc_right[new_idx] = NIL_NODE;

        if (root_idx_ == NIL_NODE) {
            root_idx_ = new_idx;
        } else {
            insert_at(key, new_idx, root_idx_,
                      acc_key, acc_left, acc_right);
        }

        runtime_->unmap_region(ctx_, pr);
    }

    // ---- search ---------------------------------------------------------
    struct SearchResult {
        bool        found;
        std::string value;
    };

    SearchResult search(Key key)
    {
        if (root_idx_ == NIL_NODE)
            return {false, ""};

        RegionRequirement req(lr_, READ_ONLY, EXCLUSIVE, lr_);
        req.add_field(FID_KEY);
        req.add_field(FID_VALUE);
        req.add_field(FID_LEFT);
        req.add_field(FID_RIGHT);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime_->map_region(ctx_, launcher);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, Key,         1> acc_key(pr, FID_KEY);
        const FieldAccessor<READ_ONLY, FixedString,  1> acc_val(pr, FID_VALUE);
        const FieldAccessor<READ_ONLY, int,          1> acc_left(pr, FID_LEFT);
        const FieldAccessor<READ_ONLY, int,          1> acc_right(pr, FID_RIGHT);

        SearchResult result = search_at(key, root_idx_,
                                        acc_key, acc_val,
                                        acc_left, acc_right);

        runtime_->unmap_region(ctx_, pr);
        return result;
    }

private:
    // Recursive insert: walk the tree and place new_idx in the correct spot.
    // Equal keys go to the right (matching the original >= branch).
    void insert_at(Key key, int new_idx, int cur,
                   const FieldAccessor<READ_WRITE, Key, 1>& acc_key,
                   const FieldAccessor<READ_WRITE, int, 1>& acc_left,
                   const FieldAccessor<READ_WRITE, int, 1>& acc_right)
    {
        if (key < static_cast<Key>(acc_key[cur])) {
            int left = acc_left[cur];
            if (left != NIL_NODE)
                insert_at(key, new_idx, left, acc_key, acc_left, acc_right);
            else
                acc_left[cur] = new_idx;
        } else {
            // key >= current node key  →  go right
            int right = acc_right[cur];
            if (right != NIL_NODE)
                insert_at(key, new_idx, right, acc_key, acc_left, acc_right);
            else
                acc_right[cur] = new_idx;
        }
    }

    // Recursive search: return found + value string.
    SearchResult search_at(Key key, int cur,
                           const FieldAccessor<READ_ONLY, Key,        1>& acc_key,
                           const FieldAccessor<READ_ONLY, FixedString,1>& acc_val,
                           const FieldAccessor<READ_ONLY, int,        1>& acc_left,
                           const FieldAccessor<READ_ONLY, int,        1>& acc_right)
    {
        if (cur == NIL_NODE)
            return {false, ""};

        Key cur_key = acc_key[cur];
        if (key == cur_key) {
            FixedString fs = acc_val[cur];
            return {true, fs.str()};
        }
        else if (key < cur_key)
            return search_at(key, static_cast<int>(acc_left[cur]),
                             acc_key, acc_val, acc_left, acc_right);
        else
            return search_at(key, static_cast<int>(acc_right[cur]),
                             acc_key, acc_val, acc_left, acc_right);
    }

    Context       ctx_;
    Runtime*      runtime_;
    IndexSpace    is_;
    FieldSpace    fs_;
    LogicalRegion lr_;
    int           root_idx_;
    int           node_count_;
};

// =========================================================================
// Top-level Legion task – mirrors the original main()
// =========================================================================
void top_level_task(const Task*                      task,
                    const std::vector<PhysicalRegion>& regions,
                    Context                           ctx,
                    Runtime*                          runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    {
        BinaryTree<int> tree(ctx, runtime);

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

// =========================================================================
// main – register tasks and start the Legion runtime
// =========================================================================
int main(int argc, char** argv)
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
