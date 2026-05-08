////////////////////////////////////////////////////////////////////////////////
//  Translated to Legion execution model (default mapper)
//  Unbalanced binary tree demo with file output: binary_tree.txt
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace Legion;

enum TaskIDs : Legion::TaskID {
    TOP_LEVEL_TASK_ID = 1
};

template <typename Key, typename Value>
class binary_tree {
public:
    struct node {
        Key key;
        Value value;
        node* left;
        node* right;

        node(Key k, const Value& v) : key(k), value(v), left(nullptr), right(nullptr) {}
    };

    binary_tree() : root_(nullptr) {}

    ~binary_tree() { destroy(root_); }

    void insert(Key key, const Value& value) {
        if (root_ == nullptr) {
            root_ = new node(key, value);
            return;
        }
        insert_impl(root_, key, value);
    }

    node* search(Key key) const { return search_impl(root_, key); }

private:
    node* root_;

    static void insert_impl(node* cur, Key key, const Value& value) {
        if (key < cur->key) {
            if (cur->left) {
                insert_impl(cur->left, key, value);
            } else {
                cur->left = new node(key, value);
            }
        } else {
            if (cur->right) {
                insert_impl(cur->right, key, value);
            } else {
                cur->right = new node(key, value);
            }
        }
    }

    static node* search_impl(node* cur, Key key) {
        if (cur == nullptr) return nullptr;
        if (key == cur->key) return cur;
        if (key < cur->key) return search_impl(cur->left, key);
        return search_impl(cur->right, key);
    }

    static void destroy(node* cur) {
        if (!cur) return;
        destroy(cur->left);
        destroy(cur->right);
        delete cur;
    }
};

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    binary_tree<int, std::string> tree;

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
        outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")" << std::endl;
        tree.insert(entry.id, entry.val);
    }
    outfile << "\n";

    outfile << "Running Search Tests:\n";

    auto perform_search = [&](int search_key) {
        outfile << "  Search(Key: " << search_key << ") -> ";
        auto n = tree.search(search_key);
        if (n != nullptr) {
            outfile << "Found! Result Value: \"" << n->value << "\"\n";
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

    outfile.close();
}

int main(int argc, char** argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
    }

    return Runtime::start(argc, argv);
}
