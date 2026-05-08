////////////////////////////////////////////////////////////////////////////////
//  Translated to Legion execution model (default mapper)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace Legion;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1
};

template <typename Key, typename Value>
struct binary_tree
{
    struct node
    {
        Key key;
        Value value;
        node* left;
        node* right;

        node() : key(), value(), left(nullptr), right(nullptr) {}
        node(Key k, Value const& v) : key(k), value(v), left(nullptr), right(nullptr) {}
    };

    binary_tree() : root(nullptr) {}

    ~binary_tree()
    {
        destroy(root);
        root = nullptr;
    }

    void insert(Key key, Value const& value)
    {
        insert_impl(root, key, value);
    }

    node* search(Key key) const
    {
        return search_impl(root, key);
    }

private:
    static void insert_impl(node*& leaf, Key key, Value const& value)
    {
        if (leaf == nullptr)
        {
            leaf = new node(key, value);
            return;
        }

        if (key < leaf->key)
            insert_impl(leaf->left, key, value);
        else
            insert_impl(leaf->right, key, value); // key >= leaf->key goes right
    }

    static node* search_impl(node* leaf, Key key)
    {
        if (leaf == nullptr)
            return nullptr;

        if (key == leaf->key)
            return leaf;

        if (key < leaf->key)
            return search_impl(leaf->left, key);
        else
            return search_impl(leaf->right, key);
    }

    static void destroy(node* n)
    {
        if (!n) return;
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    node* root;
};

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*)
{
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    binary_tree<int, std::string> tree;

    struct DataEntry
    {
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
    for (const auto& entry : data)
    {
        outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")" << std::endl;
        tree.insert(entry.id, entry.val);
    }
    outfile << "\n";

    outfile << "Running Search Tests:\n";

    auto perform_search = [&](int search_key) {
        outfile << "  Search(Key: " << search_key << ") -> ";

        auto n = tree.search(search_key);

        if (n != nullptr)
            outfile << "Found! Result Value: \"" << n->value << "\"\n";
        else
            outfile << "Not Found.\n";
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

int main(int argc, char** argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "binary_tree_top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "binary_tree_top_level");
    }

    return Runtime::start(argc, argv);
}
