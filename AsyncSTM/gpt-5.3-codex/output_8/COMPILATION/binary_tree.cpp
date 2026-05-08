////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "astm.hpp"

using namespace astm;

template <typename Key, typename Value>
struct binary_tree
{
    struct node
    {
        shared_var<Key> key;
        Value value;
        shared_var<node*> left;
        shared_var<node*> right;

        node()
          : key(), value(), left(), right()
        {}

        node(Key k, Value const& v, node* l, node* r)
          : key(k), value(v), left(l), right(r)
        {}

        node(Key k, Value const& v, shared_var<node*> l, shared_var<node*> r)
          : key(k), value(v), left(l), right(r)
        {}

        node(Key k, Value const& v)
          : key(k), value(v), left(), right()
        {}
    };

    binary_tree()
      : root()
    {}

    void insert(Key key, Value const& value)
    {
        transaction t;
        do {
            auto root_ = root.get_local(t);

            if (root_.get() != nullptr)
                insert(key, value, root, t);
            else
                root_ = new node(key, value);
        } while (!t.commit_transaction());
    }

    node* search(Key key)
    {
        node* n = nullptr;

        transaction t;
        do {
            n = search(key, root, t);
        } while (!t.commit_transaction());

        return n;
    }

  private:
    void insert(Key key, Value const& value, shared_var<node*>& leaf, transaction& t)
    {
        auto leaf_ = leaf.get_local(t);

        if (key < leaf_.get()->key.get_local(t).get())
        {
            if (leaf_.get()->left.get_local(t).get() != nullptr)
                insert(key, value, leaf_.get()->left, t);
            else
                leaf_.get()->left.get_local(t) = new node(key, value);
        }
        else // key >= current key
        {
            if (leaf_.get()->right.get_local(t).get() != nullptr)
                insert(key, value, leaf_.get()->right, t);
            else
                leaf_.get()->right.get_local(t) = new node(key, value);
        }
    }

    node* search(Key key, shared_var<node*>& leaf, transaction& t)
    {
        auto leaf_ = leaf.get_local(t);

        if (leaf_.get() != nullptr)
        {
            if (key == leaf_.get()->key.get_local(t).get())
                return leaf_.get();
            if (key < leaf_.get()->key.get_local(t).get())
                return search(key, leaf_.get()->left, t);
            else
                return search(key, leaf_.get()->right, t);
        }
        else
        {
            return nullptr;
        }
    }

    shared_var<node*> root;
};

int main()
{
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return 1;
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
    return 0;
}
