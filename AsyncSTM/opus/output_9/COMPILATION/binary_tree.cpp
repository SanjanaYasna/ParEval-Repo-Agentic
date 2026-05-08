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
#include <cstring>
#include <vector>

#include "legion.h"

using namespace Legion;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int MAX_NODES   = 128;
static const int MAX_VAL_LEN = 64;
static const int NULL_NODE   = -1;

// ---------------------------------------------------------------------------
// Task IDs
// ---------------------------------------------------------------------------
enum TaskID {
  TOP_LEVEL_TASK_ID,
  INSERT_TASK_ID,
  SEARCH_TASK_ID,
};

// ---------------------------------------------------------------------------
// Field IDs – tree node region
// ---------------------------------------------------------------------------
enum TreeFieldID {
  FID_KEY   = 0,
  FID_VALUE = 1,
  FID_LEFT  = 2,
  FID_RIGHT = 3,
};

// ---------------------------------------------------------------------------
// Field IDs – metadata region
// ---------------------------------------------------------------------------
enum MetaFieldID {
  FID_ROOT_IDX   = 10,
  FID_NODE_COUNT = 11,
};

// ---------------------------------------------------------------------------
// Fixed-size value wrapper (POD, safe for region storage and Future results)
// ---------------------------------------------------------------------------
struct NodeValue {
  char data[MAX_VAL_LEN];
};

// ---------------------------------------------------------------------------
// Task argument / result types
// ---------------------------------------------------------------------------
struct InsertArgs {
  int       key;
  NodeValue value;
};

struct SearchResult {
  bool      found;
  NodeValue value;
};

// ---------------------------------------------------------------------------
// Insert Task
//   regions[0] : tree node region  (READ_WRITE)
//   regions[1] : metadata region   (READ_WRITE)
// ---------------------------------------------------------------------------
void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  const InsertArgs &args = *reinterpret_cast<const InsertArgs *>(task->args);

  const FieldAccessor<READ_WRITE, int, 1>       acc_key  (regions[0], FID_KEY);
  const FieldAccessor<READ_WRITE, NodeValue, 1> acc_val  (regions[0], FID_VALUE);
  const FieldAccessor<READ_WRITE, int, 1>       acc_left (regions[0], FID_LEFT);
  const FieldAccessor<READ_WRITE, int, 1>       acc_right(regions[0], FID_RIGHT);

  const FieldAccessor<READ_WRITE, int, 1> acc_root (regions[1], FID_ROOT_IDX);
  const FieldAccessor<READ_WRITE, int, 1> acc_count(regions[1], FID_NODE_COUNT);

  int root_idx = acc_root[0];
  int count    = acc_count[0];

  // Allocate the new node at the next free slot
  int new_idx = count;
  acc_key[new_idx]   = args.key;
  acc_val[new_idx]   = args.value;
  acc_left[new_idx]  = NULL_NODE;
  acc_right[new_idx] = NULL_NODE;
  acc_count[0]       = count + 1;

  // Empty tree → new node becomes root
  if (root_idx == NULL_NODE) {
    acc_root[0] = new_idx;
    return;
  }

  // Walk the tree to find the correct insertion point
  int current = root_idx;
  while (true) {
    if (args.key < acc_key[current]) {
      int left = acc_left[current];
      if (left != NULL_NODE) {
        current = left;
      } else {
        acc_left[current] = new_idx;
        break;
      }
    } else {                          // key >= current key (matches original)
      int right = acc_right[current];
      if (right != NULL_NODE) {
        current = right;
      } else {
        acc_right[current] = new_idx;
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Search Task
//   regions[0] : tree node region  (READ_ONLY)
//   regions[1] : metadata region   (READ_ONLY)
//   returns    : SearchResult
// ---------------------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  int search_key = *reinterpret_cast<const int *>(task->args);

  const FieldAccessor<READ_ONLY, int, 1>       acc_key  (regions[0], FID_KEY);
  const FieldAccessor<READ_ONLY, NodeValue, 1> acc_val  (regions[0], FID_VALUE);
  const FieldAccessor<READ_ONLY, int, 1>       acc_left (regions[0], FID_LEFT);
  const FieldAccessor<READ_ONLY, int, 1>       acc_right(regions[0], FID_RIGHT);

  const FieldAccessor<READ_ONLY, int, 1> acc_root(regions[1], FID_ROOT_IDX);

  SearchResult result;
  result.found = false;
  std::memset(result.value.data, 0, MAX_VAL_LEN);

  int current = acc_root[0];

  while (current != NULL_NODE) {
    int cur_key = acc_key[current];
    if (search_key == cur_key) {
      result.found = true;
      result.value = acc_val[current];
      return result;
    } else if (search_key < cur_key) {
      current = acc_left[current];
    } else {
      current = acc_right[current];
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Top-level Task
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // ---- Create the tree-node region (MAX_NODES entries) --------------------
  IndexSpace tree_is = runtime->create_index_space(ctx,
                          Rect<1>(0, MAX_NODES - 1));
  FieldSpace tree_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, tree_fs);
    fa.allocate_field(sizeof(int),       FID_KEY);
    fa.allocate_field(sizeof(NodeValue), FID_VALUE);
    fa.allocate_field(sizeof(int),       FID_LEFT);
    fa.allocate_field(sizeof(int),       FID_RIGHT);
  }
  LogicalRegion tree_lr = runtime->create_logical_region(ctx, tree_is, tree_fs);

  // ---- Create the metadata region (1 entry) -------------------------------
  IndexSpace meta_is = runtime->create_index_space(ctx, Rect<1>(0, 0));
  FieldSpace meta_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, meta_fs);
    fa.allocate_field(sizeof(int), FID_ROOT_IDX);
    fa.allocate_field(sizeof(int), FID_NODE_COUNT);
  }
  LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

  // ---- Initialise metadata (root = NULL, count = 0) ----------------------
  {
    InlineLauncher il(RegionRequirement(meta_lr, WRITE_DISCARD,
                                        EXCLUSIVE, meta_lr));
    il.add_field(FID_ROOT_IDX);
    il.add_field(FID_NODE_COUNT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    const FieldAccessor<WRITE_DISCARD, int, 1> wr_root(pr, FID_ROOT_IDX);
    const FieldAccessor<WRITE_DISCARD, int, 1> wr_cnt (pr, FID_NODE_COUNT);
    wr_root[0] = NULL_NODE;
    wr_cnt[0]  = 0;

    runtime->unmap_region(ctx, pr);
  }

  // ---- Initialise tree-node region ----------------------------------------
  {
    InlineLauncher il(RegionRequirement(tree_lr, WRITE_DISCARD,
                                        EXCLUSIVE, tree_lr));
    il.add_field(FID_KEY);
    il.add_field(FID_VALUE);
    il.add_field(FID_LEFT);
    il.add_field(FID_RIGHT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    const FieldAccessor<WRITE_DISCARD, int, 1>       wr_key  (pr, FID_KEY);
    const FieldAccessor<WRITE_DISCARD, NodeValue, 1> wr_val  (pr, FID_VALUE);
    const FieldAccessor<WRITE_DISCARD, int, 1>       wr_left (pr, FID_LEFT);
    const FieldAccessor<WRITE_DISCARD, int, 1>       wr_right(pr, FID_RIGHT);

    NodeValue empty_val;
    std::memset(empty_val.data, 0, MAX_VAL_LEN);

    for (int i = 0; i < MAX_NODES; ++i) {
      wr_key[i]   = 0;
      wr_val[i]   = empty_val;
      wr_left[i]  = NULL_NODE;
      wr_right[i] = NULL_NODE;
    }

    runtime->unmap_region(ctx, pr);
  }

  // ---- Open output file ---------------------------------------------------
  std::ofstream outfile("binary_tree.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_logical_region(ctx, meta_lr);
    runtime->destroy_field_space(ctx, tree_fs);
    runtime->destroy_field_space(ctx, meta_fs);
    runtime->destroy_index_space(ctx, tree_is);
    runtime->destroy_index_space(ctx, meta_is);
    return;
  }

  // ---- 1. Data Insertion Phase --------------------------------------------
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
  const int num_entries = 7;

  outfile << "Inserting Data:\n";

  for (int i = 0; i < num_entries; ++i) {
    outfile << "  Insert(Key: " << data[i].id
            << ", Value: \"" << data[i].val << "\")" << std::endl;

    InsertArgs args;
    args.key = data[i].id;
    std::memset(args.value.data, 0, MAX_VAL_LEN);
    std::strncpy(args.value.data, data[i].val, MAX_VAL_LEN - 1);

    TaskLauncher launcher(INSERT_TASK_ID,
                          TaskArgument(&args, sizeof(args)));

    launcher.add_region_requirement(
      RegionRequirement(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr));
    launcher.region_requirements[0].add_field(FID_KEY);
    launcher.region_requirements[0].add_field(FID_VALUE);
    launcher.region_requirements[0].add_field(FID_LEFT);
    launcher.region_requirements[0].add_field(FID_RIGHT);

    launcher.add_region_requirement(
      RegionRequirement(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr));
    launcher.region_requirements[1].add_field(FID_ROOT_IDX);
    launcher.region_requirements[1].add_field(FID_NODE_COUNT);

    // Sequential execution: the runtime enforces ordering because
    // successive inserts share READ_WRITE access to the same regions.
    runtime->execute_task(ctx, launcher);
  }

  outfile << "\n";

  // ---- 2. Search Test Phase -----------------------------------------------
  outfile << "Running Search Tests:\n";

  int search_keys[] = {99, 10, 50, 20, 60, 80, 45};
  const int num_searches = 7;

  for (int i = 0; i < num_searches; ++i) {
    int skey = search_keys[i];

    TaskLauncher launcher(SEARCH_TASK_ID,
                          TaskArgument(&skey, sizeof(int)));

    launcher.add_region_requirement(
      RegionRequirement(tree_lr, READ_ONLY, EXCLUSIVE, tree_lr));
    launcher.region_requirements[0].add_field(FID_KEY);
    launcher.region_requirements[0].add_field(FID_VALUE);
    launcher.region_requirements[0].add_field(FID_LEFT);
    launcher.region_requirements[0].add_field(FID_RIGHT);

    launcher.add_region_requirement(
      RegionRequirement(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr));
    launcher.region_requirements[1].add_field(FID_ROOT_IDX);
    launcher.region_requirements[1].add_field(FID_NODE_COUNT);

    Future f = runtime->execute_task(ctx, launcher);
    SearchResult res = f.get_result<SearchResult>();

    outfile << "  Search(Key: " << skey << ") -> ";
    if (res.found) {
      outfile << "Found! Result Value: \"" << res.value.data << "\"\n";
    } else {
      outfile << "Not Found.\n";
    }
  }

  outfile.close();

  // ---- Cleanup ------------------------------------------------------------
  runtime->destroy_logical_region(ctx, tree_lr);
  runtime->destroy_logical_region(ctx, meta_lr);
  runtime->destroy_field_space(ctx, tree_fs);
  runtime->destroy_field_space(ctx, meta_fs);
  runtime->destroy_index_space(ctx, tree_is);
  runtime->destroy_index_space(ctx, meta_is);
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
