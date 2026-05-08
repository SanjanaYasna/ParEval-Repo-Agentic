////////////////////////////////////////////////////////////////////////////////
// Legion translation of binary_tree.cpp
////////////////////////////////////////////////////////////////////////////////

#include <legion.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace Legion;

//------------------------------------------------------------------------------
// Task and Field IDs
//------------------------------------------------------------------------------
enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INIT_TASK_ID,
  INSERT_TASK_ID,
  SEARCH_TASK_ID
};

enum NodeFieldIDs {
  FID_NODE_USED = 100,
  FID_NODE_KEY,
  FID_NODE_VALUE,
  FID_NODE_LEFT,
  FID_NODE_RIGHT
};

enum MetaFieldIDs {
  FID_META_ROOT = 200,
  FID_META_NEXT_FREE
};

//------------------------------------------------------------------------------
// Data layout
//------------------------------------------------------------------------------
static constexpr int MAX_NODES = 1024;
static constexpr size_t VALUE_CAP = 128;

struct ValueBuffer {
  char data[VALUE_CAP];
};

struct InsertArgs {
  int key;
  ValueBuffer value;
};

struct SearchArgs {
  int key;
};

struct SearchResult {
  int found; // 0 = false, 1 = true
  ValueBuffer value;
};

static ValueBuffer make_value_buffer(const char *s) {
  ValueBuffer out{};
  std::snprintf(out.data, VALUE_CAP, "%s", s ? s : "");
  return out;
}

//------------------------------------------------------------------------------
// INIT task
//------------------------------------------------------------------------------
void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx,
               Runtime *runtime) {
  (void)task;
  assert(regions.size() == 2);

  FieldAccessor<WRITE_DISCARD, int, 1> acc_used(regions[0], FID_NODE_USED);
  FieldAccessor<WRITE_DISCARD, int, 1> acc_key(regions[0], FID_NODE_KEY);
  FieldAccessor<WRITE_DISCARD, ValueBuffer, 1> acc_value(regions[0], FID_NODE_VALUE);
  FieldAccessor<WRITE_DISCARD, int, 1> acc_left(regions[0], FID_NODE_LEFT);
  FieldAccessor<WRITE_DISCARD, int, 1> acc_right(regions[0], FID_NODE_RIGHT);

  FieldAccessor<WRITE_DISCARD, int, 1> acc_root(regions[1], FID_META_ROOT);
  FieldAccessor<WRITE_DISCARD, int, 1> acc_next_free(regions[1], FID_META_NEXT_FREE);

  Rect<1> node_rect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(node_rect); it(); it++) {
    const Point<1> p = *it;
    acc_used[p] = 0;
    acc_key[p] = 0;
    acc_left[p] = -1;
    acc_right[p] = -1;
    ValueBuffer empty{};
    empty.data[0] = '\0';
    acc_value[p] = empty;
  }

  const Point<1> mp(0);
  acc_root[mp] = -1;
  acc_next_free[mp] = 0;
}

//------------------------------------------------------------------------------
// INSERT task
//------------------------------------------------------------------------------
void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context /*ctx*/,
                 Runtime * /*runtime*/) {
  assert(task->arglen == sizeof(InsertArgs));
  const InsertArgs *args = static_cast<const InsertArgs *>(task->args);
  assert(regions.size() == 2);

  FieldAccessor<READ_WRITE, int, 1> acc_used(regions[0], FID_NODE_USED);
  FieldAccessor<READ_WRITE, int, 1> acc_key(regions[0], FID_NODE_KEY);
  FieldAccessor<READ_WRITE, ValueBuffer, 1> acc_value(regions[0], FID_NODE_VALUE);
  FieldAccessor<READ_WRITE, int, 1> acc_left(regions[0], FID_NODE_LEFT);
  FieldAccessor<READ_WRITE, int, 1> acc_right(regions[0], FID_NODE_RIGHT);

  FieldAccessor<READ_WRITE, int, 1> acc_root(regions[1], FID_META_ROOT);
  FieldAccessor<READ_WRITE, int, 1> acc_next_free(regions[1], FID_META_NEXT_FREE);

  const Point<1> mp(0);
  int root = acc_root[mp];
  int next_free = acc_next_free[mp];

  auto allocate_node = [&](int key, const ValueBuffer &value) -> int {
    if (next_free >= MAX_NODES) {
      return -1;
    }
    const int idx = next_free++;
    const Point<1> p(idx);
    acc_used[p] = 1;
    acc_key[p] = key;
    acc_value[p] = value;
    acc_left[p] = -1;
    acc_right[p] = -1;
    return idx;
  };

  if (root == -1) {
    root = allocate_node(args->key, args->value);
    acc_root[mp] = root;
    acc_next_free[mp] = next_free;
    return;
  }

  int cur = root;
  while (true) {
    const Point<1> cp(cur);
    const int cur_key = acc_key[cp];

    if (args->key < cur_key) {
      int left = acc_left[cp];
      if (left != -1) {
        cur = left;
      } else {
        int ni = allocate_node(args->key, args->value);
        if (ni != -1) acc_left[cp] = ni;
        break;
      }
    } else { // >= goes right (matches original behavior)
      int right = acc_right[cp];
      if (right != -1) {
        cur = right;
      } else {
        int ni = allocate_node(args->key, args->value);
        if (ni != -1) acc_right[cp] = ni;
        break;
      }
    }
  }

  acc_root[mp] = root;
  acc_next_free[mp] = next_free;
}

//------------------------------------------------------------------------------
// SEARCH task
//------------------------------------------------------------------------------
SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context /*ctx*/,
                         Runtime * /*runtime*/) {
  assert(task->arglen == sizeof(SearchArgs));
  const SearchArgs *args = static_cast<const SearchArgs *>(task->args);
  assert(regions.size() == 2);

  FieldAccessor<READ_ONLY, int, 1> acc_used(regions[0], FID_NODE_USED);
  FieldAccessor<READ_ONLY, int, 1> acc_key(regions[0], FID_NODE_KEY);
  FieldAccessor<READ_ONLY, ValueBuffer, 1> acc_value(regions[0], FID_NODE_VALUE);
  FieldAccessor<READ_ONLY, int, 1> acc_left(regions[0], FID_NODE_LEFT);
  FieldAccessor<READ_ONLY, int, 1> acc_right(regions[0], FID_NODE_RIGHT);

  FieldAccessor<READ_ONLY, int, 1> acc_root(regions[1], FID_META_ROOT);

  SearchResult result{};
  result.found = 0;
  result.value.data[0] = '\0';

  const Point<1> mp(0);
  int cur = acc_root[mp];

  while (cur != -1) {
    const Point<1> cp(cur);
    if (acc_used[cp] == 0) break;

    const int cur_key = acc_key[cp];
    if (args->key == cur_key) {
      result.found = 1;
      result.value = acc_value[cp];
      return result;
    } else if (args->key < cur_key) {
      cur = acc_left[cp];
    } else {
      cur = acc_right[cp];
    }
  }

  return result;
}

//------------------------------------------------------------------------------
// Top-level task
//------------------------------------------------------------------------------
void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx,
                    Runtime *runtime) {
  std::ofstream outfile("binary_tree.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
    return;
  }

  // Create node region
  Rect<1> node_bounds(0, MAX_NODES - 1);
  IndexSpace node_is = runtime->create_index_space(ctx, node_bounds);
  FieldSpace node_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, node_fs);
    alloc.allocate_field(sizeof(int), FID_NODE_USED);
    alloc.allocate_field(sizeof(int), FID_NODE_KEY);
    alloc.allocate_field(sizeof(ValueBuffer), FID_NODE_VALUE);
    alloc.allocate_field(sizeof(int), FID_NODE_LEFT);
    alloc.allocate_field(sizeof(int), FID_NODE_RIGHT);
  }
  LogicalRegion node_lr = runtime->create_logical_region(ctx, node_is, node_fs);

  // Create metadata region (single element)
  Rect<1> meta_bounds(0, 0);
  IndexSpace meta_is = runtime->create_index_space(ctx, meta_bounds);
  FieldSpace meta_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, meta_fs);
    alloc.allocate_field(sizeof(int), FID_META_ROOT);
    alloc.allocate_field(sizeof(int), FID_META_NEXT_FREE);
  }
  LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

  // Initialize regions
  {
    TaskLauncher launcher(INIT_TASK_ID, TaskArgument(nullptr, 0));

    RegionRequirement nreq(node_lr, WRITE_DISCARD, EXCLUSIVE, node_lr);
    nreq.add_field(FID_NODE_USED);
    nreq.add_field(FID_NODE_KEY);
    nreq.add_field(FID_NODE_VALUE);
    nreq.add_field(FID_NODE_LEFT);
    nreq.add_field(FID_NODE_RIGHT);
    launcher.add_region_requirement(nreq);

    RegionRequirement mreq(meta_lr, WRITE_DISCARD, EXCLUSIVE, meta_lr);
    mreq.add_field(FID_META_ROOT);
    mreq.add_field(FID_META_NEXT_FREE);
    launcher.add_region_requirement(mreq);

    runtime->execute_task(ctx, launcher).get_void_result();
  }

  struct DataEntry {
    int id;
    const char *val;
  };

  std::vector<DataEntry> data = {
      {50, "Root Node"},
      {30, "Left Child"},
      {70, "Right Child"},
      {20, "Left-Left Grandchild"},
      {40, "Left-Right Grandchild"},
      {60, "Right-Left Grandchild"},
      {80, "Right-Right Grandchild"}};

  outfile << "Inserting Data:\n";
  for (const auto &entry : data) {
    outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")" << std::endl;

    InsertArgs args{};
    args.key = entry.id;
    args.value = make_value_buffer(entry.val);

    TaskLauncher launcher(INSERT_TASK_ID, TaskArgument(&args, sizeof(args)));

    RegionRequirement nreq(node_lr, READ_WRITE, EXCLUSIVE, node_lr);
    nreq.add_field(FID_NODE_USED);
    nreq.add_field(FID_NODE_KEY);
    nreq.add_field(FID_NODE_VALUE);
    nreq.add_field(FID_NODE_LEFT);
    nreq.add_field(FID_NODE_RIGHT);
    launcher.add_region_requirement(nreq);

    RegionRequirement mreq(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr);
    mreq.add_field(FID_META_ROOT);
    mreq.add_field(FID_META_NEXT_FREE);
    launcher.add_region_requirement(mreq);

    runtime->execute_task(ctx, launcher).get_void_result();
  }
  outfile << "\n";

  outfile << "Running Search Tests:\n";
  auto perform_search = [&](int search_key) {
    outfile << "  Search(Key: " << search_key << ") -> ";

    SearchArgs args{search_key};
    TaskLauncher launcher(SEARCH_TASK_ID, TaskArgument(&args, sizeof(args)));

    RegionRequirement nreq(node_lr, READ_ONLY, EXCLUSIVE, node_lr);
    nreq.add_field(FID_NODE_USED);
    nreq.add_field(FID_NODE_KEY);
    nreq.add_field(FID_NODE_VALUE);
    nreq.add_field(FID_NODE_LEFT);
    nreq.add_field(FID_NODE_RIGHT);
    launcher.add_region_requirement(nreq);

    RegionRequirement mreq(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr);
    mreq.add_field(FID_META_ROOT);
    launcher.add_region_requirement(mreq);

    Future f = runtime->execute_task(ctx, launcher);
    SearchResult result = f.get_result<SearchResult>();

    if (result.found) {
      outfile << "Found! Result Value: \"" << result.value.data << "\"\n";
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

  // Cleanup Legion resources
  runtime->destroy_logical_region(ctx, node_lr);
  runtime->destroy_field_space(ctx, node_fs);
  runtime->destroy_index_space(ctx, node_is);

  runtime->destroy_logical_region(ctx, meta_lr);
  runtime->destroy_field_space(ctx, meta_fs);
  runtime->destroy_index_space(ctx, meta_is);
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "init");
  }

  {
    TaskVariantRegistrar registrar(INSERT_TASK_ID, "insert");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<insert_task>(registrar, "insert");
  }

  {
    TaskVariantRegistrar registrar(SEARCH_TASK_ID, "search");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SearchResult, search_task>(registrar, "search");
  }

  return Runtime::start(argc, argv);
}