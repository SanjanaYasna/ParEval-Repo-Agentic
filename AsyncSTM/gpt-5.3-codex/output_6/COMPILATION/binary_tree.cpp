////////////////////////////////////////////////////////////////////////////////
//  Translated from HPX/ASTM style to Legion execution model
//  Default mapper assumed (no custom mapping).
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace Legion;

namespace {

constexpr int MAX_TREE_NODES = 1024;
constexpr size_t VALUE_SIZE = 128;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INIT_TREE_TASK_ID,
  INSERT_TASK_ID,
  SEARCH_TASK_ID
};

enum FieldIDs {
  // Node region fields
  FID_KEY = 1,
  FID_LEFT,
  FID_RIGHT,
  FID_USED,
  FID_VALUE,

  // Meta region fields (single element)
  FID_ROOT,
  FID_NEXT_FREE
};

struct FixedValue {
  char data[VALUE_SIZE];
};

struct InsertArgs {
  int key;
  char value[VALUE_SIZE];
};

struct SearchArgs {
  int key;
};

struct SearchResult {
  int found;  // 0/1
  char value[VALUE_SIZE];
};

static inline void copy_cstr(char *dst, size_t dst_size, const std::string &src) {
  std::memset(dst, 0, dst_size);
  std::strncpy(dst, src.c_str(), dst_size - 1);
}

void init_tree_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);

  // Node fields
  FieldAccessor<READ_WRITE, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<READ_WRITE, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<READ_WRITE, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<READ_WRITE, int, 1> used_acc(regions[0], FID_USED);
  FieldAccessor<READ_WRITE, FixedValue, 1> value_acc(regions[0], FID_VALUE);

  Rect<1> node_rect = runtime->get_index_space_domain(
      ctx, regions[0].get_logical_region().get_index_space());

  FixedValue empty{};
  for (PointInRectIterator<1> pir(node_rect); pir(); pir++) {
    key_acc[*pir] = 0;
    left_acc[*pir] = -1;
    right_acc[*pir] = -1;
    used_acc[*pir] = 0;
    value_acc[*pir] = empty;
  }

  // Meta fields
  FieldAccessor<READ_WRITE, int, 1> root_acc(regions[1], FID_ROOT);
  FieldAccessor<READ_WRITE, int, 1> next_free_acc(regions[1], FID_NEXT_FREE);

  Point<1> m(0);
  root_acc[m] = -1;
  next_free_acc[m] = 0;
}

void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context /*ctx*/, Runtime * /*runtime*/) {
  assert(task->arglen == sizeof(InsertArgs));
  assert(regions.size() == 2);

  const InsertArgs &args = *reinterpret_cast<const InsertArgs *>(task->args);

  FieldAccessor<READ_WRITE, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<READ_WRITE, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<READ_WRITE, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<READ_WRITE, int, 1> used_acc(regions[0], FID_USED);
  FieldAccessor<READ_WRITE, FixedValue, 1> value_acc(regions[0], FID_VALUE);

  FieldAccessor<READ_WRITE, int, 1> root_acc(regions[1], FID_ROOT);
  FieldAccessor<READ_WRITE, int, 1> next_free_acc(regions[1], FID_NEXT_FREE);

  Point<1> meta_p(0);
  int root = root_acc[meta_p];
  int next_free = next_free_acc[meta_p];

  auto alloc_node = [&](int key, const char *value) -> int {
    int idx = next_free++;
    Point<1> p(idx);

    key_acc[p] = key;
    left_acc[p] = -1;
    right_acc[p] = -1;
    used_acc[p] = 1;

    FixedValue fv{};
    std::strncpy(fv.data, value, VALUE_SIZE - 1);
    value_acc[p] = fv;
    return idx;
  };

  if (root == -1) {
    int new_root = alloc_node(args.key, args.value);
    root_acc[meta_p] = new_root;
    next_free_acc[meta_p] = next_free;
    return;
  }

  int cur = root;
  while (true) {
    Point<1> pcur(cur);
    int cur_key = key_acc[pcur];

    if (args.key < cur_key) {
      int l = left_acc[pcur];
      if (l == -1) {
        int n = alloc_node(args.key, args.value);
        left_acc[pcur] = n;
        break;
      }
      cur = l;
    } else {
      int r = right_acc[pcur];
      if (r == -1) {
        int n = alloc_node(args.key, args.value);
        right_acc[pcur] = n;
        break;
      }
      cur = r;
    }
  }

  next_free_acc[meta_p] = next_free;
}

SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context /*ctx*/, Runtime * /*runtime*/) {
  assert(task->arglen == sizeof(SearchArgs));
  assert(regions.size() == 2);

  const SearchArgs &args = *reinterpret_cast<const SearchArgs *>(task->args);

  FieldAccessor<READ_ONLY, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<READ_ONLY, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<READ_ONLY, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<READ_ONLY, int, 1> used_acc(regions[0], FID_USED);
  FieldAccessor<READ_ONLY, FixedValue, 1> value_acc(regions[0], FID_VALUE);

  FieldAccessor<READ_ONLY, int, 1> root_acc(regions[1], FID_ROOT);

  SearchResult result{};
  result.found = 0;
  std::memset(result.value, 0, VALUE_SIZE);

  Point<1> meta_p(0);
  int cur = root_acc[meta_p];

  while (cur != -1) {
    Point<1> pcur(cur);
    if (!used_acc[pcur]) break;

    int cur_key = key_acc[pcur];
    if (args.key == cur_key) {
      result.found = 1;
      FixedValue fv = value_acc[pcur];
      std::strncpy(result.value, fv.data, VALUE_SIZE - 1);
      return result;
    } else if (args.key < cur_key) {
      cur = left_acc[pcur];
    } else {
      cur = right_acc[pcur];
    }
  }

  return result;
}

void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime) {
  // Create tree node region
  Rect<1> node_bounds(0, MAX_TREE_NODES - 1);
  IndexSpace node_is = runtime->create_index_space(ctx, node_bounds);
  FieldSpace node_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, node_fs);
    alloc.allocate_field(sizeof(int), FID_KEY);
    alloc.allocate_field(sizeof(int), FID_LEFT);
    alloc.allocate_field(sizeof(int), FID_RIGHT);
    alloc.allocate_field(sizeof(int), FID_USED);
    alloc.allocate_field(sizeof(FixedValue), FID_VALUE);
  }
  LogicalRegion node_lr = runtime->create_logical_region(ctx, node_is, node_fs);

  // Create metadata region (single element): root and next_free
  Rect<1> meta_bounds(0, 0);
  IndexSpace meta_is = runtime->create_index_space(ctx, meta_bounds);
  FieldSpace meta_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, meta_fs);
    alloc.allocate_field(sizeof(int), FID_ROOT);
    alloc.allocate_field(sizeof(int), FID_NEXT_FREE);
  }
  LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

  // Initialize tree storage
  {
    TaskLauncher init_launcher(INIT_TREE_TASK_ID, TaskArgument(nullptr, 0));

    RegionRequirement rr_nodes(node_lr, READ_WRITE, EXCLUSIVE, node_lr);
    rr_nodes.add_field(FID_KEY);
    rr_nodes.add_field(FID_LEFT);
    rr_nodes.add_field(FID_RIGHT);
    rr_nodes.add_field(FID_USED);
    rr_nodes.add_field(FID_VALUE);
    init_launcher.add_region_requirement(rr_nodes);

    RegionRequirement rr_meta(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr);
    rr_meta.add_field(FID_ROOT);
    rr_meta.add_field(FID_NEXT_FREE);
    init_launcher.add_region_requirement(rr_meta);

    runtime->execute_task(ctx, init_launcher).get_void_result();
  }

  // Open output file (same behavior as original)
  std::ofstream outfile("binary_tree.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
    return;
  }

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
      {80, "Right-Right Grandchild"}};

  outfile << "Inserting Data:\n";
  for (const auto &entry : data) {
    outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")\n";

    InsertArgs args{};
    args.key = entry.id;
    copy_cstr(args.value, VALUE_SIZE, entry.val);

    TaskLauncher ins_launcher(INSERT_TASK_ID, TaskArgument(&args, sizeof(args)));

    RegionRequirement rr_nodes(node_lr, READ_WRITE, EXCLUSIVE, node_lr);
    rr_nodes.add_field(FID_KEY);
    rr_nodes.add_field(FID_LEFT);
    rr_nodes.add_field(FID_RIGHT);
    rr_nodes.add_field(FID_USED);
    rr_nodes.add_field(FID_VALUE);
    ins_launcher.add_region_requirement(rr_nodes);

    RegionRequirement rr_meta(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr);
    rr_meta.add_field(FID_ROOT);
    rr_meta.add_field(FID_NEXT_FREE);
    ins_launcher.add_region_requirement(rr_meta);

    runtime->execute_task(ctx, ins_launcher).get_void_result();
  }
  outfile << "\n";

  outfile << "Running Search Tests:\n";
  std::vector<int> search_keys = {99, 10, 50, 20, 60, 80, 45};

  for (int key : search_keys) {
    outfile << "  Search(Key: " << key << ") -> ";

    SearchArgs args{key};
    TaskLauncher search_launcher(SEARCH_TASK_ID, TaskArgument(&args, sizeof(args)));

    RegionRequirement rr_nodes(node_lr, READ_ONLY, EXCLUSIVE, node_lr);
    rr_nodes.add_field(FID_KEY);
    rr_nodes.add_field(FID_LEFT);
    rr_nodes.add_field(FID_RIGHT);
    rr_nodes.add_field(FID_USED);
    rr_nodes.add_field(FID_VALUE);
    search_launcher.add_region_requirement(rr_nodes);

    RegionRequirement rr_meta(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr);
    rr_meta.add_field(FID_ROOT);
    search_launcher.add_region_requirement(rr_meta);

    Future f = runtime->execute_task(ctx, search_launcher);
    SearchResult sr = f.get_result<SearchResult>();

    if (sr.found) {
      outfile << "Found! Result Value: \"" << sr.value << "\"\n";
    } else {
      outfile << "Not Found.\n";
    }
  }

  outfile.close();

  // Cleanup regions/spaces
  runtime->destroy_logical_region(ctx, node_lr);
  runtime->destroy_field_space(ctx, node_fs);
  runtime->destroy_index_space(ctx, node_is);

  runtime->destroy_logical_region(ctx, meta_lr);
  runtime->destroy_field_space(ctx, meta_fs);
  runtime->destroy_index_space(ctx, meta_is);
}

}  // namespace

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(INIT_TREE_TASK_ID, "init_tree_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<init_tree_task>(registrar, "init_tree_task");
  }

  {
    TaskVariantRegistrar registrar(INSERT_TASK_ID, "insert_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<insert_task>(registrar, "insert_task");
  }

  {
    TaskVariantRegistrar registrar(SEARCH_TASK_ID, "search_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SearchResult, search_task>(registrar, "search_task");
  }

  return Runtime::start(argc, argv);
}
