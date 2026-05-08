#include "legion.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace Legion;

//------------------------------------------------------------------------------
// Legion task/field IDs
//------------------------------------------------------------------------------

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INIT_TASK_ID,
  INSERT_TASK_ID,
  SEARCH_TASK_ID
};

enum FieldIDs {
  FID_KEY = 1,
  FID_LEFT,
  FID_RIGHT,
  FID_VALUE,
  FID_ROOT,
  FID_COUNT
};

//------------------------------------------------------------------------------
// Tree storage types
//------------------------------------------------------------------------------

static constexpr int MAX_NODES = 1024;
static constexpr size_t MAX_VALUE_LEN = 128;

struct FixedString {
  char data[MAX_VALUE_LEN];
};

static FixedString make_fixed_string(const std::string &s) {
  FixedString out{};
  std::strncpy(out.data, s.c_str(), MAX_VALUE_LEN - 1);
  out.data[MAX_VALUE_LEN - 1] = '\0';
  return out;
}

static std::string to_std_string(const FixedString &s) {
  return std::string(s.data);
}

struct InsertArgs {
  int key;
  FixedString value;
};

struct SearchArgs {
  int key;
};

struct SearchResult {
  bool found;
  FixedString value;
};

//------------------------------------------------------------------------------
// Tasks
//------------------------------------------------------------------------------

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
  FieldAccessor<WRITE_DISCARD, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<WRITE_DISCARD, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<WRITE_DISCARD, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<WRITE_DISCARD, FixedString, 1> value_acc(regions[0], FID_VALUE);

  FieldAccessor<WRITE_DISCARD, int, 1> root_acc(regions[1], FID_ROOT);
  FieldAccessor<WRITE_DISCARD, int, 1> count_acc(regions[1], FID_COUNT);

  Rect<1> nodes_rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  FixedString empty = make_fixed_string("");

  for (PointInRectIterator<1> it(nodes_rect); it(); it++) {
    key_acc[*it] = std::numeric_limits<int>::min();
    left_acc[*it] = -1;
    right_acc[*it] = -1;
    value_acc[*it] = empty;
  }

  Point<1> meta_p(0);
  root_acc[meta_p] = -1;
  count_acc[meta_p] = 0;
}

void insert_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context, Runtime *) {
  const InsertArgs &args = *reinterpret_cast<const InsertArgs *>(task->args);

  FieldAccessor<READ_WRITE, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<READ_WRITE, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<READ_WRITE, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<READ_WRITE, FixedString, 1> value_acc(regions[0], FID_VALUE);

  FieldAccessor<READ_WRITE, int, 1> root_acc(regions[1], FID_ROOT);
  FieldAccessor<READ_WRITE, int, 1> count_acc(regions[1], FID_COUNT);

  Point<1> meta_p(0);
  int root = root_acc[meta_p];
  int count = count_acc[meta_p];

  auto allocate_node = [&](int idx, int key, const FixedString &value) {
    Point<1> p(idx);
    key_acc[p] = key;
    left_acc[p] = -1;
    right_acc[p] = -1;
    value_acc[p] = value;
  };

  if (count >= MAX_NODES) {
    // Capacity reached; ignore insert to keep execution safe.
    return;
  }

  if (root == -1) {
    int idx = count++;
    allocate_node(idx, args.key, args.value);
    root = idx;
    root_acc[meta_p] = root;
    count_acc[meta_p] = count;
    return;
  }

  int cur = root;
  while (true) {
    Point<1> cur_p(cur);
    const int cur_key = key_acc[cur_p];

    if (args.key < cur_key) {
      int left_idx = left_acc[cur_p];
      if (left_idx == -1) {
        int idx = count++;
        allocate_node(idx, args.key, args.value);
        left_acc[cur_p] = idx;
        break;
      } else {
        cur = left_idx;
      }
    } else {
      int right_idx = right_acc[cur_p];
      if (right_idx == -1) {
        int idx = count++;
        allocate_node(idx, args.key, args.value);
        right_acc[cur_p] = idx;
        break;
      } else {
        cur = right_idx;
      }
    }

    if (count >= MAX_NODES) break;
  }

  count_acc[meta_p] = count;
}

SearchResult search_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context, Runtime *) {
  const SearchArgs &args = *reinterpret_cast<const SearchArgs *>(task->args);

  FieldAccessor<READ_ONLY, int, 1> key_acc(regions[0], FID_KEY);
  FieldAccessor<READ_ONLY, int, 1> left_acc(regions[0], FID_LEFT);
  FieldAccessor<READ_ONLY, int, 1> right_acc(regions[0], FID_RIGHT);
  FieldAccessor<READ_ONLY, FixedString, 1> value_acc(regions[0], FID_VALUE);

  FieldAccessor<READ_ONLY, int, 1> root_acc(regions[1], FID_ROOT);

  SearchResult result{};
  result.found = false;
  result.value = make_fixed_string("");

  int cur = root_acc[Point<1>(0)];
  while (cur != -1) {
    Point<1> p(cur);
    const int k = key_acc[p];

    if (args.key == k) {
      result.found = true;
      result.value = value_acc[p];
      return result;
    } else if (args.key < k) {
      cur = left_acc[p];
    } else {
      cur = right_acc[p];
    }
  }

  return result;
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
  // Create node region
  Rect<1> nodes_rect(0, MAX_NODES - 1);
  IndexSpace nodes_is = runtime->create_index_space(ctx, nodes_rect);
  FieldSpace nodes_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, nodes_fs);
    alloc.allocate_field(sizeof(int), FID_KEY);
    alloc.allocate_field(sizeof(int), FID_LEFT);
    alloc.allocate_field(sizeof(int), FID_RIGHT);
    alloc.allocate_field(sizeof(FixedString), FID_VALUE);
  }
  LogicalRegion nodes_lr = runtime->create_logical_region(ctx, nodes_is, nodes_fs);

  // Create metadata region (single element: root and node count)
  Rect<1> meta_rect(0, 0);
  IndexSpace meta_is = runtime->create_index_space(ctx, meta_rect);
  FieldSpace meta_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, meta_fs);
    alloc.allocate_field(sizeof(int), FID_ROOT);
    alloc.allocate_field(sizeof(int), FID_COUNT);
  }
  LogicalRegion meta_lr = runtime->create_logical_region(ctx, meta_is, meta_fs);

  // Initialize regions
  {
    TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(nullptr, 0));

    init_launcher.add_region_requirement(
        RegionRequirement(nodes_lr, WRITE_DISCARD, EXCLUSIVE, nodes_lr));
    init_launcher.add_field(0, FID_KEY);
    init_launcher.add_field(0, FID_LEFT);
    init_launcher.add_field(0, FID_RIGHT);
    init_launcher.add_field(0, FID_VALUE);

    init_launcher.add_region_requirement(
        RegionRequirement(meta_lr, WRITE_DISCARD, EXCLUSIVE, meta_lr));
    init_launcher.add_field(1, FID_ROOT);
    init_launcher.add_field(1, FID_COUNT);

    runtime->execute_task(ctx, init_launcher).wait();
  }

  std::ofstream outfile("binary_tree.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
    // Cleanup before returning
    runtime->destroy_logical_region(ctx, nodes_lr);
    runtime->destroy_field_space(ctx, nodes_fs);
    runtime->destroy_index_space(ctx, nodes_is);
    runtime->destroy_logical_region(ctx, meta_lr);
    runtime->destroy_field_space(ctx, meta_fs);
    runtime->destroy_index_space(ctx, meta_is);
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
    outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")"
            << std::endl;

    InsertArgs args{entry.id, make_fixed_string(entry.val)};
    TaskLauncher insert_launcher(INSERT_TASK_ID, TaskArgument(&args, sizeof(args)));

    insert_launcher.add_region_requirement(
        RegionRequirement(nodes_lr, READ_WRITE, EXCLUSIVE, nodes_lr));
    insert_launcher.add_field(0, FID_KEY);
    insert_launcher.add_field(0, FID_LEFT);
    insert_launcher.add_field(0, FID_RIGHT);
    insert_launcher.add_field(0, FID_VALUE);

    insert_launcher.add_region_requirement(
        RegionRequirement(meta_lr, READ_WRITE, EXCLUSIVE, meta_lr));
    insert_launcher.add_field(1, FID_ROOT);
    insert_launcher.add_field(1, FID_COUNT);

    runtime->execute_task(ctx, insert_launcher).wait();
  }
  outfile << "\n";

  outfile << "Running Search Tests:\n";
  auto perform_search = [&](int search_key) {
    SearchArgs args{search_key};
    TaskLauncher search_launcher(SEARCH_TASK_ID, TaskArgument(&args, sizeof(args)));

    search_launcher.add_region_requirement(
        RegionRequirement(nodes_lr, READ_ONLY, EXCLUSIVE, nodes_lr));
    search_launcher.add_field(0, FID_KEY);
    search_launcher.add_field(0, FID_LEFT);
    search_launcher.add_field(0, FID_RIGHT);
    search_launcher.add_field(0, FID_VALUE);

    search_launcher.add_region_requirement(
        RegionRequirement(meta_lr, READ_ONLY, EXCLUSIVE, meta_lr));
    search_launcher.add_field(1, FID_ROOT);

    Future f = runtime->execute_task(ctx, search_launcher);
    SearchResult res = f.get_result<SearchResult>();

    outfile << "  Search(Key: " << search_key << ") -> ";
    if (res.found) {
      outfile << "Found! Result Value: \"" << to_std_string(res.value) << "\"\n";
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

  runtime->destroy_logical_region(ctx, nodes_lr);
  runtime->destroy_field_space(ctx, nodes_fs);
  runtime->destroy_index_space(ctx, nodes_is);

  runtime->destroy_logical_region(ctx, meta_lr);
  runtime->destroy_field_space(ctx, meta_fs);
  runtime->destroy_index_space(ctx, meta_is);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "init_task");
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
