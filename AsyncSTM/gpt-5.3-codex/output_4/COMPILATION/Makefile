CXX ?= g++

ifndef Legion_DIR
$(error Legion_DIR is not set. Please export Legion_DIR to the directory containing LegionConfig.cmake)
endif

ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB :=
endif

LEGION_PREFIX_GUESS := $(abspath $(Legion_DIR)/../../..)
LEGION_ALT_PREFIX_GUESS := $(abspath $(Legion_DIR)/../..)

CXX_VERSION_LINE := $(shell $(CXX) --version 2>/dev/null | head -n 1 | tr '[:upper:]' '[:lower:]')
ifneq ($(findstring clang,$(CXX_VERSION_LINE)),)
LEGION_COMPILER_IDS := AppleClang Clang GNU
else
LEGION_COMPILER_IDS := GNU Clang AppleClang
endif

define query_legion_flags
$(strip $(shell \
for cid in $(LEGION_COMPILER_IDS); do \
  out="$$(cmake --find-package \
    -DNAME=Legion \
    -DCOMPILER_ID=$$cid \
    -DLANGUAGE=CXX \
    -DMODE=$(1) \
    -DLegion_DIR="$(Legion_DIR)" 2>/dev/null)"; \
  if [ -n "$$out" ]; then echo "$$out"; exit 0; fi; \
done; \
for cid in $(LEGION_COMPILER_IDS); do \
  out="$$(cmake --find-package \
    -DNAME=Legion \
    -DCOMPILER_ID=$$cid \
    -DLANGUAGE=CXX \
    -DMODE=$(1) \
    -DCMAKE_PREFIX_PATH="$(Legion_DIR);$(LEGION_PREFIX_GUESS);$(LEGION_ALT_PREFIX_GUESS)" 2>/dev/null)"; \
  if [ -n "$$out" ]; then echo "$$out"; exit 0; fi; \
done))
endef

LEGION_COMPILE_FLAGS := $(call query_legion_flags,COMPILE)
LEGION_LINK_FLAGS := $(call query_legion_flags,LINK)

ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
ifneq ($(wildcard $(LEGION_PREFIX_GUESS)/include/legion.h),)
LEGION_COMPILE_FLAGS := -I$(LEGION_PREFIX_GUESS)/include
else ifneq ($(wildcard $(LEGION_ALT_PREFIX_GUESS)/include/legion.h),)
LEGION_COMPILE_FLAGS := -I$(LEGION_ALT_PREFIX_GUESS)/include
else
$(error Failed to query Legion compile flags via CMake. Check Legion_DIR=$(Legion_DIR))
endif
endif

LEGION_LIBDIR_GUESS :=
ifneq ($(wildcard $(LEGION_PREFIX_GUESS)/lib/liblegion*),)
LEGION_LIBDIR_GUESS := $(LEGION_PREFIX_GUESS)/lib
else ifneq ($(wildcard $(LEGION_ALT_PREFIX_GUESS)/lib/liblegion*),)
LEGION_LIBDIR_GUESS := $(LEGION_ALT_PREFIX_GUESS)/lib
endif

ifeq ($(strip $(LEGION_LINK_FLAGS)),)
ifneq ($(strip $(LEGION_LIBDIR_GUESS)),)
LEGION_LINK_FLAGS := -L$(LEGION_LIBDIR_GUESS) -Wl,-rpath,$(LEGION_LIBDIR_GUESS) -llegion
else
$(error Failed to query Legion link flags via CMake. Check Legion_DIR=$(Legion_DIR))
endif
endif

CXXFLAGS += -std=c++17 -I. -g $(BOOST_INCLUDE) $(LEGION_COMPILE_FLAGS)
LDFLAGS += $(BOOST_LIB) $(LEGION_LINK_FLAGS)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean directories
all: directories $(TARGETS)

directories: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) -DASTM_LEGION $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
