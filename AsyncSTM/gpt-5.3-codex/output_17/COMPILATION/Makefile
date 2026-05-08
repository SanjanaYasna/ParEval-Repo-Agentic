CXX ?= g++

# Detect compiler id for cmake --find-package.
LEGION_COMPILER_ID := GNU
CXX_VERSION_LINE := $(shell $(CXX) --version 2>/dev/null | head -n 1 | tr A-Z a-z)
ifneq (,$(findstring clang,$(CXX_VERSION_LINE)))
LEGION_COMPILER_ID := Clang
endif

# Accept Legion location from either LEGION_DIR or LEGION_ROOT.
# If LEGION_DIR is not set, try common LEGION_ROOT install layouts.
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_DIR_CANDIDATES := \
	$(LEGION_ROOT)/lib/cmake/Legion \
	$(LEGION_ROOT)/lib64/cmake/Legion \
	$(LEGION_ROOT)/share/Legion/cmake
LEGION_DIR_FROM_ROOT := $(firstword $(foreach d,$(LEGION_DIR_CANDIDATES),$(if $(wildcard $(d)/LegionConfig.cmake),$(d),)))
ifneq ($(strip $(LEGION_DIR_FROM_ROOT)),)
LEGION_DIR ?= $(LEGION_DIR_FROM_ROOT)
endif
endif

LEGION_FIND_BASE := cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=$(LEGION_COMPILER_ID) \
	-DLANGUAGE=CXX

ifneq ($(strip $(LEGION_DIR)),)
LEGION_CFLAGS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=COMPILE \
	-DLegion_DIR=$(LEGION_DIR) 2>/dev/null)

LEGION_LIBS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=LINK \
	-DLegion_DIR=$(LEGION_DIR) 2>/dev/null)
else
LEGION_CFLAGS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=COMPILE 2>/dev/null)

LEGION_LIBS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=LINK 2>/dev/null)
endif

# Retry via CMAKE_PREFIX_PATH when Legion_DIR discovery did not produce flags.
ifeq ($(strip $(LEGION_CFLAGS)$(LEGION_LIBS)),)
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_CFLAGS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=COMPILE \
	-DCMAKE_PREFIX_PATH=$(LEGION_ROOT) 2>/dev/null)

LEGION_LIBS := $(shell $(LEGION_FIND_BASE) \
	-DMODE=LINK \
	-DCMAKE_PREFIX_PATH=$(LEGION_ROOT) 2>/dev/null)
endif
endif

# Fallback to pkg-config when CMake package discovery does not return flags.
ifeq ($(strip $(LEGION_CFLAGS)$(LEGION_LIBS)),)
LEGION_CFLAGS := $(shell pkg-config --cflags legion 2>/dev/null)
LEGION_LIBS := $(shell pkg-config --libs legion 2>/dev/null)
endif

# Final manual fallback for common Legion installs.
ifeq ($(strip $(LEGION_CFLAGS)),)
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_CFLAGS := -I$(LEGION_ROOT)/include
endif
endif

ifeq ($(strip $(LEGION_LIBS)),)
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_LIBS := \
	-L$(LEGION_ROOT)/lib \
	-L$(LEGION_ROOT)/lib64 \
	-Wl,-rpath,$(LEGION_ROOT)/lib \
	-Wl,-rpath,$(LEGION_ROOT)/lib64 \
	-llegion -lrealm
endif
endif

# Include Boost paths only when BOOST_ROOT is set.
BOOST_INCLUDE :=
BOOST_LIB :=
ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS += $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
PROGRAM_BINS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean directories $(PROGRAMS)

all: directories $(PROGRAM_BINS)

directories:
	mkdir -p $(BUILD_DIR)

# Optional convenience targets: `make binary_tree`, etc.
$(PROGRAMS): %: $(BUILD_DIR)/%

$(BUILD_DIR)/%: %.cpp
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
