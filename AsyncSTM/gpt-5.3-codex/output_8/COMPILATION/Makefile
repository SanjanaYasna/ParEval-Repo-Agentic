CXX ?= g++

# Require Legion CMake package location from environment.
ifndef Legion_DIR
$(error Legion_DIR is not set. Please export Legion_DIR to the directory containing LegionConfig.cmake)
endif

# Detect compiler ID for cmake --find-package
CMAKE_COMPILER_ID := $(shell $(CXX) --version | head -n1 | grep -qi clang && echo Clang || echo GNU)

# Query Legion compile/link flags from CMake package config.
LEGION_CFLAGS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=$(CMAKE_COMPILER_ID) \
	-DLANGUAGE=CXX \
	-DMODE=COMPILE \
	-DLegion_DIR=$(Legion_DIR) 2>/dev/null)

LEGION_LIBS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=$(CMAKE_COMPILER_ID) \
	-DLANGUAGE=CXX \
	-DMODE=LINK \
	-DLegion_DIR=$(Legion_DIR) 2>/dev/null)

# Fallback include path if cmake --find-package does not return compile flags.
LEGION_PREFIX := $(abspath $(Legion_DIR)/../../..)
ifeq ($(strip $(LEGION_CFLAGS)),)
LEGION_CFLAGS := -I$(LEGION_PREFIX)/include
endif

# Boost paths (provided by environment)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION -pthread $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS += -pthread $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
DIRECTORIES := build

all: directories $(PROGRAMS)

.PHONY: all clean directories
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

%: %.cpp
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o build/$@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
