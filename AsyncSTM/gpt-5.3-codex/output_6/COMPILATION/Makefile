CXX ?= g++
CXXSTD ?= -std=c++17

# Optional: override if not using GCC
CMAKE_COMPILER_ID ?= GNU

# Legion can be provided via LEGION_DIR or LEGION_ROOT.
LEGION_PREFIX ?= $(if $(LEGION_DIR),$(LEGION_DIR),$(LEGION_ROOT))
CMAKE_PREFIX_ARG := $(if $(LEGION_PREFIX),-DCMAKE_PREFIX_PATH=$(LEGION_PREFIX),)

# Query Legion compile/link flags via CMake package config
LEGION_CFLAGS := $(strip $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=$(CMAKE_COMPILER_ID) \
	-DLANGUAGE=CXX \
	-DMODE=COMPILE \
	$(CMAKE_PREFIX_ARG) 2>/dev/null))

LEGION_LIBS := $(strip $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=$(CMAKE_COMPILER_ID) \
	-DLANGUAGE=CXX \
	-DMODE=LINK \
	$(CMAKE_PREFIX_ARG) 2>/dev/null))

# Fallback to pkg-config if CMake package lookup does not succeed.
ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(strip $(shell pkg-config --cflags legion 2>/dev/null))
endif

ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(strip $(shell pkg-config --libs legion 2>/dev/null))
endif

ifdef BOOST_ROOT
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB :=
endif

CXXFLAGS += $(CXXSTD) -I. -g -pthread $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += -pthread $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAM_NAMES := binary_tree unit_tests concurrency_tests
PROGRAMS := $(addprefix build/,$(PROGRAM_NAMES))
DIRECTORIES := build

.PHONY: all clean directories
all: directories $(PROGRAMS)

directories: $(DIRECTORIES)

$(DIRECTORIES):
	mkdir -p $@

build/%: %.cpp | build
	$(CXX) -DASTM_LEGION $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf build
