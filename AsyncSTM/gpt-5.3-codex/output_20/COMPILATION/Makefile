CXX ?= g++

# Legion CMake package location must be provided by environment.
ifndef Legion_DIR
$(error Legion_DIR is not set. Please export Legion_DIR to Legion's CMake config directory)
endif

# Resolve Legion compile/link flags through CMake package config.
LEGION_CFLAGS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=GNU \
	-DLANGUAGE=CXX \
	-DMODE=COMPILE \
	-DLegion_DIR="$(Legion_DIR)" 2>/dev/null)

LEGION_LIBS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=GNU \
	-DLANGUAGE=CXX \
	-DMODE=LINK \
	-DLegion_DIR="$(Legion_DIR)" 2>/dev/null)

# Include Boost paths
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS += -std=c++17 -I. -g $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS := $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
DIRECTORIES := build

all: directories $(PROGRAMS)

.PHONY: all directories clean
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

%: %.cpp | $(DIRECTORIES)/
	$(CXX) -DASTM_LEGION $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o build/$@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
