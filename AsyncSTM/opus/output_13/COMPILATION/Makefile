CXX=g++

# Derive the Legion installation prefix from Legion_DIR
# Legion_DIR typically points to <prefix>/share/Legion/cmake or <prefix>/lib/cmake/Legion
LEGION_ROOT ?= $(shell cmake -DLegion_DIR=$(Legion_DIR) -P /dev/stdin <<< \
	"find_package(Legion REQUIRED CONFIG); get_filename_component(_p \"\$${Legion_DIR}/../../..\" ABSOLUTE); message(\$${_p})" 2>&1 | tail -1)
# Fallback: allow the user to set LEGION_ROOT directly
LEGION_INCLUDE=-I$(LEGION_ROOT)/include
LEGION_LIB=-L$(LEGION_ROOT)/lib -llegion -lrealm

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIB) $(BOOST_LIB) -Wl,-rpath,$(LEGION_ROOT)/lib -lpthread -ldl -lz

ADDITIONAL_SOURCES=
PROGRAMS=binary_tree unit_tests concurrency_tests
DIRECTORIES=build

all: directories $(PROGRAMS)

.PHONY: directories
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o build/$@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
