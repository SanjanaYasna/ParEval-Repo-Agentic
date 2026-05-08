CXX=g++

# Derive Legion installation root from Legion_DIR (cmake config directory)
# Legion_DIR is typically <prefix>/share/Legion/cmake or <prefix>/lib/cmake/Legion
# Users can override by setting LEGION_ROOT directly
ifndef LEGION_ROOT
    LEGION_ROOT:=$(abspath $(Legion_DIR)/../../..)
endif

LEGION_INCLUDE=-I$(LEGION_ROOT)/include
LEGION_LIB=-L$(LEGION_ROOT)/lib -llegion -lrealm -Wl,-rpath,$(LEGION_ROOT)/lib

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIB) $(BOOST_LIB) -lpthread -ldl -lz

ADDITIONAL_SOURCES=
PROGRAMS=binary_tree unit_tests concurrency_tests
DIRECTORIES=build

all: directories $(PROGRAMS)

.PHONY: directories
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

% : %.cpp
	$(CXX) -DASTM_LEGION $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o build/$@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
