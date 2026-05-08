CXX=g++

# Derive the Legion install prefix from Legion_DIR
# Legion_DIR typically points to <prefix>/share/Legion/cmake
LEGION_PREFIX=$(shell cd $(Legion_DIR)/../../.. 2>/dev/null && pwd)
LEGION_INCLUDE=-I$(LEGION_PREFIX)/include
LEGION_LIB=-L$(LEGION_PREFIX)/lib -llegion -lrealm -Wl,-rpath,$(LEGION_PREFIX)/lib

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g -DASTM_LEGION $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIB) $(BOOST_LIB) -lpthread -ldl -lz

ADDITIONAL_SOURCES=
PROGRAMS=binary_tree concurrency_tests unit_tests
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
