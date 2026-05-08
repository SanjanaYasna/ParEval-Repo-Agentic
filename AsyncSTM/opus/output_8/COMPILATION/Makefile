CXX=g++

# Derive Legion install prefix from Legion_DIR
# Legion_DIR typically points to <prefix>/share/Legion/cmake
LEGION_PREFIX=$(shell cd $(Legion_DIR)/../../.. 2>/dev/null && pwd)

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g -I$(LEGION_PREFIX)/include $(BOOST_INCLUDE)
LDFLAGS=-L$(LEGION_PREFIX)/lib -llegion -lrealm $(BOOST_LIB) -Wl,-rpath,$(LEGION_PREFIX)/lib -lpthread -ldl -lz

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
