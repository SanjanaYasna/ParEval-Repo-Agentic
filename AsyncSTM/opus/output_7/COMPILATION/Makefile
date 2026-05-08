CXX=g++

# Derive Legion install prefix from Legion_DIR
# Legion_DIR is expected to point to <prefix>/lib/cmake/Legion
LEGION_PREFIX ?= $(abspath $(Legion_DIR)/../../..)

LEGION_INCLUDE=-I$(LEGION_PREFIX)/include
LEGION_LIBDIR=-L$(LEGION_PREFIX)/lib
LEGION_LIBS=-llegion -lrealm
LEGION_RPATH=-Wl,-rpath,$(LEGION_PREFIX)/lib

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIBDIR) $(LEGION_LIBS) $(LEGION_RPATH) $(BOOST_LIB) -lpthread -ldl -lz

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
