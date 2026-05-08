CXX=g++

# Derive Legion install prefix from Legion_DIR
# (Legion_DIR typically points to <prefix>/share/Legion/cmake or <prefix>/lib/cmake/Legion)
LEGION_ROOT=$(abspath $(Legion_DIR)/../../..)

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
