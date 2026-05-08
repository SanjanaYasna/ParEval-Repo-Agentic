CXX=g++

# Legion_DIR points to the cmake config directory, typically:
#   <prefix>/share/Legion/cmake  or  <prefix>/lib/cmake/Legion
# Derive the installation prefix by going up three directory levels.
ifndef Legion_DIR
$(error Legion_DIR is not set. Point it to the directory containing LegionConfig.cmake)
endif
LEGION_ROOT := $(abspath $(Legion_DIR)/../../..)

LEGION_INCLUDE=-I$(LEGION_ROOT)/include
LEGION_LIB_DIR=$(LEGION_ROOT)/lib
LEGION_LIBS=-L$(LEGION_LIB_DIR) -llegion -lrealm -Wl,-rpath,$(LEGION_LIB_DIR)

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIBS) $(BOOST_LIB) -lpthread -ldl -lz

ADDITIONAL_SOURCES=
PROGRAMS=binary_tree concurrency_tests unit_tests
DIRECTORIES=build

all: directories $(PROGRAMS)

.PHONY: directories clean
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o build/$@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
