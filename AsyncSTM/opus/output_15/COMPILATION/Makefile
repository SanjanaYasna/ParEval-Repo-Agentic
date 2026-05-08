CXX=g++

# Derive Legion install prefix from Legion_DIR
# (Legion_DIR typically points to <prefix>/lib/cmake/Legion)
ifndef Legion_DIR
$(error Legion_DIR is not set. Please set it to the Legion CMake config directory.)
endif

LEGION_PREFIX=$(abspath $(Legion_DIR)/../../..)

LEGION_INCLUDE=-I$(LEGION_PREFIX)/include
LEGION_LIB=-L$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIB) -llegion -lrealm $(BOOST_LIB) -lpthread -ldl -lz

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
