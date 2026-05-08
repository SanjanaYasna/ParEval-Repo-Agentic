CXX=g++

# Derive Legion install prefix from Legion_DIR (cmake config directory)
# Works for common layouts: <prefix>/share/Legion/cmake or <prefix>/lib/cmake/Legion
LEGION_PREFIX=$(shell cd $(Legion_DIR)/../../.. 2>/dev/null && pwd)

LEGION_INCLUDE=-I$(LEGION_PREFIX)/include
LEGION_LIBS=-L$(LEGION_PREFIX)/lib -llegion -lrealm

# Include Boost paths
BOOST_INCLUDE=-I$(BOOST_ROOT)/include
BOOST_LIB=-L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS=$(LEGION_LIBS) $(BOOST_LIB) -Wl,-rpath,$(LEGION_PREFIX)/lib -lpthread -ldl -lz

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
