CXX=g++

# Derive Legion install prefix from Legion_DIR
# (Legion_DIR typically points to <prefix>/lib/cmake/Legion)
LEGION_ROOT ?= $(abspath $(Legion_DIR)/../../..)

LEGION_CFLAGS=-I$(LEGION_ROOT)/include
LEGION_LIBS=-L$(LEGION_ROOT)/lib -llegion -lrealm -Wl,-rpath,$(LEGION_ROOT)/lib

CXXFLAGS+=-std=c++17 -I. -g $(LEGION_CFLAGS)
LDFLAGS=$(LEGION_LIBS) -lpthread -ldl -lz

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
