CXX ?= g++

# Legion CMake package directory (exported in environment as Legion_DIR)
LEGION_CMAKE_DIR ?= $(Legion_DIR)
ifeq ($(strip $(LEGION_CMAKE_DIR)),)
$(error Legion_DIR is not set. Please export Legion_DIR (e.g. <install-prefix>/lib/cmake/Legion))
endif

# Infer Legion install prefix from Legion_DIR
LEGION_PREFIX ?= $(abspath $(LEGION_CMAKE_DIR)/../../..)

# Handle lib vs lib64 installs
ifneq ("$(wildcard $(LEGION_PREFIX)/lib64)","")
LEGION_LIB_PATH := $(LEGION_PREFIX)/lib64
else
LEGION_LIB_PATH := $(LEGION_PREFIX)/lib
endif

# Includes / libs
LEGION_INCLUDE := -I$(LEGION_PREFIX)/include
LEGION_LIBS    := -L$(LEGION_LIB_PATH) -Wl,-rpath,$(LEGION_LIB_PATH) \
                  -llegion -lrealm -lpthread -ldl -lrt -lz

BOOST_INCLUDE  := -I$(BOOST_ROOT)/include
BOOST_LIBS     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS += -std=c++17 -O2 -g -I. $(LEGION_INCLUDE) $(BOOST_INCLUDE) -DASTM_LEGION
LDFLAGS  += $(LEGION_LIBS) $(BOOST_LIBS)

PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.cpp astm.hpp astm_config.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
