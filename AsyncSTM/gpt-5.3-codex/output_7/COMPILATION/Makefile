CXX ?= g++

PROGRAMS := binary_tree unit_tests concurrency_tests
BIN_DIR := build
BINARIES := $(addprefix $(BIN_DIR)/,$(PROGRAMS))

ADDITIONAL_SOURCES :=

LEGION_ROOT ?=
LEGION_DIR ?= $(LEGION_ROOT)

# Boost paths
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

# Discover Legion flags from CMake package config (default mapper; no custom mapper)
LEGION_FIND_PACKAGE := cmake --find-package -DNAME=Legion -DLANGUAGE=CXX -DCOMPILER_ID=GNU

LEGION_CMAKE_DIR := $(if $(wildcard $(LEGION_DIR)/lib/cmake/Legion/LegionConfig.cmake),$(LEGION_DIR)/lib/cmake/Legion,$(if $(wildcard $(LEGION_DIR)/share/cmake/Legion/LegionConfig.cmake),$(LEGION_DIR)/share/cmake/Legion,$(LEGION_DIR)))
LEGION_CFLAGS := $(shell CMAKE_PREFIX_PATH=$(LEGION_DIR) $(LEGION_FIND_PACKAGE) -DLegion_DIR=$(LEGION_CMAKE_DIR) -DMODE=COMPILE 2>/dev/null)
LEGION_LIBS := $(shell CMAKE_PREFIX_PATH=$(LEGION_DIR) $(LEGION_FIND_PACKAGE) -DLegion_DIR=$(LEGION_CMAKE_DIR) -DMODE=LINK 2>/dev/null)

LEGION_INCLUDEDIR := $(if $(wildcard $(LEGION_DIR)/include),$(LEGION_DIR)/include,$(LEGION_DIR))
LEGION_LIBDIR := $(if $(wildcard $(LEGION_DIR)/lib),$(LEGION_DIR)/lib,$(LEGION_DIR)/lib64)

ifeq ($(strip $(LEGION_CFLAGS)),)
LEGION_CFLAGS := -I$(LEGION_INCLUDEDIR) -I$(LEGION_INCLUDEDIR)/legion
endif

ifeq ($(strip $(LEGION_LIBS)),)
LEGION_LIBS := -L$(LEGION_LIBDIR) -Wl,-rpath,$(LEGION_LIBDIR) -llegion -lrealm -lpthread -ldl -lrt
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE) -pthread
LDFLAGS += $(LEGION_LIBS) $(BOOST_LIB) -pthread

.PHONY: all clean
all: $(BIN_DIR) $(BINARIES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
