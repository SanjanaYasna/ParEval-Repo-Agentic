CXX ?= g++

# Required environment variables:
#   Legion_DIR -> directory containing LegionConfig.cmake
#   BOOST_ROOT -> Boost installation prefix
ifndef Legion_DIR
$(error Legion_DIR is not set. Please set it to the path containing LegionConfig.cmake)
endif

ifndef BOOST_ROOT
$(error BOOST_ROOT is not set)
endif

BUILD_DIR := build
PROGRAMS := binary_tree unit_tests concurrency_tests
PROGRAM_PATHS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))
ADDITIONAL_SOURCES :=

LEGION_PREFIX := $(abspath $(Legion_DIR)/../../..)

# Query Legion compile/link flags from CMake package config.
LEGION_CFLAGS := $(shell cmake --find-package \
  -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE \
  -DLegion_DIR=$(Legion_DIR) 2>/dev/null)

LEGION_LIBS := $(shell cmake --find-package \
  -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK \
  -DLegion_DIR=$(Legion_DIR) 2>/dev/null)

# Fallback if cmake --find-package returns nothing.
ifeq ($(strip $(LEGION_CFLAGS)),)
LEGION_CFLAGS := -I$(LEGION_PREFIX)/include
LEGION_LIBS := -L$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib -llegion
endif

# Some Legion installs provide liblegion as a static archive and require Realm explicitly.
LEGION_EXTRA_LIBS :=
ifneq ($(wildcard $(LEGION_PREFIX)/lib/librealm.*),)
LEGION_EXTRA_LIBS += -lrealm
endif

UNAME_S := $(shell uname -s)
SYSTEM_LIBS := -pthread -lz
ifeq ($(UNAME_S),Linux)
SYSTEM_LIBS += -ldl -lrt
endif

BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE) -pthread
LDFLAGS  += $(LEGION_LIBS) $(LEGION_EXTRA_LIBS) $(BOOST_LIB) $(SYSTEM_LIBS)

all: $(BUILD_DIR) $(PROGRAM_PATHS) $(PROGRAMS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PROGRAMS): %: $(BUILD_DIR)/%
	ln -sf $< $@

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(PROGRAMS)

.PHONY: all clean
