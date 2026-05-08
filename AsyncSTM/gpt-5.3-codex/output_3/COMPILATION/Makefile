CXX ?= g++

# Accept either Legion_DIR or LEGION_DIR from environment.
ifndef Legion_DIR
ifneq ($(LEGION_DIR),)
Legion_DIR := $(LEGION_DIR)
endif
endif

ifndef Legion_DIR
$(error Legion_DIR is not set. Please export Legion_DIR to the directory containing LegionConfig.cmake)
endif

# Query Legion compile/link flags from its CMake package config.
# Try GNU first, then Clang, and finally fall back to a prefix guess.
define legion_find
$(strip $(shell cmake --find-package -DNAME=Legion -DCOMPILER_ID=$(1) -DLANGUAGE=CXX -DMODE=$(2) -DLegion_DIR=$(Legion_DIR) -DCMAKE_PREFIX_PATH=$(Legion_DIR) 2>/dev/null))
endef

LEGION_CFLAGS := $(call legion_find,GNU,COMPILE)
LEGION_LIBS   := $(call legion_find,GNU,LINK)

ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(call legion_find,Clang,COMPILE)
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(call legion_find,Clang,LINK)
endif

LEGION_PREFIX := $(Legion_DIR)
ifneq ($(wildcard $(Legion_DIR)/include/legion.h),)
LEGION_PREFIX := $(Legion_DIR)
else ifneq ($(wildcard $(Legion_DIR)/../../include/legion.h),)
LEGION_PREFIX := $(abspath $(Legion_DIR)/../..)
else ifneq ($(wildcard $(Legion_DIR)/../../../include/legion.h),)
LEGION_PREFIX := $(abspath $(Legion_DIR)/../../..)
endif

ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := -I$(LEGION_PREFIX)/include
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := -L$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib -llegion -lrealm
endif

# Boost paths (optional)
BOOST_INCLUDE :=
BOOST_LIB :=
ifneq ($(BOOST_ROOT),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += $(LEGION_LIBS) $(BOOST_LIB) -lz -pthread

ADDITIONAL_SOURCES :=
PROGRAMS := build/binary_tree build/unit_tests build/concurrency_tests

all: build $(PROGRAMS)

.PHONY: all clean

build:
	mkdir -p build

build/%: %.cpp | build
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf build
