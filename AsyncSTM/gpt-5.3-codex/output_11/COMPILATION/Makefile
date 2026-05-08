CXX ?= g++

LEGION_DIR ?= $(LEGION_ROOT)

UNAME_S := $(shell uname -s)
RT_LIB :=
ifneq ($(UNAME_S),Darwin)
RT_LIB += -lrt
endif

# Query Legion compile/link flags from its CMake package when a prefix is given.
ifneq ($(strip $(LEGION_DIR)),)
LEGION_COMPILE_FLAGS := $(shell cmake --find-package \
  -DNAME=Legion \
  -DCOMPILER_ID=GNU \
  -DLANGUAGE=CXX \
  -DMODE=COMPILE \
  -DCMAKE_PREFIX_PATH=$(LEGION_DIR) 2>/dev/null)

LEGION_LINK_FLAGS := $(shell cmake --find-package \
  -DNAME=Legion \
  -DCOMPILER_ID=GNU \
  -DLANGUAGE=CXX \
  -DMODE=LINK \
  -DCMAKE_PREFIX_PATH=$(LEGION_DIR) 2>/dev/null)
endif

# Fallback to pkg-config if CMake package discovery is unavailable.
ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
LEGION_COMPILE_FLAGS := $(shell pkg-config --cflags legion 2>/dev/null)
endif
ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
LEGION_COMPILE_FLAGS := $(shell pkg-config --cflags Legion 2>/dev/null)
endif

ifeq ($(strip $(LEGION_LINK_FLAGS)),)
LEGION_LINK_FLAGS := $(shell pkg-config --libs legion 2>/dev/null)
endif
ifeq ($(strip $(LEGION_LINK_FLAGS)),)
LEGION_LINK_FLAGS := $(shell pkg-config --libs Legion 2>/dev/null)
endif

# Direct fallback for common Legion install layouts.
LEGION_FALLBACK_INC := $(firstword $(foreach d,$(LEGION_DIR)/include $(LEGION_DIR)/runtime,$(if $(wildcard $(d)/legion.h),$(d))))
LEGION_FALLBACK_LIB := $(firstword $(foreach d,$(LEGION_DIR)/lib $(LEGION_DIR)/lib64 $(LEGION_DIR)/build/lib $(LEGION_DIR)/build/lib64,$(if $(wildcard $(d)/liblegion.*),$(d))))

ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
ifneq ($(strip $(LEGION_FALLBACK_INC)),)
LEGION_COMPILE_FLAGS := -I$(LEGION_FALLBACK_INC)
endif
endif

ifeq ($(strip $(LEGION_LINK_FLAGS)),)
ifneq ($(strip $(LEGION_FALLBACK_LIB)),)
LEGION_LINK_FLAGS := -L$(LEGION_FALLBACK_LIB) -Wl,-rpath,$(LEGION_FALLBACK_LIB) -llegion -lrealm -ldl -pthread $(RT_LIB)
endif
endif

ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB :=
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(BOOST_INCLUDE) $(LEGION_COMPILE_FLAGS)
LDFLAGS  += $(BOOST_LIB) $(LEGION_LINK_FLAGS)

ADDITIONAL_SOURCES =
PROGRAMS = binary_tree unit_tests concurrency_tests
DIRECTORIES = build

all: directories $(PROGRAMS)

.PHONY: all clean directories
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

$(PROGRAMS): %: %.cpp | $(DIRECTORIES)/
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)
	cp -f $@ build/$@

clean:
	rm -rf $(DIRECTORIES) $(PROGRAMS)
