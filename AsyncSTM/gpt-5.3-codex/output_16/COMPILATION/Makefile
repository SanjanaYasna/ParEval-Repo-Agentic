CXX ?= g++

# Accept either:
#  - LEGION_DIR=<install-prefix>/lib/cmake/Legion
#  - LEGION_ROOT=<install-prefix>
ifeq ($(strip $(LEGION_DIR)),)
  ifeq ($(strip $(LEGION_ROOT)),)
    LEGION_PREFIX := $(strip $(shell pkg-config --variable=prefix legion 2>/dev/null))
  else
    LEGION_PREFIX := $(abspath $(LEGION_ROOT))
  endif
else
  LEGION_PREFIX := $(abspath $(LEGION_DIR)/../../..)
endif

ifeq ($(strip $(LEGION_PREFIX)),)
$(error Could not determine Legion installation. Please set LEGION_DIR=<legion-install>/lib/cmake/Legion or LEGION_ROOT=<legion-install>)
endif

# Prefer lib, fall back to lib64 when needed.
ifneq ("$(wildcard $(LEGION_PREFIX)/lib)","")
LEGION_LIB_PATH := $(LEGION_PREFIX)/lib
else ifneq ("$(wildcard $(LEGION_PREFIX)/lib64)","")
LEGION_LIB_PATH := $(LEGION_PREFIX)/lib64
else
LEGION_LIB_PATH := $(LEGION_PREFIX)/lib
endif

# Legion include/link flags (default mapper; no custom mapper sources)
LEGION_INCLUDE := -I$(LEGION_PREFIX)/include
LEGION_LIBDIR  := -L$(LEGION_LIB_PATH) -Wl,-rpath,$(LEGION_LIB_PATH)
LEGION_LIBS    := -llegion -lrealm -lrt -lpthread -ldl -lz

# Boost paths (optional)
ifdef BOOST_ROOT
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB     :=
endif

CXXFLAGS += -std=c++17 -O2 -g -I. -DASTM_LEGION $(LEGION_INCLUDE) $(BOOST_INCLUDE)
LDFLAGS  += $(LEGION_LIBDIR) $(BOOST_LIB) $(LEGION_LIBS)

PROGRAMS  := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
TARGETS   := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean directories
all: directories $(TARGETS)

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp | directories
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
