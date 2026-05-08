CXX ?= g++

# Allow LEGION_DIR, LEGION_ROOT, or common system install paths.
ifeq ($(strip $(LEGION_DIR)),)
  ifneq ($(strip $(LEGION_ROOT)),)
    LEGION_PREFIX := $(abspath $(LEGION_ROOT))
    LEGION_DIR := $(LEGION_PREFIX)/lib/cmake/Legion
  else ifneq ($(wildcard /usr/local/lib/cmake/Legion),)
    LEGION_DIR := /usr/local/lib/cmake/Legion
    LEGION_PREFIX := /usr/local
  else ifneq ($(wildcard /usr/lib/cmake/Legion),)
    LEGION_DIR := /usr/lib/cmake/Legion
    LEGION_PREFIX := /usr
  else
    $(warning LEGION_DIR/LEGION_ROOT not set; relying on compiler/linker default search paths for Legion)
    LEGION_PREFIX :=
  endif
else
  LEGION_PREFIX := $(abspath $(LEGION_DIR)/../../..)
endif

# Legion include/link flags (default mapper; no custom mapper sources)
ifneq ($(strip $(LEGION_PREFIX)),)
LEGION_INCLUDE := -I$(LEGION_PREFIX)/include
LEGION_LIBDIR  := -L$(LEGION_PREFIX)/lib -L$(LEGION_PREFIX)/lib64 \
                  -Wl,-rpath,$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib64
else
LEGION_INCLUDE :=
LEGION_LIBDIR  :=
endif
LEGION_LIBS    := -llegion -lrealm -lrt -lpthread -ldl -lz

# Boost paths (optional)
ifneq ($(strip $(BOOST_ROOT)),)
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
