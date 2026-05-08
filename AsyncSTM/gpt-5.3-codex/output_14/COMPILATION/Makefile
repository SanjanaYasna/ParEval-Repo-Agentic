CXX ?= g++

# Legion package discovery (Legion_DIR must point to the directory containing LegionConfig.cmake)
ifndef Legion_DIR
$(error Legion_DIR is not set. Please export Legion_DIR to the Legion CMake package directory.)
endif

# Detect compiler family for cmake --find-package
CMAKE_COMPILER_ID := $(shell $(CXX) -dM -E -x c++ /dev/null 2>/dev/null | grep -q __clang__ && echo Clang || echo GNU)

LEGION_FIND = cmake --find-package -DNAME=Legion -DLANGUAGE=CXX -DCOMPILER_ID=$(CMAKE_COMPILER_ID) -DLegion_DIR=$(Legion_DIR)
LEGION_CFLAGS := $(strip $(shell $(LEGION_FIND) -DMODE=COMPILE 2>/dev/null))
LEGION_LIBS   := $(strip $(shell $(LEGION_FIND) -DMODE=LINK 2>/dev/null))

# Fallbacks if cmake --find-package does not return flags
LEGION_LIBDIR ?= $(abspath $(Legion_DIR)/../..)
LEGION_PREFIX ?= $(abspath $(LEGION_LIBDIR)/..)

ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS = -I$(LEGION_PREFIX)/include
endif

ifeq ($(LEGION_LIBS),)
LEGION_LIBS = -L$(LEGION_LIBDIR) -Wl,-rpath,$(LEGION_LIBDIR) -llegion -lrealm -lpthread -ldl -lz
endif

# Include Boost paths (optional)
ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE = -I$(BOOST_ROOT)/include
BOOST_LIB     = -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE =
BOOST_LIB     =
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES =
PROGRAMS = binary_tree unit_tests concurrency_tests
BUILD_DIR = build

.PHONY: all clean directories
all: directories $(PROGRAMS)

directories: $(BUILD_DIR)/

$(BUILD_DIR)/:
	mkdir -p $@

%: %.cpp | $(BUILD_DIR)/
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $(BUILD_DIR)/$@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
