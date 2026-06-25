CXX ?= g++

# Query Legion compile/link flags via CMake package discovery.
# If LEGION_DIR is not set, try default CMake search paths.
ifeq ($(strip $(LEGION_DIR)),)
LEGION_CFLAGS_RAW := $(shell if command -v cmake >/dev/null 2>&1; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE 2>/dev/null; fi)
LEGION_LIBS_RAW   := $(shell if command -v cmake >/dev/null 2>&1; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK    2>/dev/null; fi)
else
LEGION_CFLAGS_RAW := $(shell if command -v cmake >/dev/null 2>&1; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE -DCMAKE_PREFIX_PATH="$(LEGION_DIR)" 2>/dev/null; fi)
LEGION_LIBS_RAW   := $(shell if command -v cmake >/dev/null 2>&1; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK    -DCMAKE_PREFIX_PATH="$(LEGION_DIR)" 2>/dev/null; fi)
endif

LEGION_CFLAGS := $(strip $(subst ;, ,$(LEGION_CFLAGS_RAW)))
LEGION_LIBS   := $(strip $(subst ;, ,$(LEGION_LIBS_RAW)))

# Fallback to pkg-config if CMake discovery was unavailable.
ifeq ($(strip $(LEGION_CFLAGS)$(LEGION_LIBS)),)
LEGION_CFLAGS := $(shell if command -v pkg-config >/dev/null 2>&1; then pkg-config --cflags legion 2>/dev/null; fi)
LEGION_LIBS   := $(shell if command -v pkg-config >/dev/null 2>&1; then pkg-config --libs legion 2>/dev/null; fi)
endif

ifeq ($(strip $(LEGION_CFLAGS)$(LEGION_LIBS)),)
$(warning Could not auto-detect Legion compile/link flags; relying on default compiler/linker search paths)
endif

# Include Boost paths only if BOOST_ROOT is provided.
BOOST_INCLUDE :=
BOOST_LIB :=
ifdef BOOST_ROOT
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

CXXFLAGS += -std=c++17 -I. -g -pthread -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += -pthread $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
PROGRAM_TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

all: directories $(PROGRAM_TARGETS)

.PHONY: all clean directories $(PROGRAMS)

directories:
	mkdir -p $(BUILD_DIR)

$(PROGRAMS): %: $(BUILD_DIR)/%
	@true

$(BUILD_DIR)/%: %.cpp | directories
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
