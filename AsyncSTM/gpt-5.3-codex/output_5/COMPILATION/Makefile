CXX ?= g++

# Try to derive Legion_DIR from LEGION_ROOT if not provided explicitly.
LEGION_DIR ?= $(if $(LEGION_ROOT),$(LEGION_ROOT)/lib/cmake/Legion,)

# Resolve Legion compile/link flags via CMake package discovery (preferred).
LEGION_CFLAGS := $(strip $(shell if [ -n "$(Legion_DIR)" ]; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE -DLegion_DIR=$(Legion_DIR) 2>/dev/null | tr ';' ' '; fi))
LEGION_LIBS   := $(strip $(shell if [ -n "$(Legion_DIR)" ]; then cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK    -DLegion_DIR=$(Legion_DIR) 2>/dev/null | tr ';' ' '; fi))

# Fallback to pkg-config if CMake package discovery is unavailable.
ifeq ($(strip $(LEGION_CFLAGS)),)
LEGION_CFLAGS := $(strip $(shell pkg-config --cflags legion 2>/dev/null))
endif

ifeq ($(strip $(LEGION_LIBS)),)
LEGION_LIBS := $(strip $(shell pkg-config --libs legion 2>/dev/null))
endif

# Final fallback using LEGION_ROOT/system defaults.
ifeq ($(strip $(LEGION_CFLAGS)),)
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_CFLAGS := -I$(LEGION_ROOT)/include -I$(LEGION_ROOT)/include/legion
endif
endif

ifeq ($(strip $(LEGION_LIBS)),)
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_LIBS := -L$(LEGION_ROOT)/lib -Wl,-rpath,$(LEGION_ROOT)/lib -llegion
else
LEGION_LIBS := -llegion
endif
endif

# Boost paths (optional)
ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION -pthread $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += -pthread $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES =
PROGRAMS = build/binary_tree build/unit_tests build/concurrency_tests
DIRECTORIES = build

all: directories $(PROGRAMS)

.PHONY: all clean directories

directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

build/%: %.cpp | $(DIRECTORIES)/
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(DIRECTORIES)
