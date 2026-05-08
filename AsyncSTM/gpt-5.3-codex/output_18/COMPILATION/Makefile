CXX ?= g++

# Discover Legion compile/link flags via CMake package lookup.
# Uses LEGION_DIR / Legion_DIR / LEGION_ROOT if provided, otherwise falls back
# to default CMake search paths.
LEGION_PREFIX ?= $(strip $(or $(LEGION_DIR),$(Legion_DIR),$(LEGION_ROOT)))
ifneq ($(LEGION_PREFIX),)
LEGION_FIND_ENV := CMAKE_PREFIX_PATH=$(LEGION_PREFIX)
endif

LEGION_CFLAGS := $(strip $(shell $(LEGION_FIND_ENV) cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE 2>/dev/null))
LEGION_LIBS   := $(strip $(shell $(LEGION_FIND_ENV) cmake --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK 2>/dev/null))

# CMake fallback for older package names
ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(strip $(shell $(LEGION_FIND_ENV) cmake --find-package -DNAME=LegionRuntime -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE 2>/dev/null))
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(strip $(shell $(LEGION_FIND_ENV) cmake --find-package -DNAME=LegionRuntime -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK 2>/dev/null))
endif

# pkg-config fallbacks (if available)
ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(strip $(shell pkg-config --cflags legion 2>/dev/null))
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(strip $(shell pkg-config --libs legion 2>/dev/null))
endif
ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(strip $(shell pkg-config --cflags Legion 2>/dev/null))
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(strip $(shell pkg-config --libs Legion 2>/dev/null))
endif

# Wrapper fallback (if provided by installation)
ifeq ($(LEGION_CFLAGS),)
LEGION_CFLAGS := $(strip $(shell legion_cxx --cflags 2>/dev/null))
endif
ifeq ($(LEGION_LIBS),)
LEGION_LIBS := $(strip $(shell legion_cxx --libs 2>/dev/null))
endif

# Prefix-only fallback
ifeq ($(LEGION_CFLAGS),)
ifneq ($(LEGION_PREFIX),)
LEGION_CFLAGS := -I$(LEGION_PREFIX)/include
endif
endif
ifeq ($(LEGION_LIBS),)
ifneq ($(LEGION_PREFIX),)
LEGION_LIBS := -L$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib -llegion
else
LEGION_LIBS := -llegion
endif
endif

# Boost paths (optional)
BOOST_INCLUDE :=
BOOST_LIB :=
ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

CXXFLAGS += -std=c++17 -I. -g -DASTM_LEGION $(LEGION_CFLAGS) $(BOOST_INCLUDE)
LDFLAGS  += $(LEGION_LIBS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
BINARIES := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean directories check-legion

all: check-legion directories $(BINARIES)

check-legion:
	@true

directories: $(BUILD_DIR)/

$(BUILD_DIR)/:
	mkdir -p $@

$(PROGRAMS): %: $(BUILD_DIR)/%
	@true

$(BUILD_DIR)/%: %.cpp | directories
	$(CXX) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)
	ln -sf $@ $*

clean:
	rm -rf $(BUILD_DIR) $(PROGRAMS)
