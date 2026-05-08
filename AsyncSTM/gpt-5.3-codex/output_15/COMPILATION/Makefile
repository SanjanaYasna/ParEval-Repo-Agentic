CXX ?= g++

# Discover Legion compile/link flags from the Legion CMake package.
# Supports LEGION_DIR and/or LEGION_ROOT as hints.
LEGION_CMAKE_HINT :=
ifneq ($(strip $(LEGION_DIR)),)
LEGION_CMAKE_HINT += -DLegion_DIR="$(LEGION_DIR)" -DCMAKE_PREFIX_PATH="$(LEGION_DIR)"
endif
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_CMAKE_HINT += -DCMAKE_PREFIX_PATH="$(LEGION_ROOT)"
endif

LEGION_COMPILE_FLAGS := $(shell cmake $(LEGION_CMAKE_HINT) --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE 2>/dev/null)
LEGION_LINK_FLAGS    := $(shell cmake $(LEGION_CMAKE_HINT) --find-package -DNAME=Legion -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=LINK 2>/dev/null)

# Optional pkg-config fallback.
ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
LEGION_COMPILE_FLAGS := $(shell pkg-config --cflags legion 2>/dev/null)
endif
ifeq ($(strip $(LEGION_LINK_FLAGS)),)
LEGION_LINK_FLAGS := $(shell pkg-config --libs legion 2>/dev/null)
endif

# Manual fallback using LEGION_ROOT/LEGION_DIR prefix when auto-detection fails.
LEGION_PREFIX :=
ifneq ($(strip $(LEGION_ROOT)),)
LEGION_PREFIX := $(LEGION_ROOT)
else ifneq ($(strip $(LEGION_DIR)),)
LEGION_PREFIX := $(abspath $(LEGION_DIR)/../../..)
endif

ifeq ($(strip $(LEGION_COMPILE_FLAGS)),)
ifneq ($(strip $(LEGION_PREFIX)),)
LEGION_COMPILE_FLAGS := -I$(LEGION_PREFIX)/include
endif
endif

ifeq ($(strip $(LEGION_LINK_FLAGS)),)
ifneq ($(strip $(LEGION_PREFIX)),)
LEGION_LINK_FLAGS := -L$(LEGION_PREFIX)/lib -Wl,-rpath,$(LEGION_PREFIX)/lib -llegion -lrealm
endif
endif

# Boost paths (provided externally via BOOST_ROOT)
ifneq ($(strip $(BOOST_ROOT)),)
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB     := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB     :=
endif

CXXFLAGS += -std=c++17 -I. -g $(LEGION_COMPILE_FLAGS) $(BOOST_INCLUDE)
LDFLAGS  += $(LEGION_LINK_FLAGS) $(BOOST_LIB)

ADDITIONAL_SOURCES :=
PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build
BINARIES := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

all: check-legion $(BUILD_DIR) $(BINARIES)

check-legion:
	@if [ -z "$(strip $(LEGION_COMPILE_FLAGS))" ] || [ -z "$(strip $(LEGION_LINK_FLAGS))" ]; then \
		echo "WARNING: Legion flags were not auto-detected. Ensure Legion is discoverable (LEGION_DIR or LEGION_ROOT/CMAKE_PREFIX_PATH)."; \
	fi

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR) check-legion
	$(CXX) -DASTM_LEGION $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean check-legion
