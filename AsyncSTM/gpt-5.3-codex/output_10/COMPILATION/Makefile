CXX ?= g++

BUILD_DIR := build
PROGRAMS := binary_tree unit_tests concurrency_tests
TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))
ADDITIONAL_SOURCES :=

ifdef BOOST_ROOT
BOOST_INCLUDE := -I$(BOOST_ROOT)/include
BOOST_LIB := -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
else
BOOST_INCLUDE :=
BOOST_LIB :=
endif

LEGION_CXXFLAGS :=
LEGION_LDFLAGS :=
ifdef Legion_DIR
LEGION_CXXFLAGS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=GNU \
	-DLANGUAGE=CXX \
	-DMODE=COMPILE \
	-DLegion_DIR=$(Legion_DIR) 2>/dev/null)

LEGION_LDFLAGS := $(shell cmake --find-package \
	-DNAME=Legion \
	-DCOMPILER_ID=GNU \
	-DLANGUAGE=CXX \
	-DMODE=LINK \
	-DLegion_DIR=$(Legion_DIR) 2>/dev/null)
endif

CPPFLAGS += -I. $(BOOST_INCLUDE) -DASTM_LEGION
CXXFLAGS += -std=c++17 -O2 -g -pthread $(LEGION_CXXFLAGS)
LDFLAGS += -pthread $(LEGION_LDFLAGS) $(BOOST_LIB)

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ADDITIONAL_SOURCES) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
