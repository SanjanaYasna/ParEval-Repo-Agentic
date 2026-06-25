# ============================================================================
# Barnes-Hut Makefile for HPX
# ============================================================================

TARGET    := barnes-hut
SRC       := main.cpp
HDR       := octree.hpp particle.hpp

# Prefer HPX compiler wrapper if available.
CXX       ?= hpxcxx
CXXFLAGS  ?= -O2 -std=c++17 -Wall

# Fallback: standard C++ compiler + pkg-config-discovered HPX flags/libs.
ifeq ($(shell command -v $(CXX) >/dev/null 2>&1; echo $$?),1)
CXX          := g++
HPX_CXXFLAGS := $(shell pkg-config --cflags hpx_application 2>/dev/null || pkg-config --cflags hpx 2>/dev/null)
HPX_LDFLAGS  := $(shell pkg-config --libs hpx_application 2>/dev/null || pkg-config --libs hpx 2>/dev/null)
else
HPX_CXXFLAGS :=
HPX_LDFLAGS  :=
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(HPX_CXXFLAGS) -o $@ $(SRC) $(HPX_LDFLAGS)

clean:
	rm -f $(TARGET)
