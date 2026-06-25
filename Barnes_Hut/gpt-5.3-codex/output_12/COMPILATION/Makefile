# ============================================================================
# Barnes-Hut Makefile for HPX
# ============================================================================

# Prefer HPX compiler wrapper; override if needed (e.g., CXX=g++)
CXX       ?= hpxcxx
CXXFLAGS  ?= -O2 -std=c++17 -Wall
TARGET    := barnes-hut
SOURCES   := main.cpp
HEADERS   := octree.hpp particle.hpp

# Optional HPX flags for non-wrapper compilers (safe if empty)
HPX_CFLAGS ?= $(shell pkg-config --cflags hpx_application 2>/dev/null || pkg-config --cflags hpx 2>/dev/null)
HPX_LIBS   ?= $(shell pkg-config --libs hpx_application 2>/dev/null || pkg-config --libs hpx 2>/dev/null)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(HPX_CFLAGS) -o $(TARGET) $(SOURCES) $(HPX_LIBS) -lm

clean:
	rm -f $(TARGET)
