# ============================================================================
# Barnes-Hut Makefile for HPX
# ============================================================================

# Prefer HPX compiler wrapper (sets include/link flags automatically)
CXX       ?= hpxcxx
CXXFLAGS  ?= -O2 -std=c++17 -Wall

# Optional overrides if not using hpxcxx:
#   make CXX=g++ HPX_CXXFLAGS="..." HPX_LDFLAGS="..."
HPX_CXXFLAGS ?=
HPX_LDFLAGS  ?=

TARGET    := barnes-hut
SRCS      := main.cpp
HDRS      := octree.hpp particle.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(HPX_CXXFLAGS) -o $@ $(SRCS) $(HPX_LDFLAGS)

clean:
	rm -f $(TARGET)
