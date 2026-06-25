# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

TARGET    := barnes-hut
SOURCES   := main.cpp
HEADERS   := octree.hpp particle.hpp

CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS   ?=

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
