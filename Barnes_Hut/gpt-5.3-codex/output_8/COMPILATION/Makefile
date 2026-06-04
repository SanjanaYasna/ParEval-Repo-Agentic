# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -pthread
TARGET    := barnes-hut

SOURCES   := main.cpp
HEADERS   := octree.hpp particle.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) -lm

clean:
	rm -f $(TARGET)
