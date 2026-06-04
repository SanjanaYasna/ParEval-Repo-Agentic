# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -fopenmp
TARGET    := barnes-hut
SOURCES   := main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) octree.hpp particle.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) -lm

clean:
	rm -f $(TARGET)
