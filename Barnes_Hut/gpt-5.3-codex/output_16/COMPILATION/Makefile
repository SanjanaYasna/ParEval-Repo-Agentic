# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

ifeq ($(origin CXX), default)
CXX = g++
endif

CXXFLAGS   ?= -O2 -std=c++17 -Wall -pthread
LDFLAGS    ?= -lm -pthread

TARGET     := barnes-hut
SOURCES    := main.cpp
HEADERS    := octree.hpp particle.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
