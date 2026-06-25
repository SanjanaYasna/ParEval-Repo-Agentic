# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra -pthread
LDFLAGS   ?=
LDLIBS    ?= -lm
TARGET    := barnes-hut
SOURCES   := main.cpp
HEADERS   := octree.hpp particle.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
