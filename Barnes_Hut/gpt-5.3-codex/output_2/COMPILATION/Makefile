# ============================================================================
# Barnes-Hut Makefile
# ============================================================================

CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS   ?=
TARGET    := barnes-hut

SOURCES   := main.cpp
HEADERS   := octree.hpp particle.hpp

# Enable OpenMP with: make OPENMP=1
OPENMP ?= 0
ifeq ($(OPENMP),1)
CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
