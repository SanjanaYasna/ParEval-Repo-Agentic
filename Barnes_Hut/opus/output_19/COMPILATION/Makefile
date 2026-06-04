# ============================================================================
# Barnes-Hut Makefile (OpenMP version)
# ============================================================================

CXX       = c++
CXXFLAGS  = -O2 -std=c++17 -Wall -fopenmp
LDFLAGS   = -lm -fopenmp
TARGET    = barnes-hut

.PHONY: all clean

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
