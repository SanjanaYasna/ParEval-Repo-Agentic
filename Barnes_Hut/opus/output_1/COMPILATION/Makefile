# ============================================================================
# Barnes-Hut Makefile – standard C++17 threads (no HPX dependency)
# ============================================================================

CXX       = c++
CXXFLAGS  = -O2 -std=c++17 -Wall -pthread
LDFLAGS   = -pthread -lm
TARGET    = barnes-hut

.PHONY: all clean

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
