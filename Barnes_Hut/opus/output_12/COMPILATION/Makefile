# ============================================================================
# Barnes-Hut Makefile (standard C++ threads, no HPX)
# ============================================================================

CXX       = c++
CXXFLAGS  = -O2 -std=c++17 -Wall -pthread
LDFLAGS   = -lm -pthread
TARGET    = barnes-hut

.PHONY: all clean

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
