CXX ?= g++

PROGRAMS := binary_tree unit_tests concurrency_tests
DIRECTORIES := build

CXXFLAGS += -std=c++17 -I. -O2 -g -pthread
LDFLAGS += -pthread

ifneq ($(strip $(BOOST_ROOT)),)
CXXFLAGS += -I$(BOOST_ROOT)/include
LDFLAGS += -L$(BOOST_ROOT)/lib -Wl,-rpath,$(BOOST_ROOT)/lib
endif

all: directories $(PROGRAMS)

.PHONY: all clean directories
directories: $(DIRECTORIES)/

$(DIRECTORIES)/:
	mkdir -p $@

# Build binaries into build/
build/%: %.cpp | $(DIRECTORIES)/
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Keep user-facing targets as before by mirroring build outputs
$(PROGRAMS): % : build/%
	cp $< $@
	chmod +x $@

clean:
	rm -rf $(DIRECTORIES) $(PROGRAMS)
