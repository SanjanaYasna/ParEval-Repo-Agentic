CXX ?= g++

CXXFLAGS += -std=c++17 -O2 -g -I. -pthread
LDFLAGS += -pthread

PROGRAMS := binary_tree unit_tests concurrency_tests
BUILD_DIR := build

.PHONY: all clean directories
all: directories $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

directories: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
