CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
HEADERS = neuron_unified.hpp synapse_unified.hpp network_unified.hpp

all: demo tests

demo: demo_unified.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

tests: tests_unified.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

test: tests
	./tests

clean:
	rm -f demo tests

.PHONY: all test clean
