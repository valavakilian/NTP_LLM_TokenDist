#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>


namespace py = pybind11;

struct InsertResult {
    py::list result;
    double execution_time_ms;
};

class Trie_module_protV1 {
public:
    Trie_module_protV1(const std::string& fname, size_t initial_size_gb, int64_t context_length);
    Trie_module_protV1(const std::string& fname);
    ~Trie_module_protV1();

    InsertResult insert(torch::Tensor tensor, bool return_prob_distr);
    py::list retrieve_softlabel(const torch::Tensor& tensor);
    void serialize_to_mmap();
    size_t get_memory_usage() const;
    size_t get_allocated_size() const;
    int64_t get_num_unique_contexts() const;
    int64_t get_num_total_contexts() const;
    void load_metadata(const std::string& metadata_filename);
    void save_metadata();
    py::dict get_memory_stats() const;
    void print_memory_stats() const;
};