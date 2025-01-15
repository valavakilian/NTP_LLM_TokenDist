#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <filesystem>
// #include "Trie_module.h" 

#include <cstdint>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <cmath>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <unistd.h> // for sleep()


namespace py = pybind11;
using namespace py::literals;


// #include <nlohmann/json.hpp>  // Include a JSON library like nlohmann/json for C++

namespace py = pybind11;

#define DEBUG_PRINT(x) std::cout << x << std::endl


struct RAMTrieNode {
    int32_t count;
    int64_t num_children;
    std::pair<int64_t, int64_t>* children;  // Raw array instead of vector
    uint32_t children_capacity;
    uint8_t node_level;
};
std::vector<RAMTrieNode> nodes;  // Our main container during construction

// Memory mapped structure for final format
struct MMAPTrieNode {
    int32_t count;
    uint16_t num_children;
    int64_t children_offset;
    uint8_t node_level;
    uint16_t value;  // Add this to store the value associated with this node
};

// For RAM-based nodes
void printTrieNode(const RAMTrieNode* node) {
    std::cout << "RAMTrieNode Information:" << std::endl;
    std::cout << "Count: " << node->count << std::endl;
    std::cout << "Number of Children: " << node->num_children << std::endl;
    std::cout << "Children Capacity: " << node->children_capacity << std::endl;
    std::cout << "Node Level: " << node->node_level << std::endl;
    if (node->children != nullptr) {
        std::cout << "First few children: ";
        for (int i = 0; i < std::min(node->num_children, (int64_t)5); i++) {
            std::cout << "(" << node->children[i].first << "," << node->children[i].second << ") ";
        }
        std::cout << std::endl;
    }
}

// For memory-mapped nodes
void printTrieNode(const MMAPTrieNode* node) {
    std::cout << "MMAPTrieNode Information:" << std::endl;
    std::cout << "Count: " << node->count << std::endl;
    std::cout << "Number of Children: " << node->num_children << std::endl;
    std::cout << "Children Offset: " << node->children_offset << std::endl;
    std::cout << "Node Level: " << node->node_level << std::endl;
}



struct MemoryStats {
    size_t total_bytes;           
    size_t node_struct_bytes;     
    size_t children_array_bytes;  
    size_t wasted_bytes;         
    size_t num_nodes;            
    size_t num_empty_children;   
    double avg_fill_ratio;       
};

struct InsertResult {
    std::vector<std::vector<std::unordered_map<int64_t, double>>> distributions;
    double execution_time_ms;
};


class Trie_module_protV1 {

private:

    // std::vector<RAMTrieNode> nodes;  // Our main container during construction

    // File handling - needed for eventual memory mapping
    int fd;
    char* mapped_memory;
    std::atomic<size_t> file_size;
    size_t allocated_size;
    std::string filename;
    std::string metadata_filename;

    // Core trie statistics
    std::atomic<int64_t> num_unique_contexts;
    std::atomic<int64_t> num_total_contexts;
    int64_t context_length;
    
    // Construction mode flag
    bool is_construction_mode;

    // During construction phase, this returns a pointer to the RAM node
    RAMTrieNode* get_node(size_t index) {
        if (index >= nodes.size()) {
            throw std::runtime_error("Invalid node index");
        }
        return &nodes[index];
    }

    // During construction phase, this returns a pointer to the children vector
    // std::vector<std::pair<int64_t, int64_t>>& get_children(RAMTrieNode* node) {
    //     return node->children;
    // }
    std::pair<int64_t, int64_t>* get_children(RAMTrieNode* node) {
        return node->children;
    }

    size_t allocate_node(int64_t parent_level) {
        size_t new_index = nodes.size();
        nodes.push_back(createNode(parent_level));
        return new_index;
    }

    // Fix node creation
    RAMTrieNode createNode(int64_t parent_level) {
        return RAMTrieNode{
            0,          // count
            0,          // num_children
            nullptr,    // children pointer (initialize as null)
            0,          // children_capacity
            parent_level + 1  // node_level
        };
    }

    int64_t find_child(RAMTrieNode* node, int64_t value) {
        if (node->children == nullptr || node->num_children == 0) {
            return -1;
        }

        // Binary search
        int64_t left = 0, right = node->num_children - 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            if (node->children[mid].first == value) {
                return node->children[mid].second;  // Return index of child node
            }
            if (node->children[mid].first < value) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    // For memory-mapped mode - keep original array version
    int64_t find_child(std::pair<uint16_t, int64_t>* children, uint16_t num_children, uint16_t value) {
        // std::cout << "find_child searching for value " << value 
        //         << " among " << num_children << " children" << std::endl;
        
        for (int64_t i = 0; i < num_children; i++) {
            // std::cout << "Comparing with child " << i << ": " << children[i].first << std::endl;
            if (children[i].first == value) {
                return i;
            }
        }
        return -1;
    }

    // Add this function to handle children array allocation/reallocation
    size_t allocate_children(RAMTrieNode* node, size_t needed_size) {
        if (node->children == nullptr) {

            size_t initial_size;
            // First allocation
            if (node->node_level > 3){
                initial_size = 1;
            } else {
                initial_size = std::max(needed_size, size_t(10));  // Start with at least 8
            }
            
            node->children = new std::pair<int64_t, int64_t>[initial_size];
            node->children_capacity = initial_size;
        }
        else if (node->num_children + needed_size > node->children_capacity) {
            // Need to grow
            size_t new_capacity;
            if (node->node_level > 3){
                new_capacity = node->children_capacity + 3;
            } else {
                new_capacity = node->children_capacity + 10;
            }

            // size_t new_capacity = node->children_capacity + 5;
            auto new_children = new std::pair<int64_t, int64_t>[new_capacity];
            // Use copy instead of memmove for proper object copying
            for (int64_t i = 0; i < node->num_children; i++) {
                new_children[i] = node->children[i];
            }
            delete[] node->children;
            node->children = new_children;
            node->children_capacity = new_capacity;
        }
        return node->children_capacity;
    }

    // And modify insert_child to use proper copying
    void insert_child(RAMTrieNode* node, int64_t value, int64_t child_index) {
        // Ensure we have space
        if (node->children == nullptr || node->num_children >= node->children_capacity) {
            allocate_children(node, 1);
        }

        // Find insertion position
        int64_t insert_pos = 0;
        while (insert_pos < node->num_children && node->children[insert_pos].first < value) {
            insert_pos++;
        }

        // Shift existing elements using proper copying
        for (int64_t i = node->num_children; i > insert_pos; i--) {
            node->children[i] = node->children[i-1];
        }

        // Insert new child
        node->children[insert_pos] = std::make_pair(value, child_index);
        node->num_children++;
    }


    // Memory-mapped version of get_node
    MMAPTrieNode* get_mmap_node(size_t offset) {
        if (offset >= file_size) {
            throw std::runtime_error("Attempted to access memory outside of mapped region");
        }
        return reinterpret_cast<MMAPTrieNode*>(mapped_memory + offset);
    }

    // Memory-mapped version of get_children
    std::pair<uint16_t, int64_t>* get_mmap_children(MMAPTrieNode* node) {
        if (node->children_offset >= file_size) {
            std::cout << "ERROR in get_mmap_children: offset " << node->children_offset 
                    << " exceeds file_size " << file_size << std::endl;
            throw std::runtime_error("Invalid children offset");
        }
        return reinterpret_cast<std::pair<uint16_t, int64_t>*>(mapped_memory + node->children_offset);
    }


public:
    // Constructor for building new trie in RAM
    Trie_module_protV1(const std::string& fname, size_t initial_size_gb, int64_t context_length) 
        : filename(fname + ".bin"), 
        context_length(context_length),
        is_construction_mode(true) {
        
        metadata_filename = filename + "_metadata.bin";

        // Reserve space based on expected number of nodes
        size_t expected_nodes = initial_size_gb * 1024 * 1024 * 1024 / sizeof(RAMTrieNode);
        DEBUG_PRINT("WE ARE HERE, expected_nodes: " << expected_nodes);
        
        nodes.reserve(expected_nodes);  // Pre-allocate space
        
        // Start with root node
        nodes.push_back(RAMTrieNode{
            0,      // count
            {},     // empty children vector
            0       // level
        });

        num_unique_contexts = 0;
        num_total_contexts = 0;

        int num_procs = omp_get_num_procs();
        DEBUG_PRINT("Number of processors available: " << num_procs);
        DEBUG_PRINT("Trie initialized in RAM construction mode");
    }

    // Constructor for loading existing memory-mapped trie
    Trie_module_protV1(const std::string& fname) 
        : filename(fname + ".bin"),
        is_construction_mode(false) {

        std::cout << "loading from fname: " << filename << "\n";
        
        // Step 1: Open the file
        fd = open(filename.c_str(), O_RDWR);
        if (fd == -1) {
            throw std::runtime_error("Error opening trie file: " + std::string(strerror(errno)));
        }

        // Step 2: Get the size of the file
        struct stat fileInfo;
        if (fstat(fd, &fileInfo) == -1) {
            close(fd);
            throw std::runtime_error("Error getting file size: " + std::string(strerror(errno)));
        }
        allocated_size = fileInfo.st_size;

        std::cout << "Loading memory-mapped file size: " << allocated_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

        // Step 3: Memory-map the file
        mapped_memory = (char*) mmap(nullptr, allocated_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Error memory mapping the file: " + std::string(strerror(errno)));
        }

        metadata_filename = filename + "_metadata.bin";
        
        // Step 4: Load metadata
        try {
            load_metadata(metadata_filename);
            std::cout << "Successfully loaded Trie with following statistics:" << std::endl;
            std::cout << "Number of unique contexts: " << num_unique_contexts.load() << std::endl;
            std::cout << "Number of total contexts: " << num_total_contexts.load() << std::endl;
            std::cout << "Context length: " << context_length << std::endl;
        } catch (const std::exception& e) {
            if (mapped_memory != MAP_FAILED) {
                munmap(mapped_memory, allocated_size);
            }
            close(fd);
            throw std::runtime_error("Error loading metadata: " + std::string(e.what()));
        }
    }

    void serialize_to_mmap() {
        std::cout << "----serialize_to_mmap----" << std::endl;
        
        if (!is_construction_mode) {
            throw std::runtime_error("Already in mmap mode");
        }

        // First calculate where each node will be stored
        size_t total_nodes = nodes.size();
        std::vector<size_t> node_addresses(total_nodes);  // Where each node will be stored
        

        // SOME DEBUG PRINTING
        std::cout << "=== Starting serialization ===" << std::endl;
        std::cout << "Total nodes to write: " << nodes.size() << std::endl;

        // // When writing nodes & children
        // for (size_t i = 0; i < total_nodes; i++) {
        //     auto& ram_node = nodes[i];
        //     std::cout << "\nWriting node " << i << ":" << std::endl;
        //     std::cout << "- children count: " << ram_node.num_children << std::endl;
        //     if (ram_node.num_children > 0) {
        //         std::cout << "- children values: ";
        //         for (int j = 0; j < ram_node.num_children; j++) {
        //             std::cout << "(" << ram_node.children[j].first 
        //                     << "," << ram_node.children[j].second << ") ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        // First block: all nodes
        size_t current_offset = 0;
        for (size_t i = 0; i < total_nodes; i++) {
            node_addresses[i] = current_offset;
            current_offset += sizeof(MMAPTrieNode);
        }

        // Second block: all children arrays
        size_t children_start = current_offset;  // Start of children arrays section
        std::vector<size_t> children_addresses(total_nodes);
        
        for (size_t i = 0; i < total_nodes; i++) {
            children_addresses[i] = current_offset;
            if (nodes[i].num_children > 0) {
                current_offset += nodes[i].num_children * sizeof(std::pair<uint16_t, int64_t>);
            }
        }

        // std::cout << "Debug offsets:\n";
        // for (size_t i = 0; i < std::min(total_nodes, size_t(5)); i++) {
        //     std::cout << "Node " << i << ": address=" << node_addresses[i] 
        //             << ", children_address=" << children_addresses[i] << std::endl;
        // }

        size_t total_size = current_offset;
        std::cout << "total_size: " << total_size << std::endl;

        // Create and map file
        fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            throw std::runtime_error("Failed to create file for memory mapping");
        }

        // Set the file size
        if (ftruncate(fd, total_size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to set file size");
        }

        // Map the file
        mapped_memory = static_cast<char*>(mmap(NULL, total_size, PROT_READ | PROT_WRITE, 
                                            MAP_SHARED, fd, 0));
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file to memory");
        }

        // Write all nodes first
        for (size_t i = 0; i < total_nodes; i++) {
            MMAPTrieNode mmap_node{
                nodes[i].count,
                static_cast<uint16_t>(nodes[i].num_children),
                children_addresses[i],  // Point to where children array will be
                static_cast<uint8_t>(nodes[i].node_level)
            };
            memcpy(mapped_memory + node_addresses[i], &mmap_node, sizeof(MMAPTrieNode));
        }

        for (size_t i = 0; i < total_nodes; i++) {
            if (nodes[i].num_children > 0) {
                std::vector<std::pair<uint16_t, int64_t>> child_pairs(nodes[i].num_children);
                for (uint16_t j = 0; j < nodes[i].num_children; j++) {

                    // Validate vocab ID before compression
                    // if (nodes[i].children[j].first > UINT16_MAX) {
                    //     throw std::runtime_error("Vocab ID exceeds uint16_t range during serialization");
                    // }
                    // Store both value and address
                    child_pairs[j] = {
                        static_cast<uint16_t>(nodes[i].children[j].first),  // Original value
                        node_addresses[nodes[i].children[j].second]  // Address where child is stored
                    };
                }
                // Write array of child pairs
                memcpy(mapped_memory + children_addresses[i], 
                    child_pairs.data(), 
                    nodes[i].num_children * sizeof(std::pair<uint16_t, int64_t>));
            }
        }

        // Update sizes and clean up
        file_size = total_size;
        allocated_size = total_size;
        save_metadata();

        // Clean up and switch mode
        for (auto& node : nodes) {
            delete[] node.children;
        }
        nodes.clear();
        nodes.shrink_to_fit();
        is_construction_mode = false;

        std::cout << "----serialization_complete----" << std::endl;
    }



    void save_metadata() {
        // Save main metadata
        std::ofstream file(metadata_filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open metadata file for writing");
        }

        file << file_size << "\n";
        file << allocated_size << "\n";
        file << num_unique_contexts.load() << "\n";
        file << num_total_contexts.load() << "\n";
        file << context_length << "\n";

        file.close();

        // // Save additional array data in binary format
        // std::ofstream per_level_data(filename.substr(0, filename.length() - 4) + "_level_data.bin", 
        //                             std::ios::binary);
        // if (!per_level_data.is_open()) {
        //     throw std::runtime_error("Failed to open level data file for writing");
        // }

        // // Write all level-specific data
        // per_level_data.close();

        std::cout << "----Saver----" << std::endl;
        std::cout << "Metadata saved to " << metadata_filename << std::endl;
        std::cout << "Memory-mapped file size: " << file_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "Memory-mapped allocated size: " << allocated_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    }

    void load_metadata(const std::string& metadata_filename) {
        // Open the metadata file
        std::ifstream file(metadata_filename);
        if (file.is_open()) {
            size_t temp_size; 
            file >> temp_size;
            file_size.store(temp_size);

            file >> allocated_size;
            int64_t num_unique_contexts_val, num_total_contexts_val;
            file >> num_unique_contexts_val;
            file >> num_total_contexts_val;
            num_unique_contexts.store(num_unique_contexts_val);
            num_total_contexts.store(num_total_contexts_val);
            file >> context_length;

            // Assuming a known format
            std::string temp;
            int64_t level;
            int count;
            file.close();
        }
        std::cout << "----Loader----" << std::endl;
        std::cout << "Metadata loaded from " << metadata_filename << std::endl;
        std::cout << "Memory-mapped file size: " << file_size << " Giga bytes" << std::endl;
        std::cout << "Memory-mapped allocated size: " << allocated_size << " Giga bytes" << std::endl;
    }

    ~Trie_module_protV1() {
        if (is_construction_mode) {
            // Clean up RAM structures
            for (auto& node : nodes) {
                delete[] node.children;
            }
            nodes.clear();
            nodes.shrink_to_fit();
        } else {
            // Clean up memory-mapped resources as before
            if (mapped_memory != MAP_FAILED) {
                munmap(mapped_memory, allocated_size);
            }
            if (fd != -1) {
                ftruncate(fd, file_size);
                close(fd);
            }
        }
    }
    
    
    std::vector<std::unordered_map<int64_t, double>> insert_context(
            const std::vector<std::vector<int64_t>>& sequences,
            size_t sequence_idx,
            bool return_prob_distr) {   
        std::vector<std::unordered_map<int64_t, double>> distributions;
        int64_t current_level = 0;
        size_t current_index = 0;  // Start at root

        const auto& sequence = sequences[sequence_idx];
        for (size_t j = 0; j < sequence.size(); j++) {
            int64_t value = sequence[j];
            
            RAMTrieNode* current = &nodes[current_index];
            
            if (j > 0) {
                num_total_contexts.fetch_add(1);
            }
            
            // Find child if it exists
            int64_t child_index = -1;
            if (current->children != nullptr && current->num_children > 0) {
                child_index = find_child(current, value);
            }

            if (child_index == -1) {
                // Create new node
                size_t new_node_index = nodes.size();
                nodes.push_back(createNode(current->node_level));
                
                // Insert new child reference
                insert_child(current, value, new_node_index);
                current_index = new_node_index;
            } else {
                current_index = child_index;
            }

            // Update counts
            nodes[current_index].count++;
            current_level++;

            if (return_prob_distr) {
                // Create distribution
                std::unordered_map<int64_t, double> current_distribution;
                if (current->children != nullptr) {
                    for (int64_t i = 0; i < current->num_children; i++) {
                        RAMTrieNode* child = &nodes[current->children[i].second];
                        current_distribution[current->children[i].first] = static_cast<double>(child->count);
                    }
                }
                distributions.push_back(current_distribution);
            }
        }

        return distributions;
    }

    double insert(const std::vector<std::vector<int64_t>>& sequences) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Get system capabilities and set thread count
        int num_procs = omp_get_num_procs();
        int num_threads = 1;  // Can be adjusted based on needs
        omp_set_num_threads(num_threads);

        // Calculate batch size per thread
        int batch_size = sequences.size();
        int chunk_size = std::max(1, batch_size / num_threads);
        
        // Dynamic scheduling with calculated chunk size
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            insert_context(sequences, seq_idx, false);  // false since we don't need distributions
        }

        // End timing and calculate duration
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        return static_cast<double>(duration.count());
    }


    std::vector<std::unordered_map<int64_t, double>> retrieve_context_softlabel(const torch::Tensor& tensor, int64_t column) {
        if (is_construction_mode) {
            throw std::runtime_error("Cannot retrieve in construction mode - need to serialize to mmap first");
        }
        

        std::vector<std::unordered_map<int64_t, double>> distributions;
        size_t current_offset = 0;  // Start at root
        
        // std::cout << "=== Starting retrieval ===" << std::endl;
        // std::cout << "File size: " << file_size << std::endl;
        
        MMAPTrieNode* current = get_mmap_node(current_offset);
        // std::cout << "Root node at offset " << current_offset << ":" << std::endl;
        // std::cout << "- count: " << current->count << std::endl;
        // std::cout << "- num_children: " << current->num_children << std::endl;
        // std::cout << "- children_offset: " << current->children_offset << std::endl;

        auto accessor = tensor.accessor<int64_t, 2>();

        for (int64_t j = 0; j < accessor.size(1); j++) {
            // int64_t value = accessor[column][j];
            // std::cout << "\nLooking for value: " << value << std::endl;

            int64_t tensor_value = accessor[column][j];
            // Validate value range
            // if (tensor_value > UINT16_MAX) {
            //     throw std::runtime_error("Tensor value exceeds uint16_t range");
            // }
            uint16_t value = static_cast<uint16_t>(tensor_value);

            if (current->num_children > 0) {
                std::pair<uint16_t, int64_t>* children = get_mmap_children(current);
                
                // std::cout << "Current node has " << current->num_children << " children:" << std::endl;
                // for (int64_t i = 0; i < current->num_children; i++) {
                //     std::cout << "Child " << i << ": value=" << children[i].first 
                //             << ", offset=" << children[i].second << std::endl;
                // }
                
                int64_t child_index = find_child(children, current->num_children, value);
                // std::cout << "find_child returned index: " << child_index << std::endl;

                if (child_index != -1) {
                    // std::cout << "Moving to child node at offset: " << children[child_index].second << std::endl;
                    current = get_mmap_node(children[child_index].second);
                    
                    std::unordered_map<int64_t, double> distribution;
                    if (current->num_children > 0) {
                        std::pair<uint16_t, int64_t>* next_children = get_mmap_children(current);
                        // std::cout << "Building distribution from " << current->num_children << " children:" << std::endl;
                        for (uint16_t k = 0; k < current->num_children; k++) {
                            MMAPTrieNode* child = get_mmap_node(next_children[k].second);
                            distribution[static_cast<int64_t>(next_children[k].first)] = static_cast<double>(child->count);
                            // std::cout << "Added to distribution: " << next_children[k].first 
                            //         << " -> " << child->count << std::endl;
                        }
                    }
                    distributions.push_back(distribution);
                    // std::cout << "Added distribution of size: " << distribution.size() << std::endl;
                } else {
                    // std::cout << "Value not found, returning early" << std::endl;
                    return distributions;
                }
            } else {
                // std::cout << "Node has no children, returning early" << std::endl;
                return distributions;
            }
        }
        
        // std::cout << "=== Retrieval complete ===" << std::endl;
        // std::cout << "Returning " << distributions.size() << " distributions" << std::endl;
        return distributions;
    }

    py::list retrieve_softlabel(const torch::Tensor& tensor) {

        // std::cout << "\n=== retrieve_softlabel ===\n";
        // Ensure that the input tensor is 2D and of type int64 (torch::kInt64)
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");

        std::vector<std::vector<std::unordered_map<int64_t, double>>> soft_label_distributions(tensor.size(0));
        
        // Process all columns sequentially in a single thread
        for (int col = 0; col < tensor.size(0); col++) {
            // std::cout << "\n=== In for loop for one context ===\n";
            soft_label_distributions[col] = this->retrieve_context_softlabel(tensor, col);
        }

        // Convert the results to Python list
        py::list soft_label_list;
        for (const auto& seq_distributions : soft_label_distributions) {
            // std::cout << "\n=== In for loop for turning to python ===\n";
            py::list seq_result;
            for (const auto& dist : seq_distributions) {
                py::dict py_dist;
                for (const auto& [token, prob] : dist) {
                    py_dist[py::int_(token)] = py::float_(prob);
                }
                seq_result.append(py_dist);
            }
            soft_label_list.append(seq_result);
        }

        return soft_label_list;
    }


    size_t get_memory_usage() const {
        if (is_construction_mode) {
            // For RAM mode, calculate total bytes used
            size_t total_bytes = 0;
            for (const auto& node : nodes) {
                // Size of node itself
                total_bytes += sizeof(RAMTrieNode);
                // Size of its children array if allocated
                if (node.children != nullptr) {
                    total_bytes += node.children_capacity * sizeof(std::pair<int64_t, int64_t>);
                }
            }
            return total_bytes;
        } else {
            // For mmap mode, return file size as before
            return file_size;
        }
    }

    size_t get_allocated_size() const {
        if (is_construction_mode) {
            // For RAM mode, return total reserved capacity
            size_t total_size = nodes.capacity() * sizeof(RAMTrieNode);
            for (const auto& node : nodes) {
                if (node.children != nullptr) {
                    total_size += node.children_capacity * sizeof(std::pair<int64_t, int64_t>);
                }
            }
            return total_size;
        } else {
            // For mmap mode, return allocated size as before
            return allocated_size;
        }
    }

    int64_t get_num_unique_contexts() const {
        // Works same way in both modes
        return num_unique_contexts.load();
    }

    int64_t get_num_total_contexts() const {
        // Works same way in both modes
        return num_total_contexts.load();
    }


    MemoryStats calculate_detailed_memory_usage() const {
        MemoryStats stats = {0, 0, 0, 0, 0, 0, 0.0};
        
        if (!is_construction_mode) {
            throw std::runtime_error("Memory stats only available in construction mode");
        }
        
        // Base node structure memory
        stats.num_nodes = nodes.size();
        stats.node_struct_bytes = nodes.size() * sizeof(RAMTrieNode);
        
        double total_fill_ratio = 0.0;
        int nodes_with_children = 0;
        
        // Analyze each node's children array
        for (const auto& node : nodes) {
            if (node.children_capacity > 0) {
                // Memory allocated for children array
                size_t array_bytes = node.children_capacity * sizeof(std::pair<int64_t, int64_t>);
                stats.children_array_bytes += array_bytes;
                
                // Calculate wasted space
                size_t used_bytes = node.num_children * sizeof(std::pair<int64_t, int64_t>);
                stats.wasted_bytes += (array_bytes - used_bytes);
                
                if (node.num_children == 0) {
                    stats.num_empty_children++;
                }
                
                // Calculate fill ratio
                if (node.children_capacity > 0) {
                    total_fill_ratio += static_cast<double>(node.num_children) / node.children_capacity;
                    nodes_with_children++;
                }
            }
        }
        
        // Calculate total memory
        stats.total_bytes = stats.node_struct_bytes + stats.children_array_bytes;
        
        // Calculate average fill ratio
        if (nodes_with_children > 0) {
            stats.avg_fill_ratio = total_fill_ratio / nodes_with_children;
        }
        
        return stats;
    }

    py::dict get_memory_stats() const {
        MemoryStats stats = calculate_detailed_memory_usage();
        
        py::dict result;
        result["total_mb"] = stats.total_bytes / (1024.0 * 1024.0);
        result["node_structs_mb"] = stats.node_struct_bytes / (1024.0 * 1024.0);
        result["children_arrays_mb"] = stats.children_array_bytes / (1024.0 * 1024.0);
        result["wasted_mb"] = stats.wasted_bytes / (1024.0 * 1024.0);
        result["num_nodes"] = stats.num_nodes;
        result["empty_children"] = stats.num_empty_children;
        result["avg_fill_ratio"] = stats.avg_fill_ratio;
        
        return result;
    }

    void print_memory_stats() const {
        MemoryStats stats = calculate_detailed_memory_usage();
        
        std::cout << "\n=== Memory Usage Statistics ===\n";
        std::cout << "Size Of Nodes: " << nodes.size() << " \n";
        std::cout << "Total Memory Used: " << stats.total_bytes / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Node Structures: " << stats.node_struct_bytes / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Children Arrays: " << stats.children_array_bytes / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Wasted Space: " << stats.wasted_bytes / (1024.0 * 1024.0) << " MB\n";
        std::cout << "Number of Nodes: " << stats.num_nodes << "\n";
        std::cout << "Nodes with Empty Children Arrays: " << stats.num_empty_children << "\n";
        std::cout << "Average Children Array Fill Ratio: " << (stats.avg_fill_ratio * 100.0) << "%\n";
        std::cout << "==============================\n";
    }

};


// Function to convert std::vector<int64_t> keys to Python tuple
py::dict convert_to_python_dict(const std::unordered_map<std::vector<int64_t>, int64_t>& sequences) {
    py::dict result;
    for (const auto& pair : sequences) {
        py::tuple key = py::cast(pair.first);  // Convert std::vector<int64_t> to tuple
        result[key] = pair.second;  // Assign the count to the corresponding tuple
    }
    return result;
}


class ProgressBar {
    size_t total;
    size_t current;
    size_t last_percent;
    std::chrono::steady_clock::time_point start_time;
    std::string description;
    std::mutex mtx;  // For thread safety

public:
    ProgressBar(size_t total, std::string desc = "Progress") 
        : total(total), current(0), last_percent(0), description(desc) {
        start_time = std::chrono::steady_clock::now();
        std::cout << description << ": 0%" << std::flush;
    }

    void update(size_t n = 1) {
        std::lock_guard<std::mutex> lock(mtx);
        current += n;
        size_t percent = (current * 100) / total;
        if (percent > last_percent) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            float items_per_sec = current / (duration.count() + 0.1f);

            std::cout << "\r" << description << ": " << percent << "% "
                     << "[" << current << "/" << total << "] "
                     << std::fixed << std::setprecision(1) 
                     << items_per_sec << " items/s" << std::flush;
            last_percent = percent;
        }
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        float items_per_sec = total / (duration.count() + 0.1f);
        
        std::cout << "\r" << description << ": 100% "
                 << "[" << total << "/" << total << "] "
                 << std::fixed << std::setprecision(1) 
                 << items_per_sec << " items/s "
                 << "Total time: " << duration.count() << "s\n" << std::flush;
    }
};

struct PairHash {
    template <typename T>
    std::size_t operator()(const std::pair<T, T>& p) const {
        return (static_cast<size_t>(p.first) << 32) | p.second;
    }
};

class TokenizedDataset {
private:
    std::vector<uint16_t> data;  // Store all tokens in memory
    size_t num_tokens;
    size_t context_length;
    size_t stride;
    bool is_root;
    size_t root_ctx_len;
    std::vector<size_t> valid_indices;
    size_t vocab_size;
    static constexpr size_t CHUNK_SIZE = 1024 * 1024;
    std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> final_counts;
    size_t num_bins;
    std::vector<std::vector<py::tuple>> bins;
    std::vector<size_t> bin_sums;

    std::unordered_map<uint16_t, std::unordered_map<uint16_t, size_t>> single_token_transitions;
    std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, size_t>, PairHash> pair_token_transitions;

    // ... existing private members ...
    std::shared_ptr<Trie_module_protV1> trie;

    std::vector<std::shared_ptr<Trie_module_protV1>> shard_tries;
    std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> pair_to_bin_mapping;



public:
    TokenizedDataset(const std::string& data_path, 
                    size_t context_length,
                    double data_percentage = 100,
                    size_t stride = 0,
                    const std::vector<std::pair<uint16_t, uint16_t>>& token_pairs = {},
                    const std::vector<size_t>& valid_indices = {},
                    bool is_root = false,
                    size_t root_ctx_len = 2,
                    size_t num_bins = 100) 
        : context_length(context_length)
        , stride(stride ? stride : context_length)
        , is_root(is_root)
        , root_ctx_len(root_ctx_len)
        , num_bins(num_bins)
        , bins(num_bins)
        , bin_sums(num_bins, 0)
    {
        std::cout << "\nInitializing TokenizedDataset...\n";
        
        // Read the entire file into memory
        std::ifstream file(data_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_path);
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Calculate number of tokens and resize data vector
        num_tokens = file_size / sizeof(uint16_t);
        num_tokens = static_cast<size_t>(num_tokens * (data_percentage / 100.0));
        data.resize(num_tokens);

        // Read all data at once
        std::cout << "Loading data into memory...\n";
        ProgressBar progress(num_tokens, "Loading tokens");
        
        const size_t batch_size = CHUNK_SIZE;
        std::vector<uint16_t> buffer(batch_size);
        
        for (size_t pos = 0; pos < num_tokens; pos += batch_size) {
            size_t tokens_to_read = std::min(batch_size, num_tokens - pos);
            file.read(reinterpret_cast<char*>(buffer.data()), tokens_to_read * sizeof(uint16_t));
            std::copy(buffer.begin(), buffer.begin() + tokens_to_read, data.begin() + pos);
            progress.update(tokens_to_read);
        }
        progress.finish();

        if (!valid_indices.empty()) {
            std::cout << "Processing valid indices... ";
            this->valid_indices = valid_indices;
            auto it = std::remove_if(this->valid_indices.begin(), this->valid_indices.end(),
                [this](size_t idx) { return idx + this->context_length + 1 > this->num_tokens; });
            this->valid_indices.erase(it, this->valid_indices.end());
        }

        calcVocabSize();

        std::cout << "\nDataset initialization complete:\n";
        std::cout << "Loaded " << num_tokens << " tokens\n";
        std::cout << "Total available windows: " << getNumWindows() << "\n";
        std::cout << "Vocabulary size: " << vocab_size << "\n\n";
    }

    py::dict analyze_window_startPairs(size_t prefix_length = 2) {
        const int num_threads = omp_get_max_threads();
        std::cout << "Using " << num_threads << " threads\n";

        size_t total_windows = (num_tokens - prefix_length - 1) / stride + 1;
        size_t windows_per_chunk = (total_windows + num_threads - 1) / num_threads;
        
        std::vector<std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash>> thread_counts(num_threads);
        
        ProgressBar progress(total_windows, "Analyzing windows");
        std::atomic<size_t> progress_counter{0};  // Use atomic for thread-safe counting
        
        #pragma omp parallel num_threads(num_threads)
        {
            int thread_id = omp_get_thread_num();
            size_t local_progress = 0;  // Track progress locally
            
            #pragma omp for schedule(dynamic, 1024)  // Add chunk size for better load balancing
            for (size_t window = 0; window < total_windows; window++) {
                size_t start_pos = window * stride;
                if (start_pos + prefix_length <= num_tokens) {
                    auto token_pair = std::make_pair(data[start_pos], data[start_pos + 1]);
                    thread_counts[thread_id][token_pair]++;
                }
                
                local_progress++;
                if (local_progress >= 10000) {  // Update progress less frequently
                    progress_counter += local_progress;
                    progress.update(local_progress);
                    local_progress = 0;
                }
            }
            
            // Update any remaining progress
            if (local_progress > 0) {
                progress_counter += local_progress;
                progress.update(local_progress);
            }
        }
        progress.finish();

        // Merge results
        std::cout << "Merging results from all threads...\n";
        for (int t = 0; t < num_threads; t++) {
            for (const auto& pair : thread_counts[t]) {
                final_counts[pair.first] += pair.second;
            }
        }

        // Convert to Python dictionary
        py::dict py_counts;
        for (const auto& pair : final_counts) {
            py::tuple token_tuple = py::make_tuple(pair.first.first, pair.first.second);
            py_counts[token_tuple] = pair.second;
        }
        
        std::cout << "Analysis complete. Found " << final_counts.size() << " unique token pairs.\n";
        return py_counts;
    }


    py::tuple distribute_tuples() {
        std::cout << "\nStarting distribution of " << final_counts.size() << " tuples...\n";
        
        // Convert to vector of pairs for easier sorting
        std::cout << "Converting to list...\n";
        std::vector<std::pair<std::pair<uint16_t, uint16_t>, size_t>> items;
        items.reserve(final_counts.size());
        
        ProgressBar progress(final_counts.size(), "Converting tuples");
        for (const auto& item : final_counts) {
            items.push_back(item);
            progress.update();
        }
        progress.finish();
        
        // Sort and print top 10
        std::cout << "\nTop 10 most frequent token pairs:\n";
        std::vector<std::pair<std::pair<uint16_t, uint16_t>, size_t>> sorted_items = items;
        std::sort(sorted_items.begin(), sorted_items.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
                
        for (size_t i = 0; i < std::min(size_t(10), sorted_items.size()); i++) {
            const auto& pair = sorted_items[i];
            std::cout << "Token pair (" << pair.first.first 
                    << "," << pair.first.second
                    << "): appeared " << pair.second << " times\n";
        }
        
        // Sort all items
        std::cout << "\nSorting all items...\n";
        std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Initialize bins
        // std::vector<std::vector<py::tuple>> bins(num_bins);
        // std::vector<size_t> bin_sums(num_bins, 0);
        
        // Distribute items
        std::cout << "\nDistributing items across " << num_bins << " bins...\n";
        ProgressBar dist_progress(items.size(), "Distributing items");
        for (size_t i = 0; i < items.size(); i++) {
            size_t bin_idx = i % num_bins;
            // Convert C++ pair to Python tuple before adding to bins
            py::tuple token_tuple = py::make_tuple(items[i].first.first, items[i].first.second);
            bins[bin_idx].push_back(token_tuple);
            bin_sums[bin_idx] += items[i].second;
            dist_progress.update();
        }
        dist_progress.finish();
        
        // Print final bin loads
        std::cout << "\nFinal bin loads:\n";
        for (size_t i = 0; i < num_bins; i++) {
            std::cout << "Bin " << i << ": " << bin_sums[i] << "\n";
        }
        
        // Convert to Python format
        py::list py_bins;
        for (const auto& bin : bins) {
            py::list py_bin;
            for (const auto& tuple_val : bin) {
                py_bin.append(tuple_val);
            }
            py_bins.append(py_bin);
        }
        
        py::list py_sums;
        for (const auto& sum : bin_sums) {
            py_sums.append(sum);
        }
        
        return py::make_tuple(py_bins, py_sums);
    }



    void save_pair_to_bin_mapping(const std::string& output_dir, 
                               const std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash>& pair_to_bin) {
        std::string filename = output_dir + "/pair_to_bin_mapping.bin";
        std::ofstream outfile(filename, std::ios::binary);
        
        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // First write the total number of pairs
        size_t num_pairs = pair_to_bin.size();
        outfile.write(reinterpret_cast<const char*>(&num_pairs), sizeof(num_pairs));

        // Write each pair and its bin index
        for (const auto& mapping : pair_to_bin) {
            // Write first token of the pair
            outfile.write(reinterpret_cast<const char*>(&mapping.first.first), sizeof(uint16_t));
            
            // Write second token of the pair
            outfile.write(reinterpret_cast<const char*>(&mapping.first.second), sizeof(uint16_t));
            
            // Write the bin index
            outfile.write(reinterpret_cast<const char*>(&mapping.second), sizeof(size_t));
        }

        outfile.close();
        std::cout << "Pair to bin mapping saved to: " << filename << std::endl;
    }

    // Function to load the mapping back
    std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> 
    load_pair_to_bin_mapping(const std::string& output_dir) {
        std::string filename = output_dir + "/pair_to_bin_mapping.bin";
        std::ifstream infile(filename, std::ios::binary);
        
        if (!infile.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> pair_to_bin;

        // Read the number of pairs
        size_t num_pairs;
        infile.read(reinterpret_cast<char*>(&num_pairs), sizeof(num_pairs));

        // Read each pair and its bin index
        for (size_t i = 0; i < num_pairs; i++) {
            uint16_t first_token, second_token;
            size_t bin_idx;

            // Read the pair
            infile.read(reinterpret_cast<char*>(&first_token), sizeof(uint16_t));
            infile.read(reinterpret_cast<char*>(&second_token), sizeof(uint16_t));
            
            // Read the bin index
            infile.read(reinterpret_cast<char*>(&bin_idx), sizeof(size_t));

            // Store in the map
            pair_to_bin[std::make_pair(first_token, second_token)] = bin_idx;
        }

        infile.close();
        std::cout << "Loaded " << pair_to_bin.size() << " pair mappings from: " << filename << std::endl;
        return pair_to_bin;
    }

    void save_pair_locations(const std::string& output_dir) {
        const int num_threads = omp_get_max_threads();
        std::cout << "Using " << num_threads << " threads for location analysis\n";

        // Create pair to bin mapping (same as before)
        std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> pair_to_bin;
        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            for (const auto& py_tuple : bins[bin_idx]) {
                pair_to_bin[std::make_pair(
                    py::cast<uint16_t>(py_tuple[0]),
                    py::cast<uint16_t>(py_tuple[1])
                )] = bin_idx;
            }
        }

        // Add this line right here to save the mapping
        save_pair_to_bin_mapping(output_dir, pair_to_bin);

        // Create output files and their mutexes
        std::vector<std::ofstream> bin_files(bins.size());
        std::vector<std::mutex> bin_mutexes(bins.size());
        
        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            std::string filename = output_dir + "/shard" + std::to_string(bin_idx) + "/shuffled_indices_locations.txt";
            bin_files[bin_idx].open(filename, std::ios::out | std::ios::binary);
            if (!bin_files[bin_idx].is_open()) {
                throw std::runtime_error("Failed to open output file: " + filename);
            }
        }

        // Smaller buffer size to prevent memory issues
        const size_t BUFFER_SIZE = 64 * 1024;  // 64KB buffer per thread per bin
        std::vector<std::vector<std::string>> thread_buffers(num_threads, 
            std::vector<std::string>(bins.size()));
        std::vector<std::vector<size_t>> buffer_sizes(num_threads, 
            std::vector<size_t>(bins.size(), 0));

        size_t total_windows = (num_tokens - 1) / stride + 1;
        ProgressBar progress(total_windows, "Processing locations");
        std::atomic<size_t> progress_counter{0};

        #pragma omp parallel num_threads(num_threads)
        {
            int thread_id = omp_get_thread_num();
            size_t local_progress = 0;
            char number_buffer[32];
            
            #pragma omp for schedule(dynamic, 1024)
            for (size_t i = 0; i < num_tokens - 1; i += stride) {
                auto token_pair = std::make_pair(data[i], data[i + 1]);
                auto bin_it = pair_to_bin.find(token_pair);
                
                if (bin_it != pair_to_bin.end()) {
                    size_t bin_idx = bin_it->second;
                    int len = snprintf(number_buffer, sizeof(number_buffer), "%zu\n", i);
                    
                    // Check if buffer needs to be flushed
                    if (buffer_sizes[thread_id][bin_idx] + len > BUFFER_SIZE) {
                        // Flush buffer to file
                        std::lock_guard<std::mutex> lock(bin_mutexes[bin_idx]);
                        bin_files[bin_idx].write(thread_buffers[thread_id][bin_idx].data(), 
                                            thread_buffers[thread_id][bin_idx].size());
                        thread_buffers[thread_id][bin_idx].clear();
                        buffer_sizes[thread_id][bin_idx] = 0;
                    }
                    
                    thread_buffers[thread_id][bin_idx].append(number_buffer, len);
                    buffer_sizes[thread_id][bin_idx] += len;
                }

                local_progress++;
                if (local_progress >= 10000) {
                    progress_counter += local_progress;
                    progress.update(local_progress);
                    local_progress = 0;
                }
            }

            // Flush remaining buffers
            for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
                if (buffer_sizes[thread_id][bin_idx] > 0) {
                    std::lock_guard<std::mutex> lock(bin_mutexes[bin_idx]);
                    bin_files[bin_idx].write(thread_buffers[thread_id][bin_idx].data(),
                                        thread_buffers[thread_id][bin_idx].size());
                }
            }

            if (local_progress > 0) {
                progress_counter += local_progress;
                progress.update(local_progress);
            }
        }
        progress.finish();

        // Close all files
        for (auto& file : bin_files) {
            if (file.is_open()) {
                file.close();
            }
        }
    }



    size_t __len__() const {
        return getNumWindows();
    }

    torch::Tensor __getitem__(size_t idx) {
        size_t start_idx;
        if (!valid_indices.empty()) {
            start_idx = valid_indices[idx] * stride;
        } else {
            start_idx = idx * stride;
        }

        size_t window_size = is_root ? root_ctx_len + 1 : context_length + 1;
        std::vector<int64_t> window(window_size);
        
        // Direct memory access instead of file reading
        std::copy(data.begin() + start_idx, 
                 data.begin() + start_idx + window_size,
                 window.begin());

        return torch::tensor(window);
    }

    size_t getNumWindows() const {
        if (!valid_indices.empty()) {
            return valid_indices.size();
        }
        size_t effective_length = is_root ? root_ctx_len : context_length;
        return std::max<size_t>(0, (num_tokens - effective_length - 1) / stride + 1);
    }

    size_t getVocabSize() const {
        return vocab_size;
    }


    void save_transitions_to_file(const std::string& output_dir) {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(output_dir);
        
        // Save single token transitions
        {
            std::ofstream single_file(output_dir + "/root_single_transitions.bin", std::ios::binary);
            if (!single_file) throw std::runtime_error("Cannot open single transitions file for writing");
            
            // Write number of tokens
            size_t num_tokens = single_token_transitions.size();
            single_file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));
            
            // Write each token's transitions
            for (const auto& [token, transitions] : single_token_transitions) {
                // Write the token
                single_file.write(reinterpret_cast<const char*>(&token), sizeof(token));
                
                // Write number of transitions
                size_t num_transitions = transitions.size();
                single_file.write(reinterpret_cast<const char*>(&num_transitions), sizeof(num_transitions));
                
                // Write each transition
                for (const auto& [next_token, count] : transitions) {
                    single_file.write(reinterpret_cast<const char*>(&next_token), sizeof(next_token));
                    single_file.write(reinterpret_cast<const char*>(&count), sizeof(count));
                }
            }
        }
        
        // Save pair transitions
        {
            std::ofstream pair_file(output_dir + "/root_pair_transitions.bin", std::ios::binary);
            if (!pair_file) throw std::runtime_error("Cannot open pair transitions file for writing");
            
            // Write number of pairs
            size_t num_pairs = pair_token_transitions.size();
            pair_file.write(reinterpret_cast<const char*>(&num_pairs), sizeof(num_pairs));
            
            // Write each pair's transitions
            for (const auto& [token_pair, transitions] : pair_token_transitions) {
                // Write the token pair
                pair_file.write(reinterpret_cast<const char*>(&token_pair.first), sizeof(token_pair.first));
                pair_file.write(reinterpret_cast<const char*>(&token_pair.second), sizeof(token_pair.second));
                
                // Write number of transitions
                size_t num_transitions = transitions.size();
                pair_file.write(reinterpret_cast<const char*>(&num_transitions), sizeof(num_transitions));
                
                // Write each transition
                for (const auto& [next_token, count] : transitions) {
                    pair_file.write(reinterpret_cast<const char*>(&next_token), sizeof(next_token));
                    pair_file.write(reinterpret_cast<const char*>(&count), sizeof(count));
                }
            }
        }
        
        std::cout << "Transitions saved to " << output_dir << std::endl;
    }

    void load_transitions_from_file(const std::string& input_dir) {
        // Clear existing transitions
        single_token_transitions.clear();
        pair_token_transitions.clear();
        
        // Load single token transitions
        {
            std::ifstream single_file(input_dir + "/root_single_transitions.bin", std::ios::binary);
            if (!single_file) throw std::runtime_error("Cannot open single transitions file for reading");
            
            // Read number of tokens
            size_t num_tokens;
            single_file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
            
            // Read each token's transitions
            for (size_t i = 0; i < num_tokens; i++) {
                uint16_t token;
                single_file.read(reinterpret_cast<char*>(&token), sizeof(token));
                
                size_t num_transitions;
                single_file.read(reinterpret_cast<char*>(&num_transitions), sizeof(num_transitions));
                
                auto& transitions = single_token_transitions[token];
                for (size_t j = 0; j < num_transitions; j++) {
                    uint16_t next_token;
                    size_t count;
                    single_file.read(reinterpret_cast<char*>(&next_token), sizeof(next_token));
                    single_file.read(reinterpret_cast<char*>(&count), sizeof(count));
                    transitions[next_token] = count;
                }
            }
        }
        
        // Load pair transitions
        {
            std::ifstream pair_file(input_dir + "/root_pair_transitions.bin", std::ios::binary);
            if (!pair_file) throw std::runtime_error("Cannot open pair transitions file for reading");
            
            // Read number of pairs
            size_t num_pairs;
            pair_file.read(reinterpret_cast<char*>(&num_pairs), sizeof(num_pairs));
            
            // Read each pair's transitions
            for (size_t i = 0; i < num_pairs; i++) {
                uint16_t first_token, second_token;
                pair_file.read(reinterpret_cast<char*>(&first_token), sizeof(first_token));
                pair_file.read(reinterpret_cast<char*>(&second_token), sizeof(second_token));
                
                size_t num_transitions;
                pair_file.read(reinterpret_cast<char*>(&num_transitions), sizeof(num_transitions));
                
                auto& transitions = pair_token_transitions[std::make_pair(first_token, second_token)];
                for (size_t j = 0; j < num_transitions; j++) {
                    uint16_t next_token;
                    size_t count;
                    pair_file.read(reinterpret_cast<char*>(&next_token), sizeof(next_token));
                    pair_file.read(reinterpret_cast<char*>(&count), sizeof(count));
                    transitions[next_token] = count;
                }
            }
        }
        
        std::cout << "Transitions loaded from " << input_dir << std::endl;
        std::cout << "Loaded " << single_token_transitions.size() << " single token transitions" << std::endl;
        std::cout << "Loaded " << pair_token_transitions.size() << " pair transitions" << std::endl;
    }


    py::dict analyze_token_transitions(const std::string& save_dir = "", bool read_from_file = false) {
        if (read_from_file) {
            if (save_dir.empty()) {
                throw std::runtime_error("save_dir must be provided when read_from_file is true");
            }
            load_transitions_from_file(save_dir);
        } else {
            std::cout << "Starting token transition analysis...\n";
            
            // Limit number of threads to reduce memory overhead
            const int max_threads = 5;  // Adjust based on your system's memory
            const int num_threads = std::min(max_threads, omp_get_max_threads());
            std::cout << "Using " << num_threads << " threads\n";
            
            single_token_transitions.clear();
            pair_token_transitions.clear();
            
            // Process in batches to manage memory
            const size_t batch_size = 10000000;  // Process 10M tokens at a time
            const size_t total_tokens = num_tokens - 2;
            const size_t num_batches = (total_tokens + batch_size - 1) / batch_size;
            
            ProgressBar progress(total_tokens, "Analyzing transitions");
            std::atomic<size_t> progress_counter{0};
            
            for (size_t batch = 0; batch < num_batches; batch++) {
                const size_t start_idx = batch * batch_size;
                const size_t end_idx = std::min(start_idx + batch_size, total_tokens);
                
                // Thread-local maps for this batch
                std::vector<std::unordered_map<uint16_t, std::unordered_map<uint16_t, size_t>>> thread_single_transitions(num_threads);
                std::vector<std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, size_t>, PairHash>> thread_pair_transitions(num_threads);
                
                #pragma omp parallel num_threads(num_threads)
                {
                    int thread_id = omp_get_thread_num();
                    size_t local_progress = 0;
                    
                    #pragma omp for schedule(dynamic, 1024)
                    for (size_t i = start_idx; i < end_idx; i++) {
                        uint16_t current_token = data[i];
                        uint16_t next_token = data[i + 1];
                        uint16_t next_next_token = data[i + 2];
                        
                        // Update thread-local maps
                        thread_single_transitions[thread_id][current_token][next_token]++;
                        thread_pair_transitions[thread_id][std::make_pair(current_token, next_token)][next_next_token]++;
                        
                        local_progress++;
                        if (local_progress >= 10000) {
                            progress_counter += local_progress;
                            progress.update(local_progress);
                            local_progress = 0;
                        }
                    }
                    
                    if (local_progress > 0) {
                        progress_counter += local_progress;
                        progress.update(local_progress);
                    }
                }
                
                // Merge batch results into global maps
                std::cout << "\nMerging batch " << (batch + 1) << "/" << num_batches << "...\n";
                for (int t = 0; t < num_threads; t++) {
                    for (const auto& [token, transitions] : thread_single_transitions[t]) {
                        auto& global_transitions = single_token_transitions[token];
                        for (const auto& [next_token, count] : transitions) {
                            global_transitions[next_token] += count;
                        }
                    }
                    
                    for (const auto& [token_pair, transitions] : thread_pair_transitions[t]) {
                        auto& global_transitions = pair_token_transitions[token_pair];
                        for (const auto& [next_token, count] : transitions) {
                            global_transitions[next_token] += count;
                        }
                    }
                }
                
                // Clear thread-local maps after merging
                thread_single_transitions.clear();
                thread_pair_transitions.clear();
            }
            
            progress.finish();
            
            // Save to file if directory is provided and we're not reading from file
            if (!save_dir.empty()) {
                save_transitions_to_file(save_dir);
            }
        }
        
        // Convert to Python dictionary
        py::dict result;
        
        // Convert single token transitions
        std::cout << "Converting single token transitions...\n";
        for (const auto& [token, transitions] : single_token_transitions) {
            py::dict token_transitions;
            for (const auto& [next_token, count] : transitions) {
                token_transitions[py::cast(next_token)] = count;
            }
            result[py::cast(token)] = token_transitions;
        }
        
        // Convert pair transitions
        std::cout << "Converting pair transitions...\n";
        for (const auto& [token_pair, transitions] : pair_token_transitions) {
            py::tuple key = py::make_tuple(token_pair.first, token_pair.second);
            py::dict pair_transitions;
            for (const auto& [next_token, count] : transitions) {
                pair_transitions[py::cast(next_token)] = count;
            }
            result[key] = pair_transitions;
        }
        
        std::cout << "Analysis complete.\n";
        std::cout << "Found " << single_token_transitions.size() << " unique single tokens\n";
        std::cout << "Found " << pair_token_transitions.size() << " unique token pairs\n";
        
        return result;
    }
    

    std::shared_ptr<Trie_module_protV1> create_and_get_trie(const std::string& trie_path, size_t initial_size_gb = 1, size_t batch_size = 1024) {
        std::cout << "Creating Trie from dataset ..." << std::endl;
        std::cout << "Trie path is set to be " << trie_path << std::endl;
        
        // Create Trie instance directly
        trie = std::make_shared<Trie_module_protV1>(trie_path, initial_size_gb, context_length);
        
        // Process dataset in batches
        ProgressBar progress(getNumWindows(), "Building Trie");
        
        for (size_t i = 0; i < getNumWindows(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, getNumWindows() - i);
            
            // Gather windows for this batch
            std::vector<std::vector<int64_t>> batch_sequences;
            for (size_t j = 0; j < current_batch_size; j++) {
                std::vector<int64_t> window(context_length + 1); // +1 for target
                std::copy(data.begin() + (i + j) * stride, 
                        data.begin() + (i + j) * stride + context_length + 1,
                        window.begin());
                batch_sequences.push_back(window);
            }
            
            // Insert into Trie directly using C++ method
            trie->insert(batch_sequences);
            
            progress.update(current_batch_size);
        }
        progress.finish();
        
        return trie;
    }


    // Add this method to load all shard tries
    void load_shard_tries(const std::string& base_trie_path, size_t num_shards) {
        std::cout << "Loading tries for " << num_shards << " shards..." << std::endl;
        
        // Initialize tries vector
        shard_tries.resize(num_shards);
        
        // Load all tries in parallel
        #pragma omp parallel for schedule(dynamic)
        for (size_t shard_idx = 0; shard_idx < num_shards; shard_idx++) {
            std::string trie_path = base_trie_path + "/shard" + std::to_string(shard_idx) + "/Trie"  + std::to_string(shard_idx) + "_MT";
            shard_tries[shard_idx] = std::make_shared<Trie_module_protV1>(trie_path);
            std::cout << "Loaded trie for shard " << shard_idx << std::endl;
        }

        std::cout << "All shard tries loaded successfully" << std::endl;
    }

    // Method to load pair mappings and initialize everything
    void initialize_shard_system(const std::string& base_trie_path, 
                            const std::string& mapping_dir, 
                            const std::string& transitions_dir,
                            size_t num_shards) {
        // Load pair to bin mapping
        pair_to_bin_mapping = load_pair_to_bin_mapping(mapping_dir);
        std::cout << "Loaded pair to bin mapping with " << pair_to_bin_mapping.size() << " entries" << std::endl;
        
        // Load transitions
        load_transitions_from_file(transitions_dir);
        
        // Load all tries
        load_shard_tries(base_trie_path, num_shards);
    }


private:
    void calcVocabSize() {
        vocab_size = 0;
        ProgressBar progress(num_tokens, "Calculating vocab size");
        
        for (size_t i = 0; i < num_tokens; ++i) {
            vocab_size = std::max(vocab_size, static_cast<size_t>(data[i]));
            progress.update();
        }
        progress.finish();
        
        vocab_size++;
    }
};


py::tuple create_dataloader(const std::string& data_path,
                          size_t context_length,
                          size_t batch_size,
                          double data_percentage = 100,
                          size_t stride = 0,
                          const std::vector<std::pair<uint16_t, uint16_t>>& token_pairs = {},
                          const std::vector<size_t>& valid_indices = {},
                          bool shuffle = true,
                          bool is_root = false,
                          size_t root_ctx_len = 2,
                          size_t num_bins = 100) {
    
    auto dataset = std::make_shared<TokenizedDataset>(
        data_path, context_length, data_percentage, stride,
        token_pairs, valid_indices, is_root, root_ctx_len, num_bins
    );
    
    py::object DataLoader = py::module::import("torch.utils.data").attr("DataLoader");
    py::object dataloader = DataLoader(
        dataset,
        "batch_size"_a = batch_size,
        "shuffle"_a = shuffle,
        "pin_memory"_a = true,
        "num_workers"_a = 4
    );
    
    return py::make_tuple(dataloader, dataset->getVocabSize());
}

PYBIND11_MODULE(Trie_dataloader, m) {
    m.doc() = "Fast tokenized dataset implementation in C++";

    py::class_<Trie_module_protV1>(m, "Trie_module_protV1")
        .def(py::init<const std::string&, size_t, int64_t>())
        .def(py::init<const std::string&>())
        .def("insert", &Trie_module_protV1::insert)  // Now just returns double
        .def("retrieve_softlabel", &Trie_module_protV1::retrieve_softlabel)
        .def("get_memory_usage", &Trie_module_protV1::get_memory_usage)
        .def("get_allocated_size", &Trie_module_protV1::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_module_protV1::get_num_unique_contexts)
        .def("get_num_total_contexts", &Trie_module_protV1::get_num_total_contexts)
        .def("load_metadata", &Trie_module_protV1::load_metadata)
        .def("save_metadata", &Trie_module_protV1::save_metadata)
        .def("serialize_to_mmap", &Trie_module_protV1::serialize_to_mmap)
        .def("get_memory_stats", &Trie_module_protV1::get_memory_stats)
        .def("print_memory_stats", &Trie_module_protV1::print_memory_stats);

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("distributions", &InsertResult::distributions)
        .def_readonly("execution_time_ms", &InsertResult::execution_time_ms);

    py::class_<TokenizedDataset, std::shared_ptr<TokenizedDataset>>(m, "TokenizedDataset")
        .def(py::init<const std::string&, size_t, double, size_t,
            const std::vector<std::pair<uint16_t, uint16_t>>&,
            const std::vector<size_t>&, bool, size_t, size_t>(),  // Added size_t for num_bins
            py::arg("data_path"),
            py::arg("context_length"),
            py::arg("data_percentage") = 100,
            py::arg("stride") = 0,
            py::arg("token_pairs") = std::vector<std::pair<uint16_t, uint16_t>>(),
            py::arg("valid_indices") = std::vector<size_t>(),
            py::arg("is_root") = false,
            py::arg("root_ctx_len") = 2,
            py::arg("num_bins") = 100)  // Added num_bins argument
        .def("__len__", &TokenizedDataset::__len__)
        .def("__getitem__", &TokenizedDataset::__getitem__)
        .def("analyze_window_startPairs", &TokenizedDataset::analyze_window_startPairs,
            py::arg("prefix_length") = 2)
        .def("save_pair_locations", &TokenizedDataset::save_pair_locations,
            py::arg("output_dir"))
        .def("distribute_tuples", &TokenizedDataset::distribute_tuples)
        .def("getVocabSize", &TokenizedDataset::getVocabSize)
        .def("getNumWindows", &TokenizedDataset::getNumWindows)
        .def("analyze_token_transitions", &TokenizedDataset::analyze_token_transitions,
            py::arg("save_dir") = "",
            py::arg("read_from_file") = false)
        .def("save_transitions_to_file", &TokenizedDataset::save_transitions_to_file)
        .def("load_transitions_from_file", &TokenizedDataset::load_transitions_from_file)
        .def("create_and_get_trie", &TokenizedDataset::create_and_get_trie,
            py::arg("trie_path"),
            py::arg("initial_size_gb") = 1,
            py::arg("batch_size") = 1024)
        .def("initialize_shard_system", &TokenizedDataset::initialize_shard_system,
            py::arg("base_trie_path"),
            py::arg("mapping_dir"),
            py::arg("transitions_dir"),
            py::arg("num_shards"));

    // Create_dataloader function binding remains the same
    m.def("create_dataloader", &create_dataloader,
          py::arg("data_path"),
          py::arg("context_length"),
          py::arg("batch_size"),
          py::arg("data_percentage") = 100,
          py::arg("stride") = 0,
          py::arg("token_pairs") = std::vector<std::pair<uint16_t, uint16_t>>(),
          py::arg("valid_indices") = std::vector<size_t>(),
          py::arg("shuffle") = true,
          py::arg("is_root") = false,
          py::arg("root_ctx_len") = 2,
          py::arg("num_bins") = 100);
}