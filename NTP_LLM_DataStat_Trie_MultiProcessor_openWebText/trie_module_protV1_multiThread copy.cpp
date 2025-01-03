#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdint>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sys/mman.h>
#include <algorithm>
#include <unistd.h> // for sleep()
#include <omp.h>

#include <iomanip>

// #include <nlohmann/json.hpp>  // Include a JSON library like nlohmann/json for C++

namespace py = pybind11;

#define DEBUG_PRINT(x) std::cout << x << std::endl


struct RAMTrieNode {
    int64_t count;
    uint16_t num_children;
    std::pair<uint16_t, int64_t>* children;  // Raw array instead of vector
    uint16_t children_capacity;
    uint16_t node_level;
};
std::vector<RAMTrieNode> nodes;  // Our main container during construction

// Memory mapped structure for final format
struct MMAPTrieNode {
    int64_t count;
    uint16_t num_children;
    int64_t children_offset;
    uint16_t node_level;
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
        for (int i = 0; i < std::min(node->num_children, (uint16_t)5); i++) {
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



struct InsertResult {
    py::list result;
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
    uint16_t context_length;
    
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
    std::pair<uint16_t, int64_t>* get_children(RAMTrieNode* node) {
        return node->children;
    }

    size_t allocate_node(uint16_t parent_level) {
        size_t new_index = nodes.size();
        nodes.push_back(createNode(parent_level));
        return new_index;
    }

    // Fix node creation
    RAMTrieNode createNode(uint16_t parent_level) {
        return RAMTrieNode{
            0,          // count
            0,          // num_children
            nullptr,    // children pointer (initialize as null)
            0,          // children_capacity
            parent_level + 1  // node_level
        };
    }

    int64_t find_child(RAMTrieNode* node, uint16_t value) {
        if (node->children == nullptr || node->num_children == 0) {
            return -1;
        }

        // Binary search
        uint16_t left = 0, right = node->num_children - 1;
        while (left <= right) {
            uint16_t mid = left + (right - left) / 2;
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
            // First allocation
            size_t initial_size = std::max(needed_size, size_t(8));  // Start with at least 8
            node->children = new std::pair<uint16_t, int64_t>[initial_size];
            node->children_capacity = initial_size;
        }
        else if (node->num_children + needed_size > node->children_capacity) {
            // Need to grow
            size_t new_capacity = node->children_capacity * 2;
            auto new_children = new std::pair<uint16_t, int64_t>[new_capacity];
            // Use copy instead of memmove for proper object copying
            for (uint16_t i = 0; i < node->num_children; i++) {
                new_children[i] = node->children[i];
            }
            delete[] node->children;
            node->children = new_children;
            node->children_capacity = new_capacity;
        }
        return node->children_capacity;
    }

    // And modify insert_child to use proper copying
    void insert_child(RAMTrieNode* node, uint16_t value, uint16_t child_index) {
        // Ensure we have space
        if (node->children == nullptr || node->num_children >= node->children_capacity) {
            allocate_children(node, 1);
        }

        // Find insertion position
        uint16_t insert_pos = 0;
        while (insert_pos < node->num_children && node->children[insert_pos].first < value) {
            insert_pos++;
        }

        // Shift existing elements using proper copying
        for (uint16_t i = node->num_children; i > insert_pos; i--) {
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
    Trie_module_protV1(const std::string& fname, size_t initial_size_gb, uint16_t context_length) 
        : filename(fname + ".bin"), 
        context_length(context_length),
        is_construction_mode(true) {
        
        metadata_filename = filename + "_metadata.bin";

        // Reserve space based on expected number of nodes
        size_t expected_nodes = initial_size_gb * 1024 * 1024 * 1024 / sizeof(RAMTrieNode);
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
                static_cast<uint16_t>(nodes[i].node_level)
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

        // Save additional array data in binary format
        std::ofstream per_level_data(filename.substr(0, filename.length() - 4) + "_level_data.bin", 
                                    std::ios::binary);
        if (!per_level_data.is_open()) {
            throw std::runtime_error("Failed to open level data file for writing");
        }

        // Write all level-specific data
        per_level_data.close();

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
            uint16_t level;
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
    
    std::vector<std::unordered_map<int64_t, double>> insert_context(const torch::Tensor& tensor, int64_t column, bool return_prob_distr) {   
        std::vector<std::unordered_map<int64_t, double>> distributions;
        auto accessor = tensor.accessor<int64_t, 2>();
        uint16_t current_level = 0;
        size_t current_index = 0;  // Start at root

        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];
            
            RAMTrieNode* current = &nodes[current_index];
            
            if (j > 0) {
                num_total_contexts.fetch_add(1);
            }
            
            // Find child if it exists
            DEBUG_PRINT("find_child");
            int64_t child_index = -1;
            if (current->children != nullptr && current->num_children > 0) {
                child_index = find_child(current, value);
            }

            DEBUG_PRINT("insert_child");
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

            DEBUG_PRINT("node assignments");
            // Update counts
            nodes[current_index].count++;
            current_level++;

            if (return_prob_distr) {
                // Create distribution
                std::unordered_map<int64_t, double> current_distribution;
                if (current->children != nullptr) {
                    for (uint16_t i = 0; i < current->num_children; i++) {
                        RAMTrieNode* child = &nodes[current->children[i].second];
                        current_distribution[current->children[i].first] = static_cast<double>(child->count);
                    }
                }
                distributions.push_back(current_distribution);
            }
        }

        return distributions;
    }


    InsertResult insert(torch::Tensor tensor, bool return_prob_distr) {
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Get system capabilities and set thread count
        int num_procs = omp_get_num_procs();
        // int num_threads = num_procs;  // Using at most 8 threads
        int num_threads = 1;
        omp_set_num_threads(num_threads);


        // Calculate batch size per thread
        int batch_size = tensor.size(0);
        int chunk_size = std::max(1, batch_size / num_threads);

        std::vector<std::vector<std::unordered_map<int64_t, double>>> soft_label_distributions(batch_size);
        
        // Dynamic scheduling with calculated chunk size
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int col = 0; col < batch_size; col++) {
            int thread_id = omp_get_thread_num();
            // if (col % chunk_size == 0) {
            //     DEBUG_PRINT("Thread " << thread_id << " processing chunk starting at " << col);
            // }
            soft_label_distributions[col] = this->insert_context(tensor, col, return_prob_distr);
        }

        // Single-threaded conversion to Python list (as before)
        py::list soft_label_list;
        for (const auto& seq_distributions : soft_label_distributions) {
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

        // Check if we're close to memory limit after insertion
        // if (is_close_to_memory_limit()) {
        //     save_metadata();
        // }

        // End timing and calculate duration
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        return InsertResult{
            soft_label_list,
            static_cast<double>(duration.count())
        };
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
        // Ensure that the input tensor is 2D and of type int64 (torch::kInt64)
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");

        std::vector<std::vector<std::unordered_map<int64_t, double>>> soft_label_distributions(tensor.size(0));
        
        // Process all columns sequentially in a single thread
        for (int col = 0; col < tensor.size(0); col++) {
            soft_label_distributions[col] = this->retrieve_context_softlabel(tensor, col);
        }

        // Convert the results to Python list
        py::list soft_label_list;
        for (const auto& seq_distributions : soft_label_distributions) {
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
                    total_bytes += node.children_capacity * sizeof(std::pair<uint16_t, int64_t>);
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
                    total_size += node.children_capacity * sizeof(std::pair<uint16_t, int64_t>);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Trie_module_protV1>(m, "Trie_module_protV1")
        .def(py::init<const std::string&, size_t, uint16_t>())
        .def(py::init<const std::string&>())
        .def("insert", &Trie_module_protV1::insert)
        .def("retrieve_softlabel", &Trie_module_protV1::retrieve_softlabel)
        .def("get_memory_usage", &Trie_module_protV1::get_memory_usage)
        .def("get_allocated_size", &Trie_module_protV1::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_module_protV1::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_module_protV1::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("load_metadata", &Trie_module_protV1::load_metadata)
        .def("save_metadata", &Trie_module_protV1::save_metadata)
        .def("serialize_to_mmap", &Trie_module_protV1::serialize_to_mmap);
        

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("result", &InsertResult::result)
        .def_readonly("execution_time_ms", &InsertResult::execution_time_ms);

}