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

// Define a custom hash function for std::vector<int64_t>
namespace std {
    template <>
    struct hash<std::vector<int64_t>> {
        size_t operator()(const std::vector<int64_t>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= std::hash<int64_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

#define DEBUG_PRINT(x) std::cout << x << std::endl



template<typename T>
class MemMapArray {
public:
    MemMapArray(const std::string& filename, size_t size) : filename(filename), size(size) {
        // Open file for reading and writing, create if doesn't exist
        fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for memory mapping");
        }

        // Set the file size to hold 'size' number of elements of type T
        if (ftruncate(fd, size * sizeof(T)) == -1) {
            throw std::runtime_error("Failed to set file size");
        }

        // Memory-map the file
        mapped_memory = (T*)mmap(nullptr, size * sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map memory");
        }

        // Initialize the memory to zero
        std::memset(mapped_memory, 0, size * sizeof(T));

        // DEBUG_PRINT("Initiated my memaparray. size is " << size << " .");
    }


    // Constructor for loading an existing memory-mapped array
    MemMapArray(const std::string& filename) : filename(filename), size(0), fd(-1), mapped_memory(nullptr) {
        // Open the existing file in read-write mode
        fd = open(filename.c_str(), O_RDWR);
        if (fd == -1) {
            throw std::runtime_error("Failed to open existing file for memory mapping");
        }

        // Get the size of the file
        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size");
        }

        size = st.st_size / sizeof(T);

        // Memory-map the file
        mapped_memory = (T*)mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map memory");
        }
    }

    ~MemMapArray() {
        if (mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, size * sizeof(T));
        }
        if (fd != -1) {
            close(fd);
        }
    }

    T& operator[](size_t index) {
        // DEBUG_PRINT("Operator. size is " << size << " .");
        // DEBUG_PRINT("Operator. index is " << index << " .");
        if (index >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return mapped_memory[index];
    }

    size_t getSize() const {
        return size;
    }

private:
    std::string filename;
    size_t size;
    int fd;
    T* mapped_memory;
};




struct TrieNode {
    int64_t count;
    int64_t node_level;
    int64_t node_index;
    std::vector<std::pair<int64_t, TrieNode*>> children;  // Changed to raw pointer
    
    TrieNode(int64_t level = 0) : count(0), node_level(level), node_index(-1) {}
    
    // // Add destructor to clean up children
    ~TrieNode() {
        std::unordered_set<TrieNode*> unique_children;
        for (auto& child : children) {
            unique_children.insert(child.second);
        }
        
        for (auto child : unique_children) {
            delete child;
        }
        std::cout << "Destroyed node at level " << node_level << std::endl;
    }
    // ~TrieNode() {
    //     std::cout << "Destroying node at level " << node_level << std::endl;
    //     for (auto& child : children) {
    //         delete child.second;
    //     }
    // }
    // ~TrieNode() {
    //     std::cout << "Destroying node at level " << node_level << std::endl;
    //     // Don't delete children here - let the vector handle cleanup
    //     children.clear();  // Just clear the vector
    // }

    // Serialization stays mostly the same
    void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(&node_level), sizeof(node_level));
        out.write(reinterpret_cast<const char*>(&node_index), sizeof(node_index));
        
        int64_t num_children = children.size();
        out.write(reinterpret_cast<const char*>(&num_children), sizeof(num_children));
        
        for (const auto& child : children) {
            out.write(reinterpret_cast<const char*>(&child.first), sizeof(child.first));
            child.second->serialize(out);
        }
    }
    
    // Deserialization changes to use raw pointers
    static TrieNode* deserialize(std::ifstream& in) {
        TrieNode* node = new TrieNode();
        
        in.read(reinterpret_cast<char*>(&node->count), sizeof(node->count));
        in.read(reinterpret_cast<char*>(&node->node_level), sizeof(node->node_level));
        in.read(reinterpret_cast<char*>(&node->node_index), sizeof(node->node_index));
        
        int64_t num_children;
        in.read(reinterpret_cast<char*>(&num_children), sizeof(num_children));
        
        for (int64_t i = 0; i < num_children; i++) {
            int64_t value;
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            TrieNode* child = deserialize(in);
            node->children.emplace_back(value, child);
        }
        
        return node;
    }
};


void printTrieNode(const TrieNode* node) {
    std::cout << "TrieNode Information:" << std::endl;
    std::cout << "Count: " << node->count << std::endl;
    std::cout << "Node Level: " << node->node_level << std::endl;
    std::cout << "Node Index: " << node->node_index << std::endl;
}



struct InsertResult {
    py::list result;
    double execution_time_ms;
};

struct EntropyResult {
    double entropy;
    int64_t total_count;
    int64_t number_of_oneHots;
    py::list entropy_results; // Optional: We can add more return values if needed
    EntropyResult(double e, int64_t c, int64_t o ): entropy(e), total_count(c), number_of_oneHots(o) {}
};

class RAMTrie {
private:
    TrieNode* root;
    int64_t context_length;
    std::atomic<int64_t> node_counter;
    std::atomic<int64_t> num_unique_contexts;
    std::atomic<int64_t> num_total_contexts;

    std::atomic<size_t> total_memory_usage;  // Track memory usage
    
    // Statistics tracking
    std::map<int64_t, int> num_unique_contexts_per_level;
    std::map<int64_t, int> num_total_contexts_per_level;
    std::map<int64_t, double> entropy_per_level;
    std::vector<double> countLog_array;
    std::vector<int> ctxLen_array;
    std::vector<int> ctxCount_array;

    // Log calculation caches
    const size_t size_logcalc_memory = 100000000;  // Same size as original
    std::vector<double> logcalc_memory_insert;
    std::vector<double> logcalc_memory_entropy;
    std::mutex logcalc_mutex;  // For thread safety when updating log caches

    std::map<int64_t, double> count_per_level;
    std::map<int64_t, int> num_oneHots_per_level;

    std::atomic<bool> is_deleted{false};


    TrieNode* find_or_create_node(TrieNode* current, int64_t value) {
        for (auto& child : current->children) {
            if (child.first == value) {
                return child.second;
            }
        }
        
        TrieNode* new_node = new TrieNode(current->node_level + 1);
        if (new_node->node_level <= context_length) {
            new_node->node_index = node_counter.fetch_add(1) + 1;
        }
        
        current->children.emplace_back(value, new_node);
        return new_node;
    }


    double get_log_value_insert(int64_t count) {
        if (count >= size_logcalc_memory) {
            return (count + 1) * std::log(count + 1) - count * std::log(count);
        }
        
        if (logcalc_memory_insert[count] == -1) {
            std::lock_guard<std::mutex> lock(logcalc_mutex);
            if (logcalc_memory_insert[count] == -1) {  // Double check after acquiring lock
                logcalc_memory_insert[count] = (count + 1) * std::log(count + 1) - 
                                             count * std::log(count);
            }
        }
        return logcalc_memory_insert[count];
    }

    double get_log_value_entropy(int64_t count) {
        if (count >= size_logcalc_memory) {
            return count * std::log(count);
        }
        
        if (logcalc_memory_entropy[count] == -1) {
            std::lock_guard<std::mutex> lock(logcalc_mutex);
            if (logcalc_memory_entropy[count] == -1) {  // Double check after acquiring lock
                logcalc_memory_entropy[count] = count * std::log(count);
            }
        }
        return logcalc_memory_entropy[count];
    }

public:
    
    RAMTrie(int64_t ctx_length, size_t initial_size = 1000000) 
        : context_length(ctx_length), 
          node_counter(0), 
          num_unique_contexts(0), 
          num_total_contexts(0),
          logcalc_memory_insert(size_logcalc_memory, -1),
          logcalc_memory_entropy(size_logcalc_memory, -1) {
        
        root = new TrieNode();
        root->node_index = node_counter.fetch_add(1);
        
        
        // Pre-allocate arrays
        countLog_array.resize(initial_size);
        ctxLen_array.resize(initial_size);
        ctxCount_array.resize(initial_size);
    }

    // Loading constructor
    RAMTrie(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Read metadata
        in.read(reinterpret_cast<char*>(&context_length), sizeof(context_length));
        
        int64_t node_count, unique_contexts, total_contexts;
        in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
        in.read(reinterpret_cast<char*>(&unique_contexts), sizeof(unique_contexts));
        in.read(reinterpret_cast<char*>(&total_contexts), sizeof(total_contexts));
        
        node_counter.store(node_count);
        num_unique_contexts.store(unique_contexts);
        num_total_contexts.store(total_contexts);
        
        // Read trie structure
        root = TrieNode::deserialize(in);  // Changed to use raw pointer
        root->node_index = 0;
        
        // Read arrays
        size_t array_size;
        in.read(reinterpret_cast<char*>(&array_size), sizeof(array_size));
        
        countLog_array.resize(array_size);
        ctxLen_array.resize(array_size);
        ctxCount_array.resize(array_size);
        
        in.read(reinterpret_cast<char*>(countLog_array.data()), array_size * sizeof(double));
        in.read(reinterpret_cast<char*>(ctxLen_array.data()), array_size * sizeof(int));
        in.read(reinterpret_cast<char*>(ctxCount_array.data()), array_size * sizeof(int));
        
        // Initialize log calculation vectors
        logcalc_memory_insert.resize(size_logcalc_memory, -1);
        logcalc_memory_entropy.resize(size_logcalc_memory, -1);
        
        in.read(reinterpret_cast<char*>(logcalc_memory_insert.data()), size_logcalc_memory * sizeof(double));
        in.read(reinterpret_cast<char*>(logcalc_memory_entropy.data()), size_logcalc_memory * sizeof(double));
    }

    // Save method
    void save(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Write metadata
        out.write(reinterpret_cast<const char*>(&context_length), sizeof(context_length));
        auto node_count = node_counter.load();
        out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
        auto unique_contexts = num_unique_contexts.load();
        out.write(reinterpret_cast<const char*>(&unique_contexts), sizeof(unique_contexts));
        auto total_contexts = num_total_contexts.load();
        out.write(reinterpret_cast<const char*>(&total_contexts), sizeof(total_contexts));
        
        // Write the trie structure
        root->serialize(out);
        
        // Write arrays
        size_t array_size = countLog_array.size();
        out.write(reinterpret_cast<const char*>(&array_size), sizeof(array_size));
        out.write(reinterpret_cast<const char*>(countLog_array.data()), array_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(ctxLen_array.data()), array_size * sizeof(int));
        out.write(reinterpret_cast<const char*>(ctxCount_array.data()), array_size * sizeof(int));
        
        // Write log calculation caches
        out.write(reinterpret_cast<const char*>(logcalc_memory_insert.data()), size_logcalc_memory * sizeof(double));
        out.write(reinterpret_cast<const char*>(logcalc_memory_entropy.data()), size_logcalc_memory * sizeof(double));
        
        out.close();
    }
    

    // ... rest of the methods ...

    // ~RAMTrie() {
    //     std::cout << "Starting cleanup" << std::endl;
        
    //     // if (root) {
    //         // std::cout << "Root node exists" << std::endl;
    //         // std::cout << "Root level: " << root->node_level << std::endl;
    //         // std::cout << "Root has " << root->children.size() << " children" << std::endl;
    //         // 
    //         // for (size_t i = 0; i < root->children.size(); i++) {
    //             // std::cout << "Deleting child " << i << std::endl;
    //             // delete root->children[i].second;
    //             // std::cout << "Child " << i << " deleted" << std::endl;
    //         // }
    //         // root->children.clear();
    //         // std::cout << "Cleared root's children" << std::endl;
    //         // 
    //         // delete root;
    //         // std::cout << "Deleted root" << std::endl;
    //     // }

    //     std::cout << "Starting cleanup" << std::endl;
    //     delete root;  // Let the node destructor handle the recursive cleanup
    //     std::cout << "Finished cleanup" << std::endl;

    //     // countLog_array.clear();
    //     // ctxLen_array.clear();
    //     // ctxCount_array.clear();
    //     // logcalc_memory_insert.clear();
    //     // logcalc_memory_entropy.clear();
        
    //     // std::cout << "Finished cleanup" << std::endl;
    // }


    // void cleanup() {
    //     // Explicitly handle memory cleanup
    //     delete root;
    //     root = nullptr;

    //     countLog_array.clear();
    //     ctxLen_array.clear();
    //     ctxCount_array.clear();
    //     logcalc_memory_insert.clear();
    //     logcalc_memory_entropy.clear();

    //     std::cout << "Finished cleanup" << std::endl;
    // }

    void cleanup() {
        if (!is_deleted.exchange(true)) {
            if (root) {
                std::cout << "Deleting root" << std::endl;
                delete root;
                root = nullptr;
            }

            countLog_array.clear();
            ctxLen_array.clear();
            ctxCount_array.clear();
            logcalc_memory_insert.clear();
            logcalc_memory_entropy.clear();
        }
    }

    void reset() {
        cleanup();
    }

    ~RAMTrie() {
        std::cerr << "C++ Destructor called for RAMTrie at " << this 
                << " with root " << root << std::endl;
        if (root) {
            std::cerr << "ME THINK ME GUON KILL MISELF !!!" << std::endl;
            delete root;
            root = nullptr;
        }
    }



    std::vector<std::unordered_map<int64_t, double>> insert_context(const torch::Tensor& tensor, int64_t column, bool return_prob_distr) {   
        std::vector<std::unordered_map<int64_t, double>> distributions;
        auto accessor = tensor.accessor<int64_t, 2>();
        int64_t current_level = 0;
        TrieNode* current = root;  // Changed to raw pointer

        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];
            
            if (j > 0) {
                num_total_contexts.fetch_add(1);
            }
            
            // Find child
            TrieNode* next_node = nullptr;
            for (auto& child : current->children) {
                if (child.first == value) {
                    next_node = child.second;
                    break;
                }
            }

            if (!next_node) {
                // Create new node
                next_node = new TrieNode(current_level + 1);  // Changed to new
                next_node->count = 0;
                total_memory_usage.fetch_add(sizeof(TrieNode));
                
                if (next_node->node_level <= context_length) {
                    next_node->node_index = node_counter.fetch_add(1);
                    num_total_contexts_per_level[next_node->node_level]++;
                }

                ctxLen_array[next_node->node_index] = next_node->node_level;
                DEBUG_PRINT("New node value " << value);
                DEBUG_PRINT("New node created with level " << ctxLen_array[next_node->node_index]);
                DEBUG_PRINT("New node index is  " << next_node->node_index);
                DEBUG_PRINT("__________________________________________________");
                
                current->children.emplace_back(value, next_node);
                total_memory_usage.fetch_add(sizeof(std::pair<int64_t, TrieNode*>));
            }

            // Update counts and entropy data
            int64_t c_t_temp = next_node->count;
            
            if (current->node_index > 0 && c_t_temp > 0 && current->node_level <= context_length) {
                double log_value;
                {
                    std::lock_guard<std::mutex> log_lock(logcalc_mutex);
                    if (logcalc_memory_insert[c_t_temp] == -1) {
                        logcalc_memory_insert[c_t_temp] = (c_t_temp + 1) * std::log(c_t_temp + 1) - 
                                                        (c_t_temp) * std::log(c_t_temp);
                    }
                    log_value = logcalc_memory_insert[c_t_temp];
                }
                countLog_array[current->node_index] += log_value;
            }
            
            ctxCount_array[current->node_index] += 1;
            next_node->count++;

            current = next_node;
            current_level++;

            if (return_prob_distr) {
                // Create distribution
                std::unordered_map<int64_t, double> current_distribution;
                for (const auto& child : current->children) {
                    current_distribution[child.first] = static_cast<double>(child.second->count);
                }
                distributions.push_back(current_distribution);
            }
        }

        current = nullptr;

        return distributions;
    }

    InsertResult insert(torch::Tensor tensor, bool return_prob_distr) {
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        int num_threads = 1;  // Currently single-threaded, can be expanded later
        omp_set_num_threads(num_threads);

        // Calculate batch size per thread
        int batch_size = tensor.size(0);
        int chunk_size = std::max(1, batch_size / num_threads);

        std::vector<std::vector<std::unordered_map<int64_t, double>>> soft_label_distributions(batch_size);
        
        // Dynamic scheduling with calculated chunk size
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int col = 0; col < batch_size; col++) {
            soft_label_distributions[col] = this->insert_context(tensor, col, return_prob_distr);
        }

        // Convert to Python list
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

        // End timing and calculate duration
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        return InsertResult{
            soft_label_list,
            static_cast<double>(duration.count())
        };
    }


    std::vector<std::unordered_map<int64_t, double>> retrieve_context_softlabel(const torch::Tensor& tensor, int64_t column) {
        std::vector<std::unordered_map<int64_t, double>> distributions;
        TrieNode* current = root;  // Changed to raw pointer

        auto accessor = tensor.accessor<int64_t, 2>();

        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];

            // Find child with matching value
            TrieNode* next_node = nullptr;
            for (const auto& child : current->children) {
                if (child.first == value) {
                    next_node = child.second;
                    break;
                }
            }
            
            if (next_node) {
                current = next_node;
                
                // Get raw counts for current node's children
                std::unordered_map<int64_t, double> distribution;
                for (const auto& child : current->children) {
                    distribution[child.first] = static_cast<double>(child.second->count);
                }
                distributions.push_back(distribution);
            } else {
                return distributions;  // Return what we have if we can't continue
            }
        }
        
        return distributions;
    }

    py::list retrieve_softlabel(const torch::Tensor& tensor) {
        // Input validation
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
        return total_memory_usage.load();
    }



    EntropyResult calculate_and_get_entropy_faster_root() {
        DEBUG_PRINT("________________________________________________________________");
        DEBUG_PRINT("calculate_and_get_entropy_faster_root");

        double total_entropy = 0;
        double entropy_temp = 0;

        // Reset all counters
        for (auto& pair : num_unique_contexts_per_level) {
            pair.second = 0;
        }
        for (auto& pair : entropy_per_level) {
            pair.second = 0;
        }
        for (auto& pair : count_per_level) {
            pair.second = 0;
        }
        for (auto& pair : num_oneHots_per_level) {
            pair.second = 0;
        }

        int64_t number_of_oneHots = 0;
        int64_t total_counter = 0;
        int counter = 0;
        
        DEBUG_PRINT(node_counter);
        DEBUG_PRINT("Printing entropy: ");
        
        #pragma omp parallel for reduction(+:total_entropy,total_counter,number_of_oneHots,count_per_level[:context_length+1],num_unique_contexts_per_level[:context_length+1],entropy_per_level[:context_length+1],num_oneHots_per_level[:context_length+1])
        for(int j = 1; j <= node_counter; j++) {
            entropy_temp = countLog_array[j] - ctxCount_array[j] * log(ctxCount_array[j]);

            DEBUG_PRINT("_________________________________________");
            DEBUG_PRINT("entropy_temp: " << entropy_temp);
            DEBUG_PRINT("Count: " << ctxCount_array[j]);
            

            if (ctxCount_array[j] == 0) {
                counter += 1;
                if (counter == 100) {
                    DEBUG_PRINT("Quitting since the counter is 0.");
                    return EntropyResult(0.0, 0, 0);
                }
            }

            if (entropy_temp == 0.0) {
                number_of_oneHots += ctxCount_array[j];
                num_oneHots_per_level[ctxLen_array[j]] += ctxCount_array[j];
            }

            num_unique_contexts_per_level[ctxLen_array[j]] += 1;
            entropy_per_level[ctxLen_array[j]] += entropy_temp;
            
            total_entropy += entropy_temp;
            total_counter += ctxCount_array[j];
            count_per_level[ctxLen_array[j]] += ctxCount_array[j];
        }

        DEBUG_PRINT("total_entropy: " << total_entropy);
        DEBUG_PRINT("total_counter: " << total_counter);

        total_entropy = -total_entropy / total_counter;
        for(int t = 0; t <= context_length; t++) {
            if (count_per_level[t] > 0) {  // Avoid division by zero
                entropy_per_level[t] /= -count_per_level[t];
            }
        }

        double perc_of_oneHots = static_cast<double>(number_of_oneHots) / total_counter;
        DEBUG_PRINT("Percentage of OneHots is: " << perc_of_oneHots);
        DEBUG_PRINT("number_of_oneHots: " << number_of_oneHots);
        DEBUG_PRINT("total_counter: " << total_counter);

        num_unique_contexts = node_counter.load();

        return EntropyResult(total_entropy, total_counter, number_of_oneHots);
    }

    EntropyResult calculate_and_get_entropy_faster_branch() {
        DEBUG_PRINT("________________________________________________________________");
        DEBUG_PRINT("calculate_and_get_entropy_faster_branch");

        double total_entropy = 0;
        double entropy_temp = 0;

        // Reset all counters
        for (auto& pair : num_unique_contexts_per_level) {
            pair.second = 0;
        }
        for (auto& pair : entropy_per_level) {
            pair.second = 0;
        }
        for (auto& pair : count_per_level) {
            pair.second = 0;
        }
        for (auto& pair : num_oneHots_per_level) {
            pair.second = 0;
        }

        int64_t number_of_oneHots = 0;
        int64_t total_counter = 0;
        int counter = 0;
        
        DEBUG_PRINT(node_counter);
        DEBUG_PRINT("Printing entropy: ");
        
        #pragma omp parallel for reduction(+:total_entropy,total_counter,number_of_oneHots,count_per_level[:context_length+1],num_unique_contexts_per_level[:context_length+1],entropy_per_level[:context_length+1],num_oneHots_per_level[:context_length+1])
        for(int j = 1; j <= node_counter; j++) {

            DEBUG_PRINT("_________________________________________");
            DEBUG_PRINT("entropy_temp: " << entropy_temp);
            DEBUG_PRINT("Count: " << ctxCount_array[j]);
            DEBUG_PRINT("Level: " << ctxLen_array[j]);
            if (ctxLen_array[j] >= 3) {  // Only difference from root version
                entropy_temp = countLog_array[j] - ctxCount_array[j] * log(ctxCount_array[j]);

                if (ctxCount_array[j] == 0) {
                    counter += 1;
                    if (counter == 100) {
                        DEBUG_PRINT("Quitting since the counter is 0.");
                        return EntropyResult(0.0, 0, 0);
                    }
                }

                if (entropy_temp == 0.0) {
                    number_of_oneHots += ctxCount_array[j];
                    num_oneHots_per_level[ctxLen_array[j]] += ctxCount_array[j];
                }

                num_unique_contexts_per_level[ctxLen_array[j]] += 1;
                entropy_per_level[ctxLen_array[j]] += entropy_temp;
                
                total_entropy += entropy_temp;
                total_counter += ctxCount_array[j];
                count_per_level[ctxLen_array[j]] += ctxCount_array[j];
            }
        }

        DEBUG_PRINT("total_entropy: " << total_entropy);
        DEBUG_PRINT("total_counter: " << total_counter);

        total_entropy = -total_entropy / total_counter;
        for(int t = 0; t <= context_length; t++) {
            if (count_per_level[t] > 0) {  // Avoid division by zero
                entropy_per_level[t] /= -count_per_level[t];
            }
        }

        double perc_of_oneHots = static_cast<double>(number_of_oneHots) / total_counter;
        DEBUG_PRINT("Percentage of OneHots is: " << perc_of_oneHots);
        DEBUG_PRINT("number_of_oneHots: " << number_of_oneHots);
        DEBUG_PRINT("total_counter: " << total_counter);

        num_unique_contexts = node_counter.load();

        return EntropyResult(total_entropy, total_counter, number_of_oneHots);
    }


    int64_t get_num_unique_contexts() const {
        return num_unique_contexts.load();
    }

    int64_t get_num_total_contexts() const {
        return num_total_contexts.load();
    }


    // Get number of unique contexts per level
    std::map<int64_t, int> get_num_unique_contexts_per_level() const {
        return num_unique_contexts_per_level;
    }

    // Get total contexts per level
    std::map<int64_t, int> get_num_total_contexts_per_level() const {
        return num_total_contexts_per_level;
    }

    // Get entropy values per level
    std::map<int64_t, double> get_entropy_per_level() const {
        return entropy_per_level;
    }

    // Get one-hot counts per level
    std::map<int64_t, int> get_oneHots_per_level() const {
        return num_oneHots_per_level;
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
    py::class_<RAMTrie>(m, "RAMTrie")
        .def(py::init<int64_t, size_t>(), py::return_value_policy::reference)
        .def(py::init<const std::string&>(), py::return_value_policy::reference)
        .def("save", &RAMTrie::save)
        .def("insert", &RAMTrie::insert)
        .def("cleanup", &RAMTrie::cleanup)
        .def("reset", &RAMTrie::reset)
        .def("retrieve_softlabel", &RAMTrie::retrieve_softlabel)
        .def("get_memory_usage", &RAMTrie::get_memory_usage)
        .def("get_num_unique_contexts", &RAMTrie::get_num_unique_contexts)
        .def("get_num_total_contexts", &RAMTrie::get_num_total_contexts)
        .def("get_num_unique_contexts_per_level", &RAMTrie::get_num_unique_contexts_per_level)
        .def("get_num_total_contexts_per_level", &RAMTrie::get_num_total_contexts_per_level)
        .def("get_entropy_per_level", &RAMTrie::get_entropy_per_level)
        .def("get_oneHots_per_level", &RAMTrie::get_oneHots_per_level)
        .def("calculate_and_get_entropy_faster_branch", &RAMTrie::calculate_and_get_entropy_faster_branch)
        .def("calculate_and_get_entropy_faster_root", &RAMTrie::calculate_and_get_entropy_faster_root);
    
    
    py::class_<EntropyResult>(m, "EntropyResult")
        .def_readonly("entropy", &EntropyResult::entropy)
        .def_readonly("total_count", &EntropyResult::total_count)
        .def_readonly("number_of_oneHots", &EntropyResult::number_of_oneHots);

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("result", &InsertResult::result)
        .def_readonly("execution_time_ms", &InsertResult::execution_time_ms);
        

    py::class_<MemMapArray<int64_t>>(m, "MemMapArrayInt")
        .def(py::init<const std::string&, size_t>())
        .def("get_size", &MemMapArray<int64_t>::getSize)
        .def("__getitem__", [](MemMapArray<int64_t> &self, size_t index) { return self[index]; });
    
    py::class_<MemMapArray<double>>(m, "MemMapArrayDouble")
        .def(py::init<const std::string&, size_t>())
        .def("get_size", &MemMapArray<double>::getSize)
        .def("__getitem__", [](MemMapArray<double> &self, size_t index) { return self[index]; });

}