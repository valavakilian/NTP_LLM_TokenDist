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



struct TrieNode {
    int64_t count;
    int64_t num_children;
    int64_t children_offset;  // Offset to children in the memory-mapped file
    int64_t node_level;
};


void printTrieNode(const TrieNode* node) {
    std::cout << "TrieNode Information:" << std::endl;
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
    int fd;
    char* mapped_memory;
    std::atomic<size_t> file_size;
    size_t allocated_size;
    std::string filename;
    std::string metadata_filename;

    std::atomic<int64_t> num_unique_contexts;  // New global parameter for unqie context count
    std::atomic<int64_t> num_total_contexts;  // New global parameter for total context count

    int64_t context_length;
    std::atomic<int64_t> node_counter;
    std::atomic<int64_t> node_mutex_counter;
    
    const size_t array_size = 1000000000; // Size of the array
                                       
    const size_t size_logcalc_memory = 1000000000;  // 1 billion integers (~4 GB)

    std::mutex alloc_memory_mutex;
    std::mutex alloc_node_mutex;

    std::vector<std::mutex> mutex_array_lock;  // Array of mutexes


    std::atomic<int> active_nodes{0};
    std::atomic<int> max_active_nodes{0};  // New variable to track maximum

    const size_t MEMORY_THRESHOLD = 1ULL * 1024 * 1024 * 1024; // 1GB threshold

    bool is_close_to_memory_limit() const {
        return (allocated_size - file_size) <= MEMORY_THRESHOLD;
    }

    TrieNode* get_node(size_t offset) {
        if (offset >= file_size) {
            throw std::runtime_error("Attempted to access memory outside of mapped region");
        }
        return reinterpret_cast<TrieNode*>(mapped_memory + offset);
    }

    std::pair<int64_t, int64_t>* get_children(TrieNode* node) {
        if (node->children_offset >= file_size) {
            throw std::runtime_error("Invalid children offset");
        }
        return reinterpret_cast<std::pair<int64_t, int64_t>*>(mapped_memory + node->children_offset);
    }


    // Modify the allocate_space function to periodically update critical info
    size_t allocate_space(size_t size) {
        return file_size.fetch_add(size);
    }

    
    size_t allocate_node(int64_t parent_level) {
        size_t offset = allocate_space(sizeof(TrieNode));
        TrieNode* new_node = get_node(offset);
        new_node->node_level = parent_level + 1; 
        return offset;
    }

    size_t allocate_children(int64_t num_children) {
        return allocate_space(num_children * sizeof(std::pair<int64_t, int64_t>));
    }

    int64_t find_child(std::pair<int64_t, int64_t>* children, int64_t num_children, int64_t value) {
        int64_t left = 0, right = num_children - 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            if (children[mid].first == value) {
                return mid;
            }
            if (children[mid].first < value) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    void insert_child(TrieNode* node, int64_t value, int64_t child_offset) {
        std::pair<int64_t, int64_t>* children = get_children(node);
        int64_t insert_pos = 0;
        while (insert_pos < node->num_children && children[insert_pos].first < value) {
            insert_pos++;
        }

        if (insert_pos < node->num_children) {
            memmove(&children[insert_pos + 1], &children[insert_pos], 
                    (node->num_children - insert_pos) * sizeof(std::pair<int64_t, int64_t>));
        }

        children[insert_pos] = {value, child_offset};
        node->num_children++;
    }

    size_t find_node(const std::vector<int64_t>& sequence) {
        size_t current_offset = 0;  // Start from the root
        TrieNode* current = get_node(current_offset);

        for (int64_t value : sequence) {
            bool found = false;
            std::pair<int64_t, int64_t>* children = get_children(current);
            for (int64_t i = 0; i < current->num_children; ++i) {
                if (children[i].first == value) {
                    current_offset = children[i].second;
                    current = get_node(current_offset);
                    found = true;
                    break;
                }
            }
            if (!found) {
                return -1;  // Sequence not found in trie
            }
        }
        return current_offset;
    }


public:
    Trie_module_protV1(const std::string& fname, size_t initial_size_gb, int64_t context_length) 
    : filename(fname + ".bin"), context_length(context_length), mutex_array_lock(0) {
        allocated_size = initial_size_gb * 1024ULL * 1024ULL * 1024ULL; // Convert GB to bytes
        fd = open(filename.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for memory mapping");
        }

        // Set the file size to the allocated size
        if (ftruncate(fd, allocated_size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to set initial file size");
        }

        mapped_memory = static_cast<char*>(mmap(NULL, allocated_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file to memory");
        }

        metadata_filename = filename + "_metadata.bin";


        // Initialize the root node
        file_size = sizeof(TrieNode);
        TrieNode* root = get_node(0);
        root->count = 0;
        root->num_children = 0;
        root->children_offset = 0;

        num_unique_contexts = 0;  // Count the root node
        num_total_contexts = 0;
        node_counter = 0;
        node_mutex_counter = 0;


        int num_procs = omp_get_num_procs();
        DEBUG_PRINT("Number of processors available: " << num_procs);

        DEBUG_PRINT("Trie initialized with allocated size: " << allocated_size << " bytes");
    }

    // Constructor to load an existing Trie from a file
    // Constructor to load an existing Trie from a file
    Trie_module_protV1(const std::string& fname) : 
        filename(fname + ".bin"), 
        mutex_array_lock(0) {
        
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

        // Step 4: Load metadata and initialize all required arrays
        try {
            load_metadata(metadata_filename);
            
            // Load the arrays from their respective files
            std::ifstream per_level_data(fname + "_level_data.bin", std::ios::binary);
            if (per_level_data.is_open()) {
                size_t num_levels;
                per_level_data.read(reinterpret_cast<char*>(&num_levels), sizeof(size_t));
                
                for (size_t i = 0; i < num_levels; i++) {
                    int64_t level;
                    per_level_data.read(reinterpret_cast<char*>(&level), sizeof(int64_t));
                }
                per_level_data.close();
            }

            // Note: countLog_array, ctxLen_array, and ctxCount_array are already loaded

            std::cout << "Successfully loaded Trie with following statistics:" << std::endl;
            std::cout << "Number of unique contexts: " << num_unique_contexts.load() << std::endl;
            std::cout << "Number of total contexts: " << num_total_contexts.load() << std::endl;
            std::cout << "Context length: " << context_length << std::endl;
            std::cout << "Node counter: " << node_counter.load() << std::endl;
            
        } catch (const std::exception& e) {
            if (mapped_memory != MAP_FAILED) {
                munmap(mapped_memory, allocated_size);
            }
            close(fd);
            throw std::runtime_error("Error loading metadata: " + std::string(e.what()));
        }
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
        file << node_counter << "\n";
        file << node_mutex_counter << "\n";

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

            int64_t temp_counter; 
            
            file >> temp_counter;
            node_counter.store(temp_counter);

            file >> temp_counter;
            node_mutex_counter.store(temp_counter);

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
        if (mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, allocated_size);
        }
        if (fd != -1) {
            // Truncate the file to the actually used size before closing
            ftruncate(fd, file_size);
            close(fd);
        }
        std::cout << "Destroyed the trie."<< std::endl;
    }

    std::vector<std::unordered_map<int64_t, double>> insert_context(const torch::Tensor& tensor, int64_t column, bool return_prob_distr) {   
        std::vector<std::unordered_map<int64_t, double>> distributions;
        auto accessor = tensor.accessor<int64_t, 2>();
        int64_t current_level = 0;
        size_t current_offset = 0;

        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];
            
            TrieNode* current = get_node(current_offset);
            
            // Lock the current node

            if (j > 0) {
                num_total_contexts.fetch_add(1);
            }
            
            int64_t child_index = -1;
            if (current->num_children > 0) {
                std::pair<int64_t, int64_t>* children = get_children(current);
                child_index = find_child(children, current->num_children, value);
            }

            if (child_index == -1) {
                // Create new node (this is already thread-safe due to atomic operations)
                size_t new_node_offset = allocate_node(current->node_level);
                TrieNode* new_node = get_node(new_node_offset);
                
                // Initialize new node (protected by the current node's lock)
                new_node->count = 0;
                new_node->num_children = 0;
                new_node->children_offset = 0;
                new_node->node_level = current_level + 1;

                // if (new_node->node_level <= context_length) {
                //     num_total_contexts_per_level[new_node->node_level]++;
                // }

                // Allocate children (protected by current node's lock)
                if (current->num_children == 0) {
                    current->children_offset = allocate_children(1);
                } else {
                    size_t new_children_offset = allocate_children(current->num_children + 1);
                    std::memcpy(mapped_memory + new_children_offset, get_children(current), 
                            current->num_children * sizeof(std::pair<int64_t, int64_t>));
                    current->children_offset = new_children_offset;
                }

                insert_child(current, value, new_node_offset);
                current_offset = new_node_offset;
            } else {
                current_offset = get_children(current)[child_index].second;
            }

            // Update counts and entropy data
            TrieNode* next_node = get_node(current_offset);
            int64_t c_t_temp = next_node->count;
            
            next_node->count++;

            current_level++;

            if (return_prob_distr) {
                // Create distribution (protected by current node's lock)
                std::unordered_map<int64_t, double> current_distribution;
                std::pair<int64_t, int64_t>* children = get_children(current);
                for (int64_t i = 0; i < current->num_children; i++) {
                    TrieNode* child = get_node(children[i].second);
                    current_distribution[children[i].first] = static_cast<double>(child->count);
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
        if (is_close_to_memory_limit() || node_mutex_counter >= array_size - 5000 || node_counter >= array_size - 5000) {
            save_metadata();
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
        size_t current_offset = 0;
        TrieNode* current = get_node(current_offset);

        auto accessor = tensor.accessor<int64_t, 2>();

        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];

            int64_t child_index = -1;
            if (current->num_children > 0) {
                std::pair<int64_t, int64_t>* children = get_children(current);
                child_index = find_child(children, current->num_children, value);
                
                if (child_index != -1) {
                    current_offset = children[child_index].second;
                    current = get_node(current_offset);
                    
                    // Get raw counts for current node's children
                    std::unordered_map<int64_t, double> distribution;
                    if (current->num_children > 0) {
                        std::pair<int64_t, int64_t>* next_children = get_children(current);
                        for (int64_t j = 0; j < current->num_children; j++) {
                            TrieNode* child = get_node(next_children[j].second);
                            distribution[next_children[j].first] = static_cast<double>(child->count);
                        }
                    }
                    distributions.push_back(distribution);
                } else {
                    return distributions;
                }
            } else {
                return distributions;
            }
        }
        
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
        return file_size;
    }

    size_t get_allocated_size() const {
        return allocated_size;
    }


    int64_t get_num_unique_contexts() const {
        return num_unique_contexts.load();
    }

    int64_t get_num_total_contexts() const {
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
        .def(py::init<const std::string&, size_t, int64_t>())
        .def(py::init<const std::string&>())
        .def("insert", &Trie_module_protV1::insert)
        .def("retrieve_softlabel", &Trie_module_protV1::retrieve_softlabel)
        .def("get_memory_usage", &Trie_module_protV1::get_memory_usage)
        .def("get_allocated_size", &Trie_module_protV1::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_module_protV1::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_module_protV1::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("load_metadata", &Trie_module_protV1::load_metadata)
        .def("save_metadata", &Trie_module_protV1::save_metadata);

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("result", &InsertResult::result)
        .def_readonly("execution_time_ms", &InsertResult::execution_time_ms);

}