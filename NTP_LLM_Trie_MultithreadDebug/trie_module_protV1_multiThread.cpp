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
    int64_t num_children;
    int64_t children_offset;  // Offset to children in the memory-mapped file
    bool is_root;  // New field to indicate if this is the root node
    int64_t node_level;
    int64_t node_index;
    int64_t node_mutex_index;
};


void printTrieNode(const TrieNode* node) {
    std::cout << "TrieNode Information:" << std::endl;
    std::cout << "Count: " << node->count << std::endl;
    std::cout << "Number of Children: " << node->num_children << std::endl;
    std::cout << "Children Offset: " << node->children_offset << std::endl;
    std::cout << "Is Root: " << (node->is_root ? "True" : "False") << std::endl;
    std::cout << "Node Level: " << node->node_level << std::endl;
    std::cout << "Node Index: " << node->node_index << std::endl;
    std::cout << "Node Mutex Index: " << node->node_mutex_index << std::endl;
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
    std::string countLog_array_filename;
    std::string ctxLen_array_filename;
    std::atomic<int64_t> num_unique_contexts;  // New global parameter for unqie context count
    std::atomic<int64_t> num_total_contexts;  // New global parameter for total context count

    int64_t context_length;
    std::atomic<int64_t> node_counter;
    std::atomic<int64_t> node_mutex_counter;
    std::map<int64_t, int> num_unique_contexts_per_level;  // int64_t for the level, int for the count
    std::map<int64_t, int> num_total_contexts_per_level;

    std::map<int64_t, double> entropy_per_level;
    std::map<int64_t, double> count_per_level;
                               
    const size_t array_size = 20000000; // Size of the array
    // std::vector<double> countLog_array;
    // std::vector<int> ctxLen_array;
    // std::vector<int64_t> ctxCount_array;
    MemMapArray<double> countLog_array;
    MemMapArray<int> ctxLen_array;
    MemMapArray<int> ctxCount_array;
                                       
    const size_t size_logcalc_memory = 20000000;  // 1 billion integers (~4 GB)
    std::vector<double> logcalc_memory_insert;
    std::vector<double> logcalc_memory_entropy;

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
        
        // size_t offset = file_size.fetch_add(size);
        // file_size += size;
        // if (file_size > allocated_size) {
        //     throw std::runtime_error("Exceeded pre-allocated file size");
        // }
        // DEBUG_PRINT("+++++++++++++++++++++++++++++++++++++++++++++++++++");
        // DEBUG_PRINT("allocate_space ");
        // DEBUG_PRINT("file_size " << file_size);
        // DEBUG_PRINT("size " << size);
        return file_size.fetch_add(size);
    }

    
    size_t allocate_node(int64_t parent_level) {
        // DEBUG_PRINT("Inside allocate_node ...");
        // DEBUG_PRINT("+++++++++++++++++++++++++++++++++++++++++++++++++++");
        // DEBUG_PRINT("allocate_node ");
        // DEBUG_PRINT("Locking alloc_memory_mutex");
        // alloc_memory_mutex.lock();
        // DEBUG_PRINT("file_size before" << file_size);
        size_t offset = allocate_space(sizeof(TrieNode));
        // DEBUG_PRINT("file_size after" << file_size);
        // DEBUG_PRINT("offset before " << offset);
        // alloc_memory_mutex.unlock();
        // DEBUG_PRINT("Unlocking alloc_memory_mutex");
        TrieNode* new_node = get_node(offset);
        new_node->node_level = parent_level + 1; 

        // DEBUG_PRINT("Locking alloc_node_mutex");
        
        if (new_node->node_level <= context_length){
            // alloc_node_mutex.lock();
            // node_counter += 1;
            // node_mutex_counter += 1;
            // DEBUG_PRINT("After allocate_node node_counter is " << node_counter.load());
            new_node->node_index = node_counter.fetch_add(1) + 1;
            new_node->node_mutex_index = node_mutex_counter.fetch_add(1) + 1;
            // alloc_node_mutex.unlock();
            ctxLen_array[new_node->node_index] = new_node->node_level;
        } else {
            // alloc_node_mutex.lock();
            // node_mutex_counter += 1;
            new_node->node_mutex_index = node_mutex_counter.fetch_add(1) + 1;
            new_node->node_index = -1;
            // alloc_node_mutex.unlock();
        }
        
        // DEBUG_PRINT("offset after " << offset);
        // DEBUG_PRINT("Unlocking alloc_node_mutex");
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

    void collect_sequences(size_t node_offset, std::vector<int64_t>& current_sequence, 
                           std::unordered_map<std::vector<int64_t>, int64_t>& sequences) {
        TrieNode* node = get_node(node_offset);
        
        if (node->num_children == 0) {
            return ; 
        }
        
        if (node->count > 0) {
            sequences[current_sequence] = node->count;
        }
        
        std::pair<int64_t, int64_t>* children = get_children(node);
        for (int64_t i = 0; i < node->num_children; ++i) {
            current_sequence.push_back(children[i].first);
            collect_sequences(children[i].second, current_sequence, sequences);
            current_sequence.pop_back();
        }
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
    : filename(fname + ".bin"), context_length(context_length), countLog_array(fname + "_countLog_arr.dat", array_size), 
    ctxLen_array(fname + "_ctxLen_arr.dat", array_size), ctxCount_array(fname + "_ctxCount_arr.dat", array_size)
    , logcalc_memory_insert(size_logcalc_memory, -1), logcalc_memory_entropy(size_logcalc_memory, -1), mutex_array_lock(array_size) {
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
        root->is_root = true;  // Set the root node indicator
        root->node_index = 0;
        root->node_mutex_index = 0;

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
        countLog_array(fname + "_countLog_arr.dat"), 
        ctxLen_array(fname + "_ctxLen_arr.dat"), 
        ctxCount_array(fname + "_ctxCount_arr.dat"),
        logcalc_memory_insert(size_logcalc_memory, -1), 
        logcalc_memory_entropy(size_logcalc_memory, -1), 
        mutex_array_lock(array_size) {
        
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
            
            // Initialize arrays with loaded data
            num_unique_contexts_per_level.clear();
            num_total_contexts_per_level.clear();
            entropy_per_level.clear();
            count_per_level.clear();

            // Load the arrays from their respective files
            std::ifstream per_level_data(fname + "_level_data.bin", std::ios::binary);
            if (per_level_data.is_open()) {
                size_t num_levels;
                per_level_data.read(reinterpret_cast<char*>(&num_levels), sizeof(size_t));
                
                for (size_t i = 0; i < num_levels; i++) {
                    int64_t level;
                    int unique_count, total_count;
                    double entropy, count;
                    
                    per_level_data.read(reinterpret_cast<char*>(&level), sizeof(int64_t));
                    per_level_data.read(reinterpret_cast<char*>(&unique_count), sizeof(int));
                    per_level_data.read(reinterpret_cast<char*>(&total_count), sizeof(int));
                    per_level_data.read(reinterpret_cast<char*>(&entropy), sizeof(double));
                    per_level_data.read(reinterpret_cast<char*>(&count), sizeof(double));
                    
                    num_unique_contexts_per_level[level] = unique_count;
                    num_total_contexts_per_level[level] = total_count;
                    entropy_per_level[level] = entropy;
                    count_per_level[level] = count;
                }
                per_level_data.close();
            }

            // Note: countLog_array, ctxLen_array, and ctxCount_array are already loaded
            // through the MemMapArray constructors in the initialization list

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

        // Save the level-specific data
        for (const auto& [level, count] : num_unique_contexts_per_level) {
            file << "level " << level << " " << count << "\n";
        }
        file.close();

        // Save additional array data in binary format
        std::ofstream per_level_data(filename.substr(0, filename.length() - 4) + "_level_data.bin", 
                                    std::ios::binary);
        if (!per_level_data.is_open()) {
            throw std::runtime_error("Failed to open level data file for writing");
        }

        // Write number of levels
        size_t num_levels = num_unique_contexts_per_level.size();
        per_level_data.write(reinterpret_cast<const char*>(&num_levels), sizeof(size_t));

        // Write all level-specific data
        for (const auto& [level, _] : num_unique_contexts_per_level) {
            per_level_data.write(reinterpret_cast<const char*>(&level), sizeof(int64_t));
            int unique_count = num_unique_contexts_per_level[level];
            int total_count = num_total_contexts_per_level[level];
            double entropy = entropy_per_level[level];
            double count = count_per_level[level];
            
            per_level_data.write(reinterpret_cast<const char*>(&unique_count), sizeof(int));
            per_level_data.write(reinterpret_cast<const char*>(&total_count), sizeof(int));
            per_level_data.write(reinterpret_cast<const char*>(&entropy), sizeof(double));
            per_level_data.write(reinterpret_cast<const char*>(&count), sizeof(double));
        }
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
            while (file >> temp >> level >> count) {
                num_unique_contexts_per_level[level] = count;
            }
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
            std::lock_guard<std::mutex> lock(mutex_array_lock[current->node_mutex_index]);

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

                if (new_node->node_level <= context_length) {
                    num_total_contexts_per_level[new_node->node_level]++;
                }

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
            
            if (current->node_index > 0 && c_t_temp > 0 && current->node_level <= context_length) {
                double log_value;
                {
                    std::lock_guard<std::mutex> log_lock(alloc_memory_mutex);
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
        int num_threads = num_procs;  // Using at most 8 threads
        omp_set_num_threads(num_threads);


        // Calculate batch size per thread
        int batch_size = tensor.size(0);
        int chunk_size = std::max(1, batch_size / num_threads);

        // DEBUG_PRINT("Running with " << num_threads << " threads");
        // DEBUG_PRINT("Total batch size: " << batch_size);
        // DEBUG_PRINT("Chunk size per thread: " << chunk_size);

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

    std::unordered_map<std::vector<int64_t>, int64_t> collect_all_sequences() {
        std::unordered_map<std::vector<int64_t>, int64_t> sequences;
        std::vector<int64_t> current_sequence;
        collect_sequences(0, current_sequence, sequences);
        return sequences;
    }

    size_t get_memory_usage() const {
        return file_size;
    }

    size_t get_allocated_size() const {
        return allocated_size;
    }


    double calculate_and_get_entropy_faster() {
        DEBUG_PRINT("________________________________________________________________");
        DEBUG_PRINT("calculate_and_get_entropy_faster");

        // int64_t total_N = 0;
        double total_entropy = 0;
        double entropy_temp = 0;

        for (auto& pair : num_unique_contexts_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }
        for (auto& pair : entropy_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }

        for (auto& pair : count_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }

        int64_t total_counter = 0;
        int counter = 0;
        DEBUG_PRINT(node_counter);
        DEBUG_PRINT("Printing entropy: ");
        for(int j = 1; j <= node_counter; j++){
            entropy_temp = countLog_array[j] - ctxCount_array[j] * log(ctxCount_array[j]);

            // DEBUG_PRINT(ctxCount_array[j]);
            if (ctxCount_array[j] == 0){
                counter += 1;
                if (counter == 100){
                    DEBUG_PRINT("Quitting since the counter is 0.");
                    return 0.0;
                }
            }
            num_unique_contexts_per_level[ctxLen_array[j]] += 1;
            entropy_per_level[ctxLen_array[j]] += entropy_temp;
            
            total_entropy += entropy_temp;

            total_counter += ctxCount_array[j];
            count_per_level[ctxLen_array[j]] += ctxCount_array[j];

            // std::cout << entropy_temp << ", ";
        }  
        // std::cout << "\n";

        // for(int j = 1; j < node_counter; j++){
        //     entropy_temp = countLog_array[j] - ctxCount_array[j] * log(ctxCount_array[j]);
        //     entropy_temp /= total_counter;
        //     DEBUG_PRINT("__________________________________________________________");
        //     DEBUG_PRINT("Counter: " << j );
        //     DEBUG_PRINT("entropy_temp: " << entropy_temp );
        //     DEBUG_PRINT("Count: " << ctxCount_array[j]);
        // }  

        DEBUG_PRINT("total_entropy: " << total_entropy);
        DEBUG_PRINT("total_counter: " << total_counter);

        total_entropy = -total_entropy / total_counter;
        for(int t = 0; t < context_length; t++){
            entropy_per_level[t] /= -count_per_level[t];
        }

        num_unique_contexts = node_counter.load();
        
        return total_entropy;  // Normalize by total number of nodes
    }


    int64_t get_num_unique_contexts() const {
        return num_unique_contexts.load();
    }

    int64_t get_num_total_contexts() const {
        return num_total_contexts.load();
    }

    std::unordered_map<int64_t, double> get_children_distribution(const std::vector<int64_t>& sequence) {
        std::unordered_map<int64_t, double> distribution;

        size_t node_offset = find_node(sequence);
        if (node_offset == -1) {
            return distribution;  // Return empty map if sequence not found
        }

        TrieNode* node = get_node(node_offset);
        std::pair<int64_t, int64_t>* children = get_children(node);

        int64_t total_children_count = 0;
        for (int64_t i = 0; i < node->num_children; ++i) {
            TrieNode* child = get_node(children[i].second);
            total_children_count += child->count;
        }

        for (int64_t i = 0; i < node->num_children; ++i) {
            int64_t child_value = children[i].first;
            TrieNode* child = get_node(children[i].second);
            double probability = static_cast<double>(child->count) / total_children_count;
            distribution[child_value] = probability;
        }

        return distribution;
    }

    int64_t get_node_count(const std::vector<int64_t>& sequence) {
        size_t node_offset = find_node(sequence);
        if (node_offset == -1) {
            return 0;  // Return 0 if sequence not found
        }

        TrieNode* node = get_node(node_offset);
        return node->count;
    }

    int64_t get_node_level(const std::vector<int64_t>& sequence) {
        size_t node_offset = find_node(sequence);
        if (node_offset == -1) {
            return -1;  // Return -1 if sequence not found
        }

        TrieNode* node = get_node(node_offset);
        return node->node_level;
    }


    // New method to get the number of nodes at each level
    std::map<int64_t, int> get_num_unique_contexts_per_level() const {
        return num_unique_contexts_per_level;
    }

    // New method to get the number of nodes at each level
    std::map<int64_t, int> get_num_total_contexts_per_level() const {
        return num_total_contexts_per_level;
    }

    // New method to get the number of nodes at each level
    std::map<int64_t, double> get_entropy_per_level() const {
        return entropy_per_level;
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
        .def("collect_all_sequences", [](Trie_module_protV1& trie) {
            auto sequences = trie.collect_all_sequences();
            return convert_to_python_dict(sequences);
        })
        .def("get_memory_usage", &Trie_module_protV1::get_memory_usage)
        .def("get_allocated_size", &Trie_module_protV1::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_module_protV1::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_module_protV1::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("get_children_distribution", &Trie_module_protV1::get_children_distribution)
        .def("get_node_count", &Trie_module_protV1::get_node_count)
        .def("get_node_level", &Trie_module_protV1::get_node_level)
        .def("get_num_unique_contexts_per_level", &Trie_module_protV1::get_num_unique_contexts_per_level)
        .def("get_num_total_contexts_per_level", &Trie_module_protV1::get_num_total_contexts_per_level)
        .def("get_entropy_per_level", &Trie_module_protV1::get_entropy_per_level)
        .def("calculate_and_get_entropy_faster", &Trie_module_protV1::calculate_and_get_entropy_faster)
        .def("load_metadata", &Trie_module_protV1::load_metadata)
        .def("save_metadata", &Trie_module_protV1::save_metadata);
    
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