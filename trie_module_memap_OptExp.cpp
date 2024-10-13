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
    std::mutex node_mutex;  // Mutex for this node
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


class Trie_memap_sorted_OptExp {
private:
    int fd;
    char* mapped_memory;
    size_t file_size;
    size_t allocated_size;
    std::string filename;
    std::string metadata_filename;
    std::string countLog_array_filename;
    std::string ctxLen_array_filename;
    std::atomic<int64_t> num_unique_contexts;  // New global parameter for unqie context count
    std::atomic<int64_t> num_total_contexts;  // New global parameter for total context count

    int64_t context_length;
    int64_t node_counter;
    int64_t node_mutex_counter;
    std::map<int64_t, int> num_unique_contexts_per_level;  // int64_t for the level, int for the count
    std::map<int64_t, int> num_total_contexts_per_level;

    std::map<int64_t, double> entropy_per_level;
                               
    const size_t array_size = 1000000000; // Size of the array
    // std::vector<double> countLog_array;
    // std::vector<int> ctxLen_array;
    // std::vector<int64_t> ctxCount_array;
    MemMapArray<double> countLog_array;
    MemMapArray<int> ctxLen_array;
    MemMapArray<int> ctxCount_array;

    const size_t size_logcalc_memory = 1000000000;  // 1 billion integers (~4 GB)
    std::vector<double> logcalc_memory;

    std::mutex alloc_memory_mutex;
    std::mutex alloc_node_mutex;

    std::vector<std::mutex> mutex_array_lock;  // Array of mutexes

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
        
        size_t offset = file_size;
        file_size += size;
        if (file_size > allocated_size) {
            throw std::runtime_error("Exceeded pre-allocated file size");
        }
        
        return offset;
    }

    size_t allocate_node(int64_t parent_level) {
        // DEBUG_PRINT("Locking alloc_memory_mutex");
        alloc_memory_mutex.lock();
        size_t offset = allocate_space(sizeof(TrieNode));
        alloc_memory_mutex.unlock();
        // DEBUG_PRINT("Unlocking alloc_memory_mutex");
        TrieNode* new_node = get_node(offset);
        new_node->node_level = parent_level + 1; 

        // DEBUG_PRINT("Locking alloc_node_mutex");
        alloc_node_mutex.lock();
        if (new_node->node_level <= context_length){
            node_counter += 1;
            node_mutex_counter += 1;
            new_node->node_index = node_counter;
            new_node->node_mutex_index = node_mutex_counter;
            ctxLen_array[new_node->node_index] = new_node->node_level;
        } else {
            node_mutex_counter += 1;
            new_node->node_mutex_index = node_mutex_counter;
            new_node->node_index = -1;
        }
        alloc_node_mutex.unlock();
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


    double calculate_and_sum_entropy(size_t root_offset) {
        auto start_time = std::chrono::steady_clock::now();
        const auto timeout_duration = std::chrono::seconds(5000);
        double total_entropy = 0.0;
        std::stack<size_t> node_stack;
        node_stack.push(root_offset);

        while (!node_stack.empty()) {
            if (std::chrono::steady_clock::now() - start_time > timeout_duration) {
                throw std::runtime_error("Entropy calculation timed out after 5000 seconds. Returning default value.");
                std::cout << "Entropy calculation timed out after 5000 seconds. Returning default value." << std::endl;
                return -1;
            }

            size_t current_offset = node_stack.top();
            node_stack.pop();

            TrieNode* node = get_node(current_offset);
            std::pair<int64_t, int64_t>* children = get_children(node);
            
            double Pi_j = static_cast<double>(node->count) / num_total_contexts;
            double context_entropy = 0.0;
            int64_t context_total = 0;

            for (int64_t i = 0; i < node->num_children; ++i) {
                TrieNode* child = get_node(children[i].second);
                context_total += child->count;
            }

            if (context_total > 0) {
                for (int64_t i = 0; i < node->num_children; ++i) {
                    TrieNode* child = get_node(children[i].second);
                    double p_t = static_cast<double>(child->count) / context_total;
                    if (p_t > 0) {
                        context_entropy -= p_t * std::log(p_t);
                    }
                }
            }

            double node_contribution = Pi_j * context_entropy;
            total_entropy += node_contribution;

            if (node->num_children > 0) {
                num_unique_contexts++;
                num_unique_contexts_per_level[node->node_level]++;
                entropy_per_level[node->node_level] += node_contribution;
            }

            for (int64_t i = 0; i < node->num_children; ++i) {
                node_stack.push(children[i].second);
            }
        }

        return total_entropy;
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
    Trie_memap_sorted_OptExp(const std::string& fname, size_t initial_size_gb, int64_t context_length) 
    : filename(fname + ".bin"), context_length(context_length), countLog_array(fname + "_countLog_arr.dat", array_size), 
    ctxLen_array(fname + "_ctxLen_arr.dat", array_size), ctxCount_array(fname + "_ctxCount_arr.dat", array_size)
    , logcalc_memory(size_logcalc_memory, -1), mutex_array_lock(array_size) {
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

        DEBUG_PRINT("Trie initialized with allocated size: " << allocated_size << " bytes");
    }

    // Constructor to load an existing Trie from a file
    Trie_memap_sorted_OptExp(const std::string& fname) : 
    filename(fname + ".bin"), countLog_array(fname + "_countLog_arr.dat", array_size), 
    ctxLen_array(fname + "_ctxLen_arr.dat", array_size), ctxCount_array(fname + "_ctxCount_arr.dat", array_size)
    , logcalc_memory(size_logcalc_memory, -1), mutex_array_lock(array_size){
        // Step 1: Open the file
        fd = open(filename.c_str(), O_RDWR);
        if (fd == -1) {
            perror("Error opening file");
            exit(EXIT_FAILURE);
        }

        // Step 2: Get the size of the file
        struct stat fileInfo;
        if (fstat(fd, &fileInfo) == -1) {
            perror("Error getting file size");
            exit(EXIT_FAILURE);
        }
        allocated_size = fileInfo.st_size;

        std::cout << "Memory-mapped file size: " << allocated_size / pow(1024, 3) << " Giga bytes" << std::endl;

        // Step 3: Memory-map the file
        mapped_memory = (char*) mmap(nullptr, allocated_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            perror("Error mmapping the file");
            exit(EXIT_FAILURE);
        }

        metadata_filename = filename + "_metadata.bin";

        // Step 4: Set the root node to the start of the mapped file memory
        TrieNode* root = reinterpret_cast<TrieNode*>(mapped_memory);

        load_metadata(metadata_filename);

    }

    void save_metadata() {
        std::ofstream file(metadata_filename);
        if (file.is_open()) {
            file << file_size << "\n";
            file << allocated_size << "\n";
            file << num_unique_contexts.load() << "\n";
            file << num_total_contexts.load() << "\n";
            file << context_length << "\n";
            file << node_counter << "\n";

            for (const auto& [level, count] : num_unique_contexts_per_level) {
                file << "level " << level << " " << count << "\n";
            }
            file.close();
        }

        std::cout << "----Saver----" << std::endl;
        std::cout << "Metadata saved to " << metadata_filename << std::endl;
        std::cout << "Memory-mapped file size: " << file_size << " Giga bytes" << std::endl;
        std::cout << "Memory-mapped allocated size: " << allocated_size << " Giga bytes" << std::endl;

    }

    void load_metadata(const std::string& metadata_filename) {
        // Open the metadata file
        std::ifstream file(metadata_filename);
        if (file.is_open()) {
            file >> file_size;
            file >> allocated_size;
            int64_t num_unique_contexts_val, num_total_contexts_val;
            file >> num_unique_contexts_val;
            file >> num_total_contexts_val;
            num_unique_contexts.store(num_unique_contexts_val);
            num_total_contexts.store(num_total_contexts_val);
            file >> context_length;
            file >> node_counter;

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


    ~Trie_memap_sorted_OptExp() {
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

    void insert_context(const torch::Tensor& tensor, int64_t column) {
        // Ensure that the input tensor is 2D and of type int64 (torch::kInt64)
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");


        // DEBUG_PRINT("HERE In the insert context function for some thread");

        int64_t c_t_temp;

        auto accessor = tensor.accessor<int64_t, 2>();
        int64_t current_level = 0;
        size_t current_offset = 0;
        for (int64_t j = 0; j < accessor.size(1); j++) {
            int64_t value = accessor[column][j];
            
            TrieNode* current = get_node(current_offset);
            // DEBUG_PRINT("Locking a parent node");
            // DEBUG_PRINT(current->node_index);
            // current->node_mutex.lock();
            mutex_array_lock[current->node_mutex_index].lock();
            // DEBUG_PRINT("After locking a node");
            // sleep(5);

            // DEBUG_PRINT("f0");
            if (j > 0) {
                num_total_contexts += 1;
            }
            
            // DEBUG_PRINT("f1");
            int64_t child_index = -1;
            if (current->num_children > 0) {
                std::pair<int64_t, int64_t>* children = get_children(current);
                child_index = find_child(children, current->num_children, value);
            }

            // DEBUG_PRINT("f2");
            if (child_index == -1) {
                size_t new_node_offset = allocate_node(current->node_level);
                TrieNode* new_node = get_node(new_node_offset);
                // DEBUG_PRINT("f3");
                new_node->count = 0;
                new_node->num_children = 0;
                new_node->children_offset = 0;
                new_node->node_level = current_level + 1;

                if (new_node->node_level <= context_length) {
                    num_total_contexts_per_level[new_node->node_level]++;
                }

                // DEBUG_PRINT("Locking alloc_memory_mutex");
                // sleep(10);
                alloc_memory_mutex.lock();
                if (current->num_children == 0) {
                    current->children_offset = allocate_children(1);
                } else {
                    size_t new_children_offset = allocate_children(current->num_children + 1);
                    std::memcpy(mapped_memory + new_children_offset, get_children(current), 
                                current->num_children * sizeof(std::pair<int64_t, int64_t>));
                    current->children_offset = new_children_offset;
                }
                alloc_memory_mutex.unlock();
                // DEBUG_PRINT("Unlocking alloc_memory_mutex");
                // sleep(10);

                insert_child(current, value, new_node_offset);
                current_offset = new_node_offset;
            } else {
                // DEBUG_PRINT("f4");
                current_offset = get_children(current)[child_index].second;
            }

            // DEBUG_PRINT("f5");
            
            // DEBUG_PRINT("Locking a child node");
            // DEBUG_PRINT(get_node(current_offset)->node_index);
            mutex_array_lock[get_node(current_offset)->node_mutex_index].lock();
            // get_node(current_offset)->node_mutex.lock();
            // DEBUG_PRINT("f6");

            c_t_temp = get_node(current_offset)->count;

            if (current->node_index > 0 && c_t_temp > 0 && current->node_level <= 32){
                if(logcalc_memory[c_t_temp] == -1){
                    logcalc_memory[c_t_temp] = (c_t_temp + 1) * std::log(c_t_temp + 1) - (c_t_temp) * std::log(c_t_temp);
                }
                countLog_array[current->node_index] += logcalc_memory[c_t_temp];
            }
            ctxCount_array[current->node_index] += 1;
            
            // DEBUG_PRINT("f7");
            get_node(current_offset)->count++;
            // get_node(current_offset)->node_mutex.unlock();
            // DEBUG_PRINT("Unlocking a child node");
            mutex_array_lock[get_node(current_offset)->node_mutex_index].unlock();
            current_level++;

            // current->node_mutex.unlock();
            // DEBUG_PRINT("Unlocking a parent node");
            mutex_array_lock[current->node_mutex_index].unlock();
        }


    }

    int insert(torch::Tensor tensor) {
        // Ensure that the input tensor is 2D and of type int64 (torch::kInt64)
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");


        // DEBUG_PRINT("In the insert function ...");
        // Multi-threaded insertion: create a thread for each word
        std::vector<std::thread> threads;
        // Create a thread for each column (word) in the tensor
        for (int64_t col = 0; col < tensor.size(0); col++) {
            threads.emplace_back([this, &tensor, col]() { this->insert_context(tensor, col); });
        }
        
        // Join all threads
        for (auto& th : threads) {
            th.join();
        }

        // Check if we're close to the memory limit after insertion
        if (is_close_to_memory_limit()) {
            // Save the model
            save_metadata();
            return 0;  // Indicating we're running out of memory
        }

        return 1;  // Indicating we're not running out of memory
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


    double calculate_and_get_entropy() {
        num_unique_contexts = 0;
        for (auto& pair : num_unique_contexts_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }
        for (auto& pair : entropy_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }

        double total_entropy = calculate_and_sum_entropy(0);  // Start from the root
        num_unique_contexts -= 1; // Remove the count for the root node
        
        return total_entropy ;  // Normalize by total number of nodes
    }


    double calculate_and_get_entropy_faster() {
        
        // int64_t total_N = 0;
        double total_entropy = 0;
        double entropy_temp = 0;

        for (auto& pair : num_unique_contexts_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }
        for (auto& pair : entropy_per_level) {
            pair.second = 0;  // Set the count to zero for each level
        }

        int64_t total_counter = 0;
        int counter = 0;
        for(int j = 1; j < node_counter; j++){
            entropy_temp = countLog_array[j] - ctxCount_array[j] * log(ctxCount_array[j]);

            if (ctxCount_array[j] == 0){
                counter += 1;
                if (counter == 100){
                    DEBUG_PRINT("Quitting since the counter is 0.");
                    return 0;
                }
            }
            num_unique_contexts_per_level[ctxLen_array[j]] += 1;
            entropy_per_level[ctxLen_array[j]] += entropy_temp;
            total_entropy += entropy_temp;

            total_counter += ctxCount_array[j];
        }  
        
        total_entropy = -total_entropy / total_counter;
        for(int t = 0; t < context_length; t++){
            entropy_per_level[t] /= -total_counter;
        }

        num_unique_contexts = node_counter;
        
        return total_entropy ;  // Normalize by total number of nodes
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
    py::class_<Trie_memap_sorted_OptExp>(m, "Trie_memap_sorted_OptExp")
        .def(py::init<const std::string&, size_t, int64_t>())
        .def(py::init<const std::string&>())
        .def("insert", &Trie_memap_sorted_OptExp::insert)
        .def("collect_all_sequences", [](Trie_memap_sorted_OptExp& trie) {
            auto sequences = trie.collect_all_sequences();
            return convert_to_python_dict(sequences);
        })
        .def("calculate_and_get_entropy", &Trie_memap_sorted_OptExp::calculate_and_get_entropy)
        .def("get_memory_usage", &Trie_memap_sorted_OptExp::get_memory_usage)
        .def("get_allocated_size", &Trie_memap_sorted_OptExp::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_memap_sorted_OptExp::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_memap_sorted_OptExp::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("get_children_distribution", &Trie_memap_sorted_OptExp::get_children_distribution)
        .def("get_node_count", &Trie_memap_sorted_OptExp::get_node_count)
        .def("get_node_level", &Trie_memap_sorted_OptExp::get_node_level)
        .def("get_num_unique_contexts_per_level", &Trie_memap_sorted_OptExp::get_num_unique_contexts_per_level)
        .def("get_num_total_contexts_per_level", &Trie_memap_sorted_OptExp::get_num_total_contexts_per_level)
        .def("get_entropy_per_level", &Trie_memap_sorted_OptExp::get_entropy_per_level)
        .def("calculate_and_get_entropy_faster", &Trie_memap_sorted_OptExp::calculate_and_get_entropy_faster)
        .def("load_metadata", &Trie_memap_sorted_OptExp::load_metadata)
        .def("save_metadata", &Trie_memap_sorted_OptExp::save_metadata);


    py::class_<MemMapArray<int64_t>>(m, "MemMapArrayInt")
        .def(py::init<const std::string&, size_t>())
        .def("get_size", &MemMapArray<int64_t>::getSize)
        .def("__getitem__", [](MemMapArray<int64_t> &self, size_t index) { return self[index]; });
    
    py::class_<MemMapArray<double>>(m, "MemMapArrayDouble")
        .def(py::init<const std::string&, size_t>())
        .def("get_size", &MemMapArray<double>::getSize)
        .def("__getitem__", [](MemMapArray<double> &self, size_t index) { return self[index]; });

}