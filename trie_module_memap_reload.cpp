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
    double entropy;  // New field to store entropy
    bool is_root;  // New field to indicate if this is the root node
    int64_t node_level;
};


class Trie_memap_sorted_old {
private:
    int fd;
    char* mapped_memory;
    size_t file_size;
    size_t allocated_size;
    std::string filename;
    std::string metadata_filename;
    std::atomic<int64_t> num_unique_contexts;  // New global parameter for unqie context count
    std::atomic<int64_t> num_total_contexts;  // New global parameter for total context count

    int64_t context_length;
    std::map<int64_t, int> num_unique_contexts_per_level;  // int64_t for the level, int for the count
    std::map<int64_t, int> num_total_contexts_per_level;

    std::map<int64_t, double> entropy_per_level;

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

    // size_t allocate_space(size_t size) {
    //     size_t offset = file_size;
    //     file_size += size;
    //     if (file_size > allocated_size) {
    //         throw std::runtime_error("Exceeded pre-allocated file size");
    //     }
    //     return offset;
    // }

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


    // double calculate_and_sum_entropy(size_t node_offset) {
    //     TrieNode* node = get_node(node_offset);
    //     std::pair<int64_t, int64_t>* children = get_children(node);
        
    //     double Pi_j = static_cast<double>(node->count) / num_total_contexts;
        
        
    //     double context_entropy = 0.0;
    //     int64_t context_total = 0;

    //     if (node->num_children > 0){
    //         num_unique_contexts++;  // Increment node count when allocating a new node
    //         num_unique_contexts_per_level[node->node_level]++;
    //     }

    //     // Calculate total count for this context
    //     for (int64_t i = 0; i < node->num_children; ++i) {
    //         TrieNode* child = get_node(children[i].second);
    //         context_total += child->count;
    //     }

    //     // Calculate entropy for this context
    //     if (context_total > 0) {
    //         for (int64_t i = 0; i < node->num_children; ++i) {
    //             TrieNode* child = get_node(children[i].second);
    //             double p_t = static_cast<double>(child->count) / context_total;
    //             if (p_t > 0) {
    //                 context_entropy -= p_t * log(p_t);
    //             }
    //         }
    //     }

    //     double node_contribution = Pi_j * context_entropy;
    //     if (node->num_children > 0){
    //         double Pi_j_level = static_cast<double>(node->count) / num_total_contexts_per_level[node -> node_level];
    //         entropy_per_level[node -> node_level] += Pi_j_level * context_entropy;
    //     }

    //     // Recursively calculate entropy for all children
    //     double children_entropy = 0.0;
    //     for (int64_t i = 0; i < node->num_children; ++i) {
    //         children_entropy += calculate_and_sum_entropy(children[i].second);
    //     }

    //     return node_contribution + children_entropy;
    // }


    double calculate_and_sum_entropy(size_t root_offset) {
        auto start_time = std::chrono::steady_clock::now();
        const auto timeout_duration = std::chrono::seconds(5000);
        double total_entropy = 0.0;
        std::stack<size_t> node_stack;
        node_stack.push(root_offset);

        while (!node_stack.empty()) {
            if (std::chrono::steady_clock::now() - start_time > timeout_duration) {
                std::cout << "Entropy calculation timed out after 5000 seconds. Returning default value." << std::endl;
                return 100.0;
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
    Trie_memap_sorted_old(const std::string& fname, const std::string& metadata_fname, size_t initial_size_gb, int64_t context_length) : filename(fname),  metadata_filename(metadata_fname) , context_length(context_length) {
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

        // Initialize the root node
        file_size = sizeof(TrieNode);
        TrieNode* root = get_node(0);
        root->count = 0;
        root->num_children = 0;
        root->children_offset = 0;
        root->is_root = true;  // Set the root node indicator

        num_unique_contexts = 0;  // Count the root node
        num_total_contexts = 0;

        DEBUG_PRINT("Trie initialized with allocated size: " << allocated_size << " bytes");
    }

    // Constructor to load an existing Trie from a file
    Trie_memap_sorted_old(const std::string& fname, const std::string& metadata_fname) : filename(fname),  metadata_filename(metadata_fname) {
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


    ~Trie_memap_sorted_old() {
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

    void insert(torch::Tensor tensor) {
        TORCH_CHECK(tensor.dim() == 2, "Input tensor must be 2-dimensional");
        TORCH_CHECK(tensor.dtype() == torch::kInt64, "Input tensor must be of type int64");
        
        auto accessor = tensor.accessor<int64_t, 2>();
        for (int64_t i = 0; i < accessor.size(0); i++) {
            int64_t current_level = 0;
            size_t current_offset = 0;
            for (int64_t j = 0; j < accessor.size(1); j++) {
                int64_t value = accessor[i][j];
                TrieNode* current = get_node(current_offset);

                if (j > 0) {
                    num_total_contexts += 1;
                }
                
                int64_t child_index = -1;
                if (current->num_children > 0) {
                    std::pair<int64_t, int64_t>* children = get_children(current);
                    child_index = find_child(children, current->num_children, value);
                }

                if (child_index == -1) {
                    size_t new_node_offset = allocate_node(current->node_level);
                    TrieNode* new_node = get_node(new_node_offset);
                    new_node->count = 0;
                    new_node->num_children = 0;
                    new_node->children_offset = 0;
                    new_node->node_level = current_level + 1;

                    if (new_node->node_level <= context_length) {
                        num_total_contexts_per_level[new_node->node_level]++;
                    }

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

                get_node(current_offset)->count++;
                current_level++;
            }
        }

        // update_critical_info();
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

        // num_total_contexts = 0;
        double total_entropy = calculate_and_sum_entropy(0);  // Start from the root
        num_unique_contexts -= 1; // Remove the count for the root node

        DEBUG_PRINT("Count for the zeroths node " << get_node(0)->count);
        
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
    py::class_<Trie_memap_sorted_old>(m, "Trie_memap_sorted_old")
        .def(py::init<const std::string&, const std::string&, size_t, int64_t>())
        .def(py::init<const std::string&, const std::string&>())
        .def("insert", &Trie_memap_sorted_old::insert)
        .def("collect_all_sequences", [](Trie_memap_sorted_old& trie) {
            auto sequences = trie.collect_all_sequences();
            return convert_to_python_dict(sequences);
        })
        .def("calculate_and_get_entropy", &Trie_memap_sorted_old::calculate_and_get_entropy)
        .def("get_memory_usage", &Trie_memap_sorted_old::get_memory_usage)
        .def("get_allocated_size", &Trie_memap_sorted_old::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_memap_sorted_old::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_memap_sorted_old::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("get_children_distribution", &Trie_memap_sorted_old::get_children_distribution)
        .def("get_node_count", &Trie_memap_sorted_old::get_node_count)
        .def("get_node_level", &Trie_memap_sorted_old::get_node_level)
        .def("get_num_unique_contexts_per_level", &Trie_memap_sorted_old::get_num_unique_contexts_per_level)
        .def("get_num_total_contexts_per_level", &Trie_memap_sorted_old::get_num_total_contexts_per_level)
        .def("get_entropy_per_level", &Trie_memap_sorted_old::get_entropy_per_level)
        .def("load_metadata", &Trie_memap_sorted_old::load_metadata)
        .def("save_metadata", &Trie_memap_sorted_old::save_metadata);
}