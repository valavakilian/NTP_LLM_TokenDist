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


class Trie_memap_sorted {
private:
    int fd;
    char* mapped_memory;
    size_t file_size;
    size_t allocated_size;
    std::string filename;
    std::string filename_metadata;
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


    double calculate_and_sum_entropy(size_t node_offset) {
        TrieNode* node = get_node(node_offset);
        std::pair<int64_t, int64_t>* children = get_children(node);
        
        double Pi_j = static_cast<double>(node->count) / num_total_contexts;
        
        
        double context_entropy = 0.0;
        int64_t context_total = 0;

        if (node->num_children > 0){
            num_unique_contexts++;  // Increment node count when allocating a new node
            num_unique_contexts_per_level[node->node_level]++;
        }

        // Calculate total count for this context
        for (int64_t i = 0; i < node->num_children; ++i) {
            TrieNode* child = get_node(children[i].second);
            context_total += child->count;
        }

        // Calculate entropy for this context
        if (context_total > 0) {
            for (int64_t i = 0; i < node->num_children; ++i) {
                TrieNode* child = get_node(children[i].second);
                double p_t = static_cast<double>(child->count) / context_total;
                if (p_t > 0) {
                    context_entropy -= p_t * log(p_t);
                }
            }
        }

        double node_contribution = Pi_j * context_entropy;
        if (node->num_children > 0){
            double Pi_j_level = static_cast<double>(node->count) / num_total_contexts_per_level[node -> node_level];
            entropy_per_level[node -> node_level] += Pi_j_level * context_entropy;
        }

        // Recursively calculate entropy for all children
        double children_entropy = 0.0;
        for (int64_t i = 0; i < node->num_children; ++i) {
            children_entropy += calculate_and_sum_entropy(children[i].second);
        }

        return node_contribution + children_entropy;
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


    void load_existing_trie() {
        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size");
        }
        file_size = st.st_size;
        allocated_size = file_size;

        mapped_memory = static_cast<char*>(mmap(NULL, allocated_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file to memory");
        }

        TrieNode* root = get_node(0);
        if (root->num_children == 0) {
            throw std::runtime_error("Invalid root node: count or num_children is zero");
        }
        if (root->children_offset >= file_size) {
            throw std::runtime_error("Invalid children offset: larger than file size");
        }
        num_unique_contexts.store(root->count);
        num_total_contexts.store(root->num_children);  // Assuming we store total contexts in root's num_children
        context_length = root->node_level;  // Assuming we store context_length in root's node_level
        // Add more initialization as needed
    }


public:
    Trie_memap_sorted(const std::string& fname,const std::string&  fname_metadata, size_t initial_size_gb, int64_t context_length) : filename(fname), filename_metadata(fname_metadata) , context_length(context_length) {
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

    // New constructor for loading an existing trie
    Trie_memap_sorted(const std::string& fname,const std::string&  fname_metadata) : filename(fname), filename_metadata(fname_metadata) {
        fd = open(filename.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            throw std::runtime_error("Failed to open existing file for memory mapping");
        }

        load_existing_trie();

        DEBUG_PRINT("Existing Trie loaded with size: " << allocated_size << " bytes");
    }

     ~Trie_memap_sorted() {
        if (mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, allocated_size);
        }
        if (fd != -1) {
            // Truncate the file to the actually used size before closing
            ftruncate(fd, file_size);
            close(fd);
        }
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
    }


    // void insert(torch::Tensor tensor) {
    //     auto accessor = tensor.accessor<int64_t, 2>();
    //     for (int64_t i = 0; i < accessor.size(0); i++) {
    //         size_t current_offset = 0;
    //         for (int64_t j = 0; j < accessor.size(1); j++) {
    //             int64_t value = accessor[i][j];
    //             TrieNode* current = get_node(current_offset);
                
    //             // Debug: Check node validity
    //             if (current == nullptr || current_offset >= file_size) {
    //                 std::cout << "Error: Invalid node at offset " << current_offset << std::endl;
    //                 return;
    //             }

    //             int64_t child_index = -1;
    //             if (current->num_children > 0) {
    //                 std::pair<int64_t, int64_t>* children = get_children(current);
    //                 // Debug: Check children array validity
    //                 if (children == nullptr || current->children_offset >= file_size) {
    //                     std::cout << "Error: Invalid children array at offset " << current->children_offset << std::endl;
    //                     return;
    //                 }
    //                 child_index = find_child(children, current->num_children, value);
    //             }

    //             if (child_index == -1) {
    //                 size_t new_node_offset = allocate_node(current->node_level);
    //                 // save_metadata();
    //                 // Debug: Check allocation
    //                 if (new_node_offset >= file_size) {
    //                     std::cout << "Error: Node allocation failed. Offset: " << new_node_offset << std::endl;
    //                     return;
    //                 }

    //                 if (current->num_children == 0) {
    //                     current->children_offset = allocate_children(1);
    //                 } else {
    //                     size_t new_children_offset = allocate_children(current->num_children + 1);
    //                     // Debug: Check reallocation
    //                     if (new_children_offset >= file_size) {
    //                         std::cout << "Error: Children reallocation failed. Offset: " << new_children_offset << std::endl;
    //                         return;
    //                     }
    //                     std::memcpy(mapped_memory + new_children_offset, get_children(current), 
    //                                 current->num_children * sizeof(std::pair<int64_t, int64_t>));
    //                     current->children_offset = new_children_offset;
    //                 }

    //                 insert_child(current, value, new_node_offset);
    //                 current_offset = new_node_offset;
    //             } else {
    //                 current_offset = get_children(current)[child_index].second;
    //             }

    //             // std::cout << "File size " << file_size << std::endl;
    //             get_node(current_offset)->count++;
    //         }
    //     }
    //     // save_metadata();
    // }

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

    void save_metadata() {
        std::ofstream file(filename_metadata, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for saving metadata");
        }

        // Save metadata
        file.write(reinterpret_cast<char*>(&file_size), sizeof(size_t));
        file.write(reinterpret_cast<char*>(&allocated_size), sizeof(size_t));
        int64_t unique_contexts = num_unique_contexts.load();
        int64_t total_contexts = num_total_contexts.load();
        file.write(reinterpret_cast<char*>(&unique_contexts), sizeof(int64_t));
        file.write(reinterpret_cast<char*>(&total_contexts), sizeof(int64_t));
        file.write(reinterpret_cast<char*>(&context_length), sizeof(int64_t));

        // Save num_unique_contexts_per_level
        size_t map_size = num_unique_contexts_per_level.size();
        file.write(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        for (const auto& pair : num_unique_contexts_per_level) {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
        }

        // Save num_total_contexts_per_level
        map_size = num_total_contexts_per_level.size();
        file.write(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        for (const auto& pair : num_total_contexts_per_level) {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
        }

        // Save entropy_per_level
        map_size = entropy_per_level.size();
        file.write(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        for (const auto& pair : entropy_per_level) {
            file.write(reinterpret_cast<const char*>(&pair.first), sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(&pair.second), sizeof(double));
        }

        file.close();
    }

    void load_metadata(const std::string& metadata_filename) {
        std::ifstream file(metadata_filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for loading metadata");
        }

        // Load metadata
        file.read(reinterpret_cast<char*>(&file_size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&allocated_size), sizeof(size_t));
        int64_t unique_contexts, total_contexts;
        file.read(reinterpret_cast<char*>(&unique_contexts), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&total_contexts), sizeof(int64_t));
        num_unique_contexts.store(unique_contexts);
        num_total_contexts.store(total_contexts);
        file.read(reinterpret_cast<char*>(&context_length), sizeof(int64_t));

        // Load num_unique_contexts_per_level
        size_t map_size;
        file.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        num_unique_contexts_per_level.clear();
        for (size_t i = 0; i < map_size; ++i) {
            int64_t level;
            int count;
            file.read(reinterpret_cast<char*>(&level), sizeof(int64_t));
            file.read(reinterpret_cast<char*>(&count), sizeof(int));
            num_unique_contexts_per_level[level] = count;
        }

        // Load num_total_contexts_per_level
        file.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        num_total_contexts_per_level.clear();
        for (size_t i = 0; i < map_size; ++i) {
            int64_t level;
            int count;
            file.read(reinterpret_cast<char*>(&level), sizeof(int64_t));
            file.read(reinterpret_cast<char*>(&count), sizeof(int));
            num_total_contexts_per_level[level] = count;
        }

        // Load entropy_per_level
        file.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        entropy_per_level.clear();
        for (size_t i = 0; i < map_size; ++i) {
            int64_t level;
            double entropy;
            file.read(reinterpret_cast<char*>(&level), sizeof(int64_t));
            file.read(reinterpret_cast<char*>(&entropy), sizeof(double));
            entropy_per_level[level] = entropy;
        }

        file.close();
    }


    bool validate_load() {
        if (!mapped_memory) {
            std::cout << "Error: Memory not mapped" << std::endl;
            return false;
        }

        TrieNode* root = get_node(0);
        
        // Check root node integrity
        std::cout << "Root node check:" << std::endl;
        std::cout << "  Is root: " << (root->is_root ? "true" : "false") << std::endl;
        std::cout << "  Node level: " << root->node_level << std::endl;
        std::cout << "  Count: " << root->count << std::endl;
        std::cout << "  Number of children: " << root->num_children << std::endl;
        std::cout << "  Children offset: " << root->children_offset << std::endl;
        std::cout << "  Entropy: " << root->entropy << std::endl;

        // Validate root node
        if (!root->is_root || root->node_level != 0) {
            std::cout << "Error: Invalid root node" << std::endl;
            return false;
        }

        // Check if children offset is within file bounds
        if (root->num_children > 0 && (root->children_offset + root->num_children * sizeof(std::pair<int64_t, int64_t>) > file_size)) {
            std::cout << "Error: Children offset out of bounds" << std::endl;
            return false;
        }

        // Print first few children of root node
        std::cout << "First few children of root node:" << std::endl;
        std::pair<int64_t, int64_t>* children = get_children(root);
        for (int i = 0; i < std::min(static_cast<int64_t>(5), root->num_children); ++i) {
            std::cout << "  Child " << i << ": value = " << children[i].first 
                    << ", offset = " << children[i].second << std::endl;
            
            // Check if child node offset is within file bounds
            if (children[i].second >= file_size) {
                std::cout << "Error: Child node offset out of bounds" << std::endl;
                return false;
            }
            
            // Print some info about the child node
            TrieNode* child_node = get_node(children[i].second);
            std::cout << "    Level: " << child_node->node_level 
                    << ", Count: " << child_node->count 
                    << ", Num children: " << child_node->num_children << std::endl;
        }
        
        // Print metadata
        std::cout << "Metadata:" << std::endl;
        std::cout << "  File size: " << file_size / pow(1024, 3) << std::endl;
        std::cout << "  Allocated size: " << allocated_size / pow(1024, 3) << std::endl;
        std::cout << "  Num unique contexts: " << num_unique_contexts.load() << std::endl;
        std::cout << "  Num total contexts: " << num_total_contexts.load() << std::endl;
        std::cout << "  Context length: " << context_length << std::endl;

        return true;
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
    py::class_<Trie_memap_sorted>(m, "Trie_memap_sorted")
        .def(py::init<const std::string&, const std::string&, size_t, int64_t>())
        .def(py::init<const std::string&, const std::string&>())
        .def("insert", &Trie_memap_sorted::insert)
        .def("collect_all_sequences", [](Trie_memap_sorted& trie) {
            auto sequences = trie.collect_all_sequences();
            return convert_to_python_dict(sequences);
        })
        .def("calculate_and_get_entropy", &Trie_memap_sorted::calculate_and_get_entropy)
        .def("get_memory_usage", &Trie_memap_sorted::get_memory_usage)
        .def("get_allocated_size", &Trie_memap_sorted::get_allocated_size)
        .def("get_num_unique_contexts", &Trie_memap_sorted::get_num_unique_contexts)  // New method to access num_unique_contexts
        .def("get_num_total_contexts", &Trie_memap_sorted::get_num_total_contexts)  // New method to access num_unique_contexts
        .def("get_children_distribution", &Trie_memap_sorted::get_children_distribution)
        .def("get_node_count", &Trie_memap_sorted::get_node_count)
        .def("get_node_level", &Trie_memap_sorted::get_node_level)
        .def("get_num_unique_contexts_per_level", &Trie_memap_sorted::get_num_unique_contexts_per_level)
        .def("get_num_total_contexts_per_level", &Trie_memap_sorted::get_num_total_contexts_per_level)
        .def("get_entropy_per_level", &Trie_memap_sorted::get_entropy_per_level)
        .def("save_metadata", &Trie_memap_sorted::save_metadata)
        .def("load_metadata", &Trie_memap_sorted::load_metadata)
        .def("validate_load", &Trie_memap_sorted::validate_load);
}