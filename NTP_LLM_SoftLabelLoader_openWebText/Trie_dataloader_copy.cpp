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


namespace py = pybind11;
using namespace py::literals;

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
            std::ifstream single_file(input_dir + "/single_transitions.bin", std::ios::binary);
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
            std::ifstream pair_file(input_dir + "/pair_transitions.bin", std::ios::binary);
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
        .def("load_transitions_from_file", &TokenizedDataset::load_transitions_from_file);

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