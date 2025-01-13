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
    std::ifstream file;
    size_t file_size;
    size_t num_tokens;
    size_t context_length;
    size_t stride;
    bool is_root;
    size_t root_ctx_len;
    std::vector<size_t> valid_indices;
    size_t vocab_size;
    std::string filepath;
    static constexpr size_t CHUNK_SIZE = 1024 * 1024;  // 1M tokens per chunk
    std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> final_counts;
    size_t num_bins;
    std::vector<std::vector<py::tuple>> bins;
    std::vector<size_t> bin_sums;

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
        , filepath(data_path)
        , num_bins(num_bins)
        , bins(num_bins)  // Initialize in constructor
        , bin_sums(num_bins, 0)  // Initialize in constructor
    {
        std::cout << "\nInitializing TokenizedDataset...\n";
        
        file.open(data_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_path);
        }

        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        num_tokens = file_size / sizeof(uint16_t);
        num_tokens = static_cast<size_t>(num_tokens * (data_percentage / 100.0));

        if (!valid_indices.empty()) {
            std::cout << "Processing valid indices... ";
            this->valid_indices = valid_indices;
            auto it = std::remove_if(this->valid_indices.begin(), this->valid_indices.end(),
                [this](size_t idx) { return idx + this->context_length + 1 > this->num_tokens; });
            this->valid_indices.erase(it, this->valid_indices.end());
        }
        

        calcVocabSize();

        // std::vector<std::vector<py::tuple>> bins(num_bins);
        // std::vector<size_t> bin_sums(num_bins, 0);

        std::cout << "\nDataset initialization complete:\n";
        std::cout << "Loaded " << num_tokens << " tokens\n";
        std::cout << "Total available windows: " << getNumWindows() << "\n";
        std::cout << "Vocabulary size: " << vocab_size << "\n\n";
    }

    // First modify analyze_window_startPairs to only return counts
    py::dict analyze_window_startPairs(size_t prefix_length = 2) {
        const int num_threads = omp_get_max_threads();
        std::cout << "Using " << num_threads << " threads\n";

        size_t total_windows = (num_tokens - prefix_length - 1) / stride + 1;
        size_t windows_per_chunk = (total_windows + num_threads - 1) / num_threads;
        
        std::vector<std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash>> thread_counts(num_threads);
        
        ProgressBar progress(total_windows, "Analyzing windows");
        
        #pragma omp parallel num_threads(num_threads)
        {
            int thread_id = omp_get_thread_num();
            size_t start_window = thread_id * windows_per_chunk;
            size_t end_window = std::min(start_window + windows_per_chunk, total_windows);
            
            std::vector<uint16_t> buffer(prefix_length);
            std::ifstream local_file(filepath, std::ios::binary);
            
            for (size_t window = start_window; window < end_window; ++window) {
                size_t start_pos = window * stride;
                if (start_pos + prefix_length <= num_tokens) {
                    local_file.seekg(start_pos * sizeof(uint16_t));
                    local_file.read(reinterpret_cast<char*>(buffer.data()), prefix_length * sizeof(uint16_t));
                    
                    auto token_pair = std::make_pair(buffer[0], buffer[1]);
                    thread_counts[thread_id][token_pair]++;
                }
                
                #pragma omp critical
                {
                    progress.update();
                }
            }
        }
        progress.finish();

        // Merge results
        std::cout << "Merging results from all threads...\n";
        // final_counts;
        
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

        // Create output files and their mutexes
        std::vector<std::ofstream> bin_files(bins.size());
        std::vector<std::mutex> bin_mutexes(bins.size());
        
        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            std::string filename = output_dir + "group" + std::to_string(bin_idx) + "/shuffled_indices_locations.txt";
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

    // Rest of the class remains the same...
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
        
        file.seekg(start_idx * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(window.data()), window_size * sizeof(uint16_t));

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

private:
    

    void calcVocabSize() {
        vocab_size = 0;
        std::vector<uint16_t> buffer(8192);
        
        size_t total_chunks = (file_size + buffer.size() * sizeof(uint16_t) - 1) / (buffer.size() * sizeof(uint16_t));
        ProgressBar progress(total_chunks, "Calculating vocab size");
        
        for (size_t pos = 0; pos < file_size; pos += buffer.size() * sizeof(uint16_t)) {
            size_t bytes_to_read = std::min(buffer.size() * sizeof(uint16_t), file_size - pos);
            file.seekg(pos);
            file.read(reinterpret_cast<char*>(buffer.data()), bytes_to_read);
            
            size_t tokens_read = bytes_to_read / sizeof(uint16_t);
            for (size_t i = 0; i < tokens_read; ++i) {
                vocab_size = std::max(vocab_size, static_cast<size_t>(buffer[i]));
            }
            
            progress.update();
        }
        progress.finish();
        
        vocab_size++;
        file.seekg(0);
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

PYBIND11_MODULE(fast_tokenized_dataset, m) {
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
        .def("getNumWindows", &TokenizedDataset::getNumWindows);

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