// token_analysis.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

namespace py = pybind11;

struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1);
    }
};

class TokenAnalyzer {
private:
    std::mutex map_mutex;
    std::atomic<size_t> processed_chunks{0};
    size_t total_chunks;
    size_t context_length;
    
    struct LocalStorage {
        std::vector<std::pair<uint16_t, uint16_t>> pairs;
        std::vector<size_t> positions;
    };

public:
    TokenAnalyzer(size_t ctx_len) : context_length(ctx_len) {
        std::cout << "Initializing TokenAnalyzer with context length: " << ctx_len << "\n";
    }

    void process_chunk(
        const uint16_t* data, 
        size_t start, 
        size_t end, 
        size_t stride,
        std::unordered_map<std::pair<uint16_t, uint16_t>, std::vector<size_t>, PairHash>& global_map,
        size_t total_size
    ) {
        LocalStorage local_storage;
        local_storage.pairs.reserve((end - start) / stride);
        local_storage.positions.reserve((end - start) / stride);

        // Only process if we have enough tokens for a full context
        for (size_t i = start; i < end && i + context_length < total_size; i += stride) {
            local_storage.pairs.emplace_back(data[i], data[i + 1]);
            local_storage.positions.push_back(i);
        }

        const size_t BATCH_SIZE = 10000;
        std::lock_guard<std::mutex> lock(map_mutex);
        
        for (size_t i = 0; i < local_storage.pairs.size(); i++) {
            global_map[local_storage.pairs[i]].push_back(local_storage.positions[i]);
            
            if (i % BATCH_SIZE == 0) {
                processed_chunks++;
                if (processed_chunks % 100 == 0) {
                    std::cout << "\rProcessed " << processed_chunks << "/" << total_chunks 
                              << " chunks (" << (processed_chunks * 100.0 / total_chunks) 
                              << "%)" << std::flush;
                }
            }
        }
    }
    
    py::tuple analyze_pairs(py::array_t<uint16_t> array, size_t stride, size_t num_threads, double data_percentage) {
        py::buffer_info buf = array.request();
        const uint16_t* data = static_cast<uint16_t*>(buf.ptr);
        const size_t total_size = buf.size;
        
        // Calculate how much data to use based on percentage
        const size_t use_size = static_cast<size_t>(total_size * (data_percentage / 100.0));
        std::cout << "Using " << use_size << " tokens out of " << total_size 
                  << " (" << data_percentage << "%)\n";
        
        const size_t MAX_CHUNK_SIZE = 1000000;
        const size_t chunk_size = std::min(MAX_CHUNK_SIZE, (use_size - 1) / num_threads);
        total_chunks = (use_size + chunk_size - 1) / chunk_size;
        
        std::cout << "Starting analysis with " << num_threads << " threads\n";
        std::cout << "Total size: " << use_size << ", Chunk size: " << chunk_size << "\n";
        std::cout << "Context length: " << context_length << ", Stride: " << stride << "\n";
        
        std::unordered_map<std::pair<uint16_t, uint16_t>, std::vector<size_t>, PairHash> global_map;
        global_map.reserve(1000000);
        
        const size_t THREAD_BATCH_SIZE = 4;
        for (size_t batch_start = 0; batch_start < use_size; batch_start += chunk_size * THREAD_BATCH_SIZE) {
            std::vector<std::thread> threads;
            
            for (size_t t = 0; t < THREAD_BATCH_SIZE && batch_start + t * chunk_size < use_size; t++) {
                size_t start = batch_start + t * chunk_size;
                size_t end = std::min(start + chunk_size, use_size);
                
                threads.emplace_back(&TokenAnalyzer::process_chunk, this, 
                                   data, start, end, stride, std::ref(global_map), use_size);
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
        }
        
        std::cout << "\nConverting results to Python...\n";
        
        py::dict counts;
        py::dict locations;
        
        size_t i = 0;
        for (const auto& entry : global_map) {
            py::tuple pair = py::make_tuple(entry.first.first, entry.first.second);
            counts[pair] = entry.second.size();
            locations[pair] = py::cast(entry.second);
            
            if (++i % 10000 == 0) {
                std::cout << "\rConverted " << i << " pairs..." << std::flush;
            }
        }
        
        std::cout << "\nDone! Found " << global_map.size() << " unique pairs.\n";
        return py::make_tuple(counts, locations);
    }

    std::vector<std::vector<std::pair<uint16_t, uint16_t>>> distribute_to_bins(
        const std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash>& counts,
        size_t num_bins
    ) {
        std::cout << "Starting bin distribution...\n";
        
        std::vector<std::pair<std::pair<uint16_t, uint16_t>, size_t>> sorted_pairs;
        sorted_pairs.reserve(counts.size());
        
        for (const auto& entry : counts) {
            sorted_pairs.emplace_back(entry.first, entry.second);
        }
        
        std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "Sorted " << sorted_pairs.size() << " pairs by frequency\n";
        
        std::vector<std::vector<std::pair<uint16_t, uint16_t>>> bins(num_bins);
        std::vector<size_t> bin_sums(num_bins, 0);
        
        for (const auto& entry : sorted_pairs) {
            auto min_it = std::min_element(bin_sums.begin(), bin_sums.end());
            size_t min_idx = std::distance(bin_sums.begin(), min_it);
            
            bins[min_idx].push_back(entry.first);
            bin_sums[min_idx] += entry.second;
        }
        
        std::cout << "\nBin distribution statistics:\n";
        for (size_t i = 0; i < num_bins; i++) {
            std::cout << "Bin " << i << ": " << bins[i].size() << " pairs, sum = " 
                      << bin_sums[i] << "\n";
        }
        
        return bins;
    }
};

PYBIND11_MODULE(token_analysis, m) {
    py::class_<TokenAnalyzer>(m, "TokenAnalyzer")
        .def(py::init<size_t>())
        .def("analyze_pairs", &TokenAnalyzer::analyze_pairs)
        .def("distribute_to_bins", [](TokenAnalyzer& self, py::dict counts_dict, size_t num_bins) {
            std::unordered_map<std::pair<uint16_t, uint16_t>, size_t, PairHash> cpp_counts;
            
            for (const auto& item : counts_dict) {
                py::tuple pair_tuple = item.first.cast<py::tuple>();
                std::pair<uint16_t, uint16_t> pair = std::make_pair(
                    pair_tuple[0].cast<uint16_t>(),
                    pair_tuple[1].cast<uint16_t>()
                );
                size_t count = item.second.cast<size_t>();
                cpp_counts[pair] = count;
            }
            
            auto result = self.distribute_to_bins(cpp_counts, num_bins);
            
            py::list py_result;
            for (const auto& bin : result) {
                py_result.append(py::cast(bin));
            }
            
            return py_result;
        });
}