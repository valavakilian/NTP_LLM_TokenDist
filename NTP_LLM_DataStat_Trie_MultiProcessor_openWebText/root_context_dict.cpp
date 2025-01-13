#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>

namespace py = pybind11;

// Hash function for token pairs
struct PairHash {
    std::size_t operator()(const std::pair<uint16_t, uint16_t>& p) const {
        return (static_cast<size_t>(p.first) << 16) | p.second;
    }
};

// Individual counter that can be assigned to a thread
class ThreadCounter {
public:
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, uint32_t>> single_counts;
    std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, uint32_t>, PairHash> pair_counts;
    uint16_t context_length;

    void save_to_file(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write context length
        file.write(reinterpret_cast<const char*>(&context_length), sizeof(context_length));

        if (context_length == 1) {
            write_single_counts(file);
        } else {
            write_pair_counts(file);
        }
        file.close();
    }

    void load_from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        // Read context length
        file.read(reinterpret_cast<char*>(&context_length), sizeof(context_length));

        if (context_length == 1) {
            read_single_counts(file);
        } else {
            read_pair_counts(file);
        }
        file.close();
    }

private:
    void write_single_counts(std::ofstream& file) const {
        size_t size = single_counts.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        for (const auto& [ctx, dist] : single_counts) {
            file.write(reinterpret_cast<const char*>(&ctx), sizeof(ctx));
            
            size_t dist_size = dist.size();
            file.write(reinterpret_cast<const char*>(&dist_size), sizeof(dist_size));
            
            for (const auto& [token, count] : dist) {
                file.write(reinterpret_cast<const char*>(&token), sizeof(token));
                file.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }
    }

    void write_pair_counts(std::ofstream& file) const {
        size_t size = pair_counts.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        for (const auto& [ctx_pair, dist] : pair_counts) {
            file.write(reinterpret_cast<const char*>(&ctx_pair.first), sizeof(ctx_pair.first));
            file.write(reinterpret_cast<const char*>(&ctx_pair.second), sizeof(ctx_pair.second));
            
            size_t dist_size = dist.size();
            file.write(reinterpret_cast<const char*>(&dist_size), sizeof(dist_size));
            
            for (const auto& [token, count] : dist) {
                file.write(reinterpret_cast<const char*>(&token), sizeof(token));
                file.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }
    }

    void read_single_counts(std::ifstream& file) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        single_counts.clear();
        for (size_t i = 0; i < size; i++) {
            uint16_t ctx;
            file.read(reinterpret_cast<char*>(&ctx), sizeof(ctx));
            
            size_t dist_size;
            file.read(reinterpret_cast<char*>(&dist_size), sizeof(dist_size));
            
            auto& dist = single_counts[ctx];
            for (size_t j = 0; j < dist_size; j++) {
                uint16_t token;
                uint32_t count;
                file.read(reinterpret_cast<char*>(&token), sizeof(token));
                file.read(reinterpret_cast<char*>(&count), sizeof(count));
                dist[token] = count;
            }
        }
    }

    void read_pair_counts(std::ifstream& file) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        pair_counts.clear();
        for (size_t i = 0; i < size; i++) {
            uint16_t first, second;
            file.read(reinterpret_cast<char*>(&first), sizeof(first));
            file.read(reinterpret_cast<char*>(&second), sizeof(second));
            
            size_t dist_size;
            file.read(reinterpret_cast<char*>(&dist_size), sizeof(dist_size));
            
            auto& dist = pair_counts[std::make_pair(first, second)];
            for (size_t j = 0; j < dist_size; j++) {
                uint16_t token;
                uint32_t count;
                file.read(reinterpret_cast<char*>(&token), sizeof(token));
                file.read(reinterpret_cast<char*>(&count), sizeof(count));
                dist[token] = count;
            }
        }
    }

    ThreadCounter(uint16_t ctx_len) : context_length(ctx_len) {}

    void process_chunk(const int64_t* data, size_t start_idx, size_t end_idx, size_t seq_length) {
        for (size_t i = start_idx; i < end_idx; i++) {
            const int64_t* seq = data + i * seq_length;
            
            if (context_length == 1) {
                uint16_t context = static_cast<uint16_t>(seq[0]);
                uint16_t next_token = static_cast<uint16_t>(seq[1]);
                single_counts[context][next_token]++;
            } else {
                std::pair<uint16_t, uint16_t> context(
                    static_cast<uint16_t>(seq[0]),
                    static_cast<uint16_t>(seq[1])
                );
                uint16_t next_token = static_cast<uint16_t>(seq[2]);
                pair_counts[context][next_token]++;
            }
        }
    }
};

class DistributionManager {
private:
    std::vector<std::unique_ptr<ThreadCounter>> thread_counters;
    uint16_t context_length;
    
public:
    DistributionManager(uint16_t ctx_len, size_t num_threads) : context_length(ctx_len) {
        if (ctx_len != 1 && ctx_len != 2) {
            throw std::runtime_error("Context length must be either 1 or 2");
        }
        
        // Create thread counters
        for (size_t i = 0; i < num_threads; i++) {
            thread_counters.push_back(std::make_unique<ThreadCounter>(ctx_len));
        }
    }

    void process_batch(py::array_t<int64_t, py::array::c_style> batch) {
        auto batch_buf = batch.request();
        if (batch_buf.ndim != 2) {
            throw std::runtime_error("Input must be 2-dimensional");
        }

        const int64_t* data = static_cast<int64_t*>(batch_buf.ptr);
        const size_t batch_size = batch_buf.shape[0];
        const size_t seq_length = batch_buf.shape[1];

        if (seq_length != context_length + 1) {
            throw std::runtime_error("Sequence length must be context_length + 1");
        }

        // Calculate chunk size
        size_t num_threads = thread_counters.size();
        size_t chunk_size = batch_size / num_threads;
        if (chunk_size == 0) {
            chunk_size = 1;
            num_threads = batch_size;
        }

        // Launch threads
        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; i++) {
            size_t start_idx = i * chunk_size;
            size_t end_idx = (i == num_threads - 1) ? batch_size : (i + 1) * chunk_size;
            
            threads.emplace_back(&ThreadCounter::process_chunk, 
                               thread_counters[i].get(),
                               data, start_idx, end_idx, seq_length);
        }

        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }
    }

    std::unique_ptr<ThreadCounter> merge_results() {
        auto merged = std::make_unique<ThreadCounter>(context_length);
        
        if (context_length == 1) {
            // Merge single context counts
            for (const auto& counter : thread_counters) {
                for (const auto& [ctx, dist] : counter->single_counts) {
                    auto& merged_dist = merged->single_counts[ctx];
                    for (const auto& [token, count] : dist) {
                        merged_dist[token] += count;
                    }
                }
            }
        } else {
            // Merge pair context counts
            for (const auto& counter : thread_counters) {
                for (const auto& [ctx_pair, dist] : counter->pair_counts) {
                    auto& merged_dist = merged->pair_counts[ctx_pair];
                    for (const auto& [token, count] : dist) {
                        merged_dist[token] += count;
                    }
                }
            }
        }
        
        return merged;
    }

    // Returns pointer to merged counter that the caller now owns
    std::unique_ptr<ThreadCounter> merge_and_reset() {
        auto merged = merge_results();
        // Clear existing thread counters to free memory
        for (auto& counter : thread_counters) {
            counter = std::make_unique<ThreadCounter>(context_length);
        }
        return merged;
    }

    py::dict get_distribution(py::object context) {
        py::dict result;
        auto merged_counter = merge_results();
        if (context_length == 1) {
                uint16_t ctx = static_cast<uint16_t>(context.cast<int64_t>());
                auto it = merged_counter->single_counts.find(ctx);
                if (it != merged_counter->single_counts.end()) {
                    for (const auto& [token, count] : it->second) {
                        result[py::int_(token)] = count;
                    }
                }
            } else {
                if (!py::isinstance<py::tuple>(context) || py::len(context) != 2) {
                    throw std::runtime_error("For context_length=2, context must be a tuple of two integers");
                }
                auto tuple = context.cast<py::tuple>();
                std::pair<uint16_t, uint16_t> ctx(
                    static_cast<uint16_t>(tuple[0].cast<int64_t>()),
                    static_cast<uint16_t>(tuple[1].cast<int64_t>())
                );
                auto it = merged_counter->pair_counts.find(ctx);
                if (it != merged_counter->pair_counts.end()) {
                    for (const auto& [token, count] : it->second) {
                        result[py::int_(token)] = count;
                    }
                }
            }
        } else {
            throw std::runtime_error("Must merge results before getting distribution");
        }
        
        return result;
    }
};

PYBIND11_MODULE(token_counter, m) {
    py::class_<ThreadCounter>(m, "ThreadCounter")
        .def(py::init<uint16_t>())
        .def_readonly("single_counts", &ThreadCounter::single_counts)
        .def_readonly("pair_counts", &ThreadCounter::pair_counts)
        .def_readonly("context_length", &ThreadCounter::context_length);

    py::class_<DistributionManager>(m, "DistributionManager")
        .def(py::init<uint16_t, size_t>())
        .def("process_batch", &DistributionManager::process_batch)
        .def("get_distribution", &DistributionManager::get_distribution)
        .def("merge_and_reset", &DistributionManager::merge_and_reset);
}