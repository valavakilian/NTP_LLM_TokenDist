#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <stdint.h>
#include <fstream>
#include <iostream>

namespace py = pybind11;

struct PairHash {
    std::size_t operator()(const std::pair<uint16_t, uint16_t>& p) const {
        return (static_cast<size_t>(p.first) << 16) | p.second;
    }
};

class TokenDistributionCounter {
private:
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, uint32_t>> single_context_counts;
    std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, uint32_t>, PairHash> pair_context_counts;
    uint16_t context_length;

    void write_map_to_file(std::ofstream& out, const std::unordered_map<uint16_t, std::unordered_map<uint16_t, uint32_t>>& map) {
        size_t size = map.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        for (const auto& [ctx, dist] : map) {
            out.write(reinterpret_cast<const char*>(&ctx), sizeof(ctx));
            
            size_t dist_size = dist.size();
            out.write(reinterpret_cast<const char*>(&dist_size), sizeof(dist_size));
            
            for (const auto& [token, count] : dist) {
                out.write(reinterpret_cast<const char*>(&token), sizeof(token));
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }
    }

    void write_pair_map_to_file(std::ofstream& out, const std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, uint32_t>, PairHash>& map) {
        size_t size = map.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        for (const auto& [ctx_pair, dist] : map) {
            out.write(reinterpret_cast<const char*>(&ctx_pair.first), sizeof(ctx_pair.first));
            out.write(reinterpret_cast<const char*>(&ctx_pair.second), sizeof(ctx_pair.second));
            
            size_t dist_size = dist.size();
            out.write(reinterpret_cast<const char*>(&dist_size), sizeof(dist_size));
            
            for (const auto& [token, count] : dist) {
                out.write(reinterpret_cast<const char*>(&token), sizeof(token));
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            }
        }
    }

    void read_map_from_file(std::ifstream& in, std::unordered_map<uint16_t, std::unordered_map<uint16_t, uint32_t>>& map) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        for (size_t i = 0; i < size; i++) {
            uint16_t ctx;
            in.read(reinterpret_cast<char*>(&ctx), sizeof(ctx));
            
            size_t dist_size;
            in.read(reinterpret_cast<char*>(&dist_size), sizeof(dist_size));
            
            auto& dist = map[ctx];
            for (size_t j = 0; j < dist_size; j++) {
                uint16_t token;
                uint32_t count;
                in.read(reinterpret_cast<char*>(&token), sizeof(token));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                dist[token] = count;
            }
        }
    }

    void read_pair_map_from_file(std::ifstream& in, std::unordered_map<std::pair<uint16_t, uint16_t>, std::unordered_map<uint16_t, uint32_t>, PairHash>& map) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        for (size_t i = 0; i < size; i++) {
            uint16_t first, second;
            in.read(reinterpret_cast<char*>(&first), sizeof(first));
            in.read(reinterpret_cast<char*>(&second), sizeof(second));
            
            size_t dist_size;
            in.read(reinterpret_cast<char*>(&dist_size), sizeof(dist_size));
            
            auto& dist = map[std::make_pair(first, second)];
            for (size_t j = 0; j < dist_size; j++) {
                uint16_t token;
                uint32_t count;
                in.read(reinterpret_cast<char*>(&token), sizeof(token));
                in.read(reinterpret_cast<char*>(&count), sizeof(count));
                dist[token] = count;
            }
        }
    }

public:
    // Constructor for new instance
    TokenDistributionCounter(uint16_t ctx_len) : context_length(ctx_len) {
        if (ctx_len != 1 && ctx_len != 2) {
            throw std::runtime_error("Context length must be either 1 or 2");
        }
    }

    // Constructor for loading from file
    TokenDistributionCounter(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }

        // Read context length
        file.read(reinterpret_cast<char*>(&context_length), sizeof(context_length));

        if (context_length == 1) {
            read_map_from_file(file, single_context_counts);
        } else if (context_length == 2) {
            read_pair_map_from_file(file, pair_context_counts);
        } else {
            throw std::runtime_error("Invalid context length in file");
        }

        file.close();
    }

    void save_to_file(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        // Write context length
        file.write(reinterpret_cast<const char*>(&context_length), sizeof(context_length));

        if (context_length == 1) {
            write_map_to_file(file, single_context_counts);
        } else {
            write_pair_map_to_file(file, pair_context_counts);
        }

        file.close();
    }

    struct InsertResult {
        double execution_time_ms;
    };

    InsertResult insert(py::array_t<int64_t, py::array::c_style> batch, bool verbose = false) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
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

        for (size_t i = 0; i < batch_size; i++) {
            const int64_t* seq = data + i * seq_length;
            
            if (context_length == 1) {
                uint16_t context = static_cast<uint16_t>(seq[0]);
                uint16_t next_token = static_cast<uint16_t>(seq[1]);
                single_context_counts[context][next_token]++;
            } else {
                std::pair<uint16_t, uint16_t> context(
                    static_cast<uint16_t>(seq[0]),
                    static_cast<uint16_t>(seq[1])
                );
                uint16_t next_token = static_cast<uint16_t>(seq[2]);
                pair_context_counts[context][next_token]++;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        return InsertResult{static_cast<double>(duration.count())};
    }

    py::dict get_distribution(py::object context) {
        py::dict result;

        if (context_length == 1) {
            if (py::isinstance<py::int_>(context)) {
                uint16_t ctx = static_cast<uint16_t>(context.cast<int64_t>());
                auto it = single_context_counts.find(ctx);
                if (it != single_context_counts.end()) {
                    for (const auto& [token, count] : it->second) {
                        result[py::int_(token)] = count;
                    }
                }
            } else {
                throw std::runtime_error("For context_length=1, context must be an integer");
            }
        } else {
            if (py::isinstance<py::tuple>(context) && py::len(context) == 2) {
                auto tuple = context.cast<py::tuple>();
                std::pair<uint16_t, uint16_t> ctx(
                    static_cast<uint16_t>(tuple[0].cast<int64_t>()),
                    static_cast<uint16_t>(tuple[1].cast<int64_t>())
                );
                auto it = pair_context_counts.find(ctx);
                if (it != pair_context_counts.end()) {
                    for (const auto& [token, count] : it->second) {
                        result[py::int_(token)] = count;
                    }
                }
            } else {
                throw std::runtime_error("For context_length=2, context must be a tuple of two integers");
            }
        }

        return result;
    }

    size_t get_num_contexts() const {
        return (context_length == 1) ? 
            single_context_counts.size() : 
            pair_context_counts.size();
    }

    size_t get_memory_usage() {
        size_t total_memory = 0;
        
        if (context_length == 1) {
            // Count memory for single context map
            for (const auto& [context, distribution] : single_context_counts) {
                // Memory for the context key
                total_memory += sizeof(uint16_t);
                // Memory for each distribution entry
                total_memory += distribution.size() * (sizeof(uint16_t) + sizeof(uint32_t));
                // Approximate overhead for unordered_map
                total_memory += sizeof(void*) * distribution.bucket_count();
            }
            // Approximate overhead for outer unordered_map
            total_memory += sizeof(void*) * single_context_counts.bucket_count();
        } else {
            // Count memory for pair context map
            for (const auto& [context_pair, distribution] : pair_context_counts) {
                // Memory for the context pair key
                total_memory += sizeof(uint16_t) * 2;
                // Memory for each distribution entry
                total_memory += distribution.size() * (sizeof(uint16_t) + sizeof(uint32_t));
                // Approximate overhead for unordered_map
                total_memory += sizeof(void*) * distribution.bucket_count();
            }
            // Approximate overhead for outer unordered_map
            total_memory += sizeof(void*) * pair_context_counts.bucket_count();
        }
        
        return total_memory;
    }
};

PYBIND11_MODULE(root_context_dict, m) {
    py::class_<TokenDistributionCounter::InsertResult>(m, "InsertResult")
        .def_readonly("execution_time_ms", &TokenDistributionCounter::InsertResult::execution_time_ms);

    py::class_<TokenDistributionCounter>(m, "TokenDistributionCounter")
        .def(py::init<uint16_t>())
        .def(py::init<const std::string&>())
        .def("insert", &TokenDistributionCounter::insert)
        .def("get_distribution", &TokenDistributionCounter::get_distribution)
        .def("get_num_contexts", &TokenDistributionCounter::get_num_contexts)
        .def("save_to_file", &TokenDistributionCounter::save_to_file)
        .def("get_memory_usage", &TokenDistributionCounter::get_memory_usage);
}