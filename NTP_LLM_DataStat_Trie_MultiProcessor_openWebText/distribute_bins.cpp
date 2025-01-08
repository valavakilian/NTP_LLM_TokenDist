#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace py = pybind11;

class ProgressBar {
    size_t total;
    size_t current;
    size_t last_percent;
    std::chrono::steady_clock::time_point start_time;
    std::string description;

public:
    ProgressBar(size_t total, std::string desc = "Progress") 
        : total(total), current(0), last_percent(0), description(desc) {
        start_time = std::chrono::steady_clock::now();
        std::cout << description << ": 0%" << std::flush;
    }

    void update(size_t n = 1) {
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
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        std::cout << "\n" << std::flush;
    }
};

py::tuple distribute_tuples(const py::dict& tuple_counts, size_t num_bins) {
    std::cout << "\nStarting distribution of " << py::len(tuple_counts) << " tuples...\n";
    
    // Convert to vector of pairs
    std::cout << "Converting to list...\n";
    std::vector<std::pair<py::tuple, size_t>> items;
    items.reserve(py::len(tuple_counts));
    
    ProgressBar progress(py::len(tuple_counts), "Converting tuples");
    for (const auto& item : tuple_counts) {
        items.push_back({item.first.cast<py::tuple>(), item.second.cast<size_t>()});
        progress.update();
    }
    progress.finish();
    
    // Sort and print top 10
    std::cout << "\nTop 10 most frequent token pairs:\n";
    std::vector<std::pair<py::tuple, size_t>> sorted_items = items;
    std::sort(sorted_items.begin(), sorted_items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
              
    for (size_t i = 0; i < std::min(size_t(10), sorted_items.size()); i++) {
        const auto& pair = sorted_items[i];
        // Convert tuple elements to string properly
        auto t = pair.first;
        std::cout << "Token pair (" << std::to_string(t[0].cast<long>()) 
                 << "," << std::to_string(t[1].cast<long>()) 
                 << "): appeared " << pair.second << " times\n";
    }
    
    // Sort all items
    std::cout << "\nSorting all items...\n";
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Initialize bins
    std::vector<std::vector<py::tuple>> bins(num_bins);
    std::vector<size_t> bin_sums(num_bins, 0);
    
    // Distribute items
    std::cout << "\nDistributing items across " << num_bins << " bins...\n";
    ProgressBar dist_progress(items.size(), "Distributing items");
    for (size_t i = 0; i < items.size(); i++) {
        size_t bin_idx = i % num_bins;
        bins[bin_idx].push_back(items[i].first);
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

PYBIND11_MODULE(distribute_bins_cpp, m) {
    m.doc() = "Tuple distribution implementation in C++";
    m.def("distribute_tuples", &distribute_tuples,
          py::arg("tuple_counts"),
          py::arg("num_bins"),
          "Distribute tuples into bins based on frequency");
}