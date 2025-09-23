#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>

class Stats {
public:
    static Stats& get() {
        static Stats instance;
        return instance;
    }

    void inc(const std::string& key, long long value = 1) {
        counters_[key] += value;
    }

    void start(const std::string& key) {
        timers_[key] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& key) {
        auto it = timers_.find(key);
        if (it != timers_.end()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - it->second).count();
            durations_[key] += duration;
            timers_.erase(it);
        }
    }

    void reset() {
        counters_.clear();
        durations_.clear();
        timers_.clear();
    }

    void report() {
        std::cout << "\n--- C++ Engine Stats Report ---\n";
        
        std::vector<std::string> counter_keys;
        for (const auto& pair : counters_) {
            counter_keys.push_back(pair.first);
        }
        std::sort(counter_keys.begin(), counter_keys.end());

        std::cout << "\nCounters:\n";
        for (const auto& key : counter_keys) {
            std::cout << "  " << std::left << std::setw(60) << key << ": " << counters_[key] << "\n";
        }

        std::vector<std::string> duration_keys;
        for (const auto& pair : durations_) {
            duration_keys.push_back(pair.first);
        }
        std::sort(duration_keys.begin(), duration_keys.end());

        std::cout << "\nTimers (ms):\n";
        for (const auto& key : duration_keys) {
            double ms = static_cast<double>(durations_[key]) / 1.0e6;
            std::cout << "  " << std::left << std::setw(60) << key << ": " << std::fixed << std::setprecision(3) << ms << "\n";
        }
        std::cout << "---------------------------------\n" << std::endl;
    }

private:
    Stats() = default;
    ~Stats() = default;
    Stats(const Stats&) = delete;
    Stats& operator=(const Stats&) = delete;

    std::unordered_map<std::string, long long> counters_;
    std::unordered_map<std::string, long long> durations_; // in nanoseconds
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
};
