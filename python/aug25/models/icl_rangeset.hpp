#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <boost/icl/interval_set.hpp>
#include "stats.hpp"

class RangeSet {
public:
    RangeSet() = default;

    static RangeSet empty() {
        return RangeSet();
    }

    static RangeSet from_indices(const std::vector<unsigned long long>& indices) {
        if (indices.empty()) {
            return RangeSet::empty();
        }
        // Create a mutable copy to sort and unique
        std::vector<unsigned long long> sorted_indices = indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        sorted_indices.erase(std::unique(sorted_indices.begin(), sorted_indices.end()), sorted_indices.end());

        RangeSet rs;
        auto it = sorted_indices.begin();
        while (it != sorted_indices.end()) {
            unsigned long long start = *it;
            unsigned long long end = start;
            auto next_it = std::next(it);
            while (next_it != sorted_indices.end() && *next_it == end + 1) {
                end = *next_it;
                ++next_it;
            }
            rs.set.add(boost::icl::discrete_interval<unsigned long long>::closed(start, end));
            it = next_it;
        }
        return rs;
    }

    static RangeSet from_singleton(unsigned long long index) {
        RangeSet rs;
        rs.set.add(index);
        return rs;
    }

    static RangeSet from_ranges(const std::vector<std::pair<unsigned long long, unsigned long long>>& ranges) {
        RangeSet rs;
        for (const auto& p : ranges) {
            rs.set.add(boost::icl::discrete_interval<unsigned long long>::closed(p.first, p.second));
        }
        return rs;
    }

    RangeSet union_with(const RangeSet& other) const {
        RangeSet result = *this;
        Stats::get().inc("bitset.union.calls");
        result.set |= other.set;
        return result;
    }

    RangeSet intersection_with(const RangeSet& other) const {
        RangeSet result = *this;
        Stats::get().inc("bitset.intersection.calls");
        result.set &= other.set;
        return result;
    }

    RangeSet difference_with(const RangeSet& other) const {
        RangeSet result = *this;
        result.set -= other.set;
        return result;
    }

    bool contains(unsigned long long index) const {
        return boost::icl::contains(set, index);
    }

    size_t size() const {
        return boost::icl::cardinality(set);
    }

    std::vector<std::pair<unsigned long long, unsigned long long>> to_ranges() const {
        std::vector<std::pair<unsigned long long, unsigned long long>> ranges;
        for (const auto& interval : set) {
            // A right-open interval [lower, upper) represents the inclusive range [lower, upper-1].
            if (interval.lower() < interval.upper()) {
                ranges.emplace_back(interval.lower(), interval.upper() - 1);
            }
        }
        return ranges;
    }

    std::vector<unsigned long long> to_indices() const {
        std::vector<unsigned long long> indices;
        for (const auto& interval : set) {
            // A right-open interval [lower, upper) contains elements from lower up to (but not including) upper.
            unsigned long long upper = interval.upper();
            for (unsigned long long i = interval.lower(); i != upper; ++i)
                indices.push_back(i);
        }
        return indices;
    }

    bool is_empty() const {
        return set.empty();
    }

    std::string repr() const {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        for (const auto& interval : set) {
            if (!first) {
                ss << ", ";
            }
            // A right-open interval [lower, upper) represents the inclusive range [lower, upper-1].
            if (interval.lower() < interval.upper()) {
                ss << "(" << interval.lower() << ", " << (interval.upper() - 1) << ")";
            }
            first = false;
        }
        ss << "]";
        return ss.str();
    }

    bool operator==(const RangeSet& other) const {
        return set == other.set;
    }

    size_t hash() const {
        size_t seed = 0;
        // The hash needs to be order-independent of the intervals, but boost::icl::interval_set
        // stores them in a sorted, non-overlapping way, so simple iteration is fine.
        for (const auto& interval : set) {
            boost::hash_combine(seed, interval.lower());
            boost::hash_combine(seed, interval.upper());
        }
        return seed;
    }
private:
    boost::icl::interval_set<unsigned long long> set;
};
