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
        RangeSet rs;
        for (unsigned long long i : indices) {
            rs.set.add(i);
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
            ranges.emplace_back(interval.lower(), interval.upper());
        }
        return ranges;
    }

    std::vector<unsigned long long> to_indices() const {
        std::vector<unsigned long long> indices;
        for (const auto& interval : set) {
            for (unsigned long long i = interval.lower(); ; ++i) {
                indices.push_back(i);
                if (i == interval.upper()) break; // handle overflow for max ull
            }
        }
        return indices;
    }

    bool is_empty() const {
        return set.empty();
    }

    std::string repr() const {
        std::stringstream ss;
        ss << "RangeSet({";
        bool first = true;
        for (const auto& interval : set) {
            if (!first) {
                ss << ", ";
            }
            ss << "[" << interval.lower() << ", " << interval.upper() << "]";
            first = false;
        }
        ss << "})";
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
