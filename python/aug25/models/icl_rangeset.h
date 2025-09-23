#pragma once

#include <boost/icl/interval_set.hpp>
#include <boost/icl/interval.hpp>

#include <vector>
#include <string>
#include <utility>

class RangeSet {
public:
    using interval_type = boost::icl::discrete_interval<unsigned long long>;
    using set_type = boost::icl::interval_set<unsigned long long>;

    RangeSet() = default;

    static RangeSet empty();
    static RangeSet from_indices(const std::vector<unsigned long long>& indices);
    static RangeSet from_ranges(const std::vector<std::pair<unsigned long long, unsigned long long>>& ranges);

    bool contains(unsigned long long v) const;
    RangeSet union_with(const RangeSet& other) const;
    RangeSet intersection_with(const RangeSet& other) const;
    RangeSet difference_with(const RangeSet& other) const;
    bool is_empty() const;
    std::vector<std::pair<unsigned long long, unsigned long long>> to_ranges() const;
    std::vector<unsigned long long> to_indices() const;
    bool operator==(const RangeSet& other) const;
    std::string repr() const;

private:
    set_type m_set;
};
