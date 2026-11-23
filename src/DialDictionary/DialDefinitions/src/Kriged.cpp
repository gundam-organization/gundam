/// @file Kriged.cpp
/// @brief Implements Kriged: building the (index, weight) list and summing
///        the weighted table entries.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in Kriged.h. The
/// comments here describe how each method does its job.

#include "Kriged.h"

/// @par Implementation
/// Sizes `_weights_` to exactly `entries_` and copies the first `entries_`
/// indices and weights into it, narrowing each weight from double to float.
Kriged::Kriged(const std::vector<double>* table_,
               int entries_, std::vector<int> index_, std::vector<double> weight_,
               const std::string& options_)
    : _table_{table_} {
    _weights_.resize(entries_);
    for (int i=0; i<entries_; ++i) {
        _weights_[i].first = index_[i];
        _weights_[i].second = weight_[i];
    }
}

/// @par Implementation
/// Dereferences the table pointer once, then accumulates in double while
/// looping over the pairs.
double Kriged::evalResponse(const DialInputBuffer& input_) const {
    const std::vector<double>& tbl = *_table_;
    double v{0.0}; // running sum of weight * table value
    for (const std::pair<int,float>& w: _weights_) { // (table index, weight)
        v += w.second*tbl[w.first];
    }
    return v;
}
