#include "Kriged.h"

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

double Kriged::evalResponse(const DialInputBuffer& input_) const {
    const std::vector<double>& tbl = *_table_;
    double v{0.0};
    for (const std::pair<int,float>& w: _weights_) {
        v += w.second*tbl[w.first];
    }
    return v;
}
