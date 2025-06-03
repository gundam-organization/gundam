#include "Kriged.h"

Kriged::Kriged(const std::vector<double>* table_,
                std::vector<int> index_, std::vector<double> weight_,
                const std::string& options_)
    : _table_{table_} {
    _weights_.reserve(index_.size());
    for (int i=0; i<index_.size(); ++i) {
        _weights_.emplace_back(index_[i],weight_[i]);
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
