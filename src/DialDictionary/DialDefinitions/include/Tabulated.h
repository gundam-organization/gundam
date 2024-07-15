#ifndef GUNDAM_TABULATED_H
#define GUNDAM_TABULATED_H

#include "DialBase.h"

// Implement a dial to look up a weight in a table and apply it.
class Tabulated : public DialBase {

public:
  Tabulated() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Tabulated>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Tabulated"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return _fraction_*_table_[_index_] + (1.0-_fraction_)*_table_[_index_+1]; }

  void buildDial(const double* table_, int index_, double fraction_,
                 const std::string& options_="") {
    _table_ = table_;
    _index_ = index_;
    _fraction_ = fraction_;
  }

private:
  const double* _table_{nullptr};
  int _index_{0};
  float _fraction_{0.0};
};
#endif //GUNDAM_TABULATED_H
