#ifndef GUNDAM_TABLEGRADIENTBUFFER_H
#define GUNDAM_TABLEGRADIENTBUFFER_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Shared by a table factory and its event dials. Publication happens before
// event workers start; event workers only read the completed buffer.
class TableGradientBuffer {
public:
  using UpdateFunction = int (*)(const char*, double*, int, const double*, int);

  explicit TableGradientBuffer(UpdateFunction gradientFunction_ = nullptr)
    : _gradientFunction_(gradientFunction_) {}

  [[nodiscard]] bool requestGradient() const {
    if( _gradientFunction_ == nullptr ){ return false; }
    _requested_ = true;
    return true;
  }

  void update(std::vector<double>& table_, const std::vector<double>& inputs_,
              UpdateFunction updateFunction_, const std::string& name_) {
    _valid_ = false;
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if( _requested_ ){ std::fill(table_.begin(), table_.end(), invalid); }
    if( updateFunction_(name_.c_str(), table_.data(), int(table_.size()),
                        inputs_.data(), int(inputs_.size())) != 0 ){
      throw std::runtime_error("Table " + name_ + ": updateFunction failed");
    }
    if( not _requested_ ){ return; }
    if( std::any_of(table_.begin(), table_.end(), [](double value_){ return not std::isfinite(value_); }) ){
      throw std::runtime_error("Table " + name_ + ": incomplete or non-finite values");
    }
    if( not inputs_.empty() and table_.size() > _derivatives_.max_size() / inputs_.size() ){
      throw std::runtime_error("Table " + name_ + ": gradient dimensions overflow");
    }
    _derivatives_.assign(table_.size() * inputs_.size(), invalid);
    if( _gradientFunction_(name_.c_str(), _derivatives_.data(), int(table_.size()),
                          inputs_.data(), int(inputs_.size())) != 0 ){
      throw std::runtime_error("Table " + name_ + ": gradientFunction failed");
    }
    if( std::any_of(_derivatives_.begin(), _derivatives_.end(), [](double value_){ return not std::isfinite(value_); }) ){
      throw std::runtime_error("Table " + name_ + ": incomplete or non-finite derivatives");
    }
    _inputs_ = inputs_;
    _entryCount_ = table_.size();
    _valid_ = true;
    _requested_ = false;
  }

  void validate(const std::vector<double>& inputs_, std::size_t entryCount_) const {
    if( not _valid_ or inputs_ != _inputs_ or entryCount_ != _entryCount_ ){
      throw std::runtime_error("Table gradient is unavailable or does not match the current inputs/table");
    }
  }

  [[nodiscard]] double getDerivative(std::size_t entry_, std::size_t input_) const {
    if( not _valid_ or entry_ >= _entryCount_ or input_ >= _inputs_.size() ){
      throw std::runtime_error("Invalid table gradient entry/input");
    }
    return _derivatives_[entry_ * _inputs_.size() + input_];
  }

private:
  UpdateFunction _gradientFunction_{nullptr};
  mutable bool _requested_{false};
  bool _valid_{false};
  std::size_t _entryCount_{0};
  std::vector<double> _inputs_;
  std::vector<double> _derivatives_;
};

#endif // GUNDAM_TABLEGRADIENTBUFFER_H
