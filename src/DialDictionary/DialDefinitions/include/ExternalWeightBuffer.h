#ifndef GUNDAM_EXTERNAL_WEIGHT_BUFFER_H
#define GUNDAM_EXTERNAL_WEIGHT_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class ExternalWeightWorker;

// A stable source identity shared by the producer and all its dispatchers.
// The aliasing shared_ptr keeps the mapped storage alive even after the worker.
class ExternalWeightBuffer {
public:
  [[nodiscard]] const double* data() const { return _values_.get(); }
  [[nodiscard]] std::size_t size() const { return _size_; }
  [[nodiscard]] std::uint64_t getGeneration() const { return _generation_; }
  [[nodiscard]] bool supportsGradient() const { return _supportsGradient_; }
  // Requested during analytic-gradient validation, before propagation workers.
  void requestGradient() const { _gradientRequested_ = true; }
  [[nodiscard]] bool isGradientRequested() const { return _gradientRequested_; }
  [[nodiscard]] bool hasCurrentGradient() const {
    return _gradientValid_ and _gradientGeneration_ == _generation_;
  }
  [[nodiscard]] std::size_t getGradientInputCount() const { return _gradientInputCount_; }
  [[nodiscard]] const double* gradientData() const { return _gradients_.get(); }
  [[nodiscard]] const std::vector<double>& getInputValues() const { return _inputValues_; }

private:
  friend class ExternalWeightWorker;
  std::shared_ptr<double> _values_{};
  std::size_t _size_{0};
  std::uint64_t _generation_{0};
  std::shared_ptr<double> _gradients_{};
  std::size_t _gradientInputCount_{0};
  std::uint64_t _gradientGeneration_{0};
  bool _supportsGradient_{false};
  bool _gradientValid_{false};
  mutable bool _gradientRequested_{false};
  std::vector<double> _inputValues_;
};

#endif // GUNDAM_EXTERNAL_WEIGHT_BUFFER_H
