#ifndef GUNDAM_EXTERNAL_WEIGHT_BUFFER_H
#define GUNDAM_EXTERNAL_WEIGHT_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <memory>

class ExternalWeightWorker;

// A stable source identity shared by the producer and all its dispatchers.
// The aliasing shared_ptr keeps the mapped storage alive even after the worker.
class ExternalWeightBuffer {
public:
  [[nodiscard]] const double* data() const { return _values_.get(); }
  [[nodiscard]] std::size_t size() const { return _size_; }
  [[nodiscard]] std::uint64_t getGeneration() const { return _generation_; }

private:
  friend class ExternalWeightWorker;
  std::shared_ptr<double> _values_{};
  std::size_t _size_{0};
  std::uint64_t _generation_{0};
};

#endif // GUNDAM_EXTERNAL_WEIGHT_BUFFER_H
