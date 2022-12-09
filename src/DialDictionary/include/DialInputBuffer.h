//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINPUTBUFFER_H
#define GUNDAM_DIALINPUTBUFFER_H

#include "FitParameterSet.h"

#include "vector"
#include "utility"


class DialInputBuffer {

public:
  DialInputBuffer() = default;

  void setIsMasked(bool isMasked);
  void setUseParameterMirroring(bool useParameterMirroring);
  void setParSetRef(std::vector<FitParameterSet> *parSetRef);

  [[nodiscard]] bool isMasked() const;
  [[nodiscard]] bool isDialUpdateRequested() const;
  [[nodiscard]] size_t getBufferSize() const;
  [[nodiscard]] const double* getBuffer() const;
  [[nodiscard]] const uint32_t& getCurrentHash() const;
  [[nodiscard]] const std::vector<std::pair<size_t, size_t>> &getInputParameterIndicesList() const;

  void updateBuffer();
  void addParameterIndices(const std::pair<size_t, size_t>& indices_);
  void addMirrorBounds(const std::pair<double, double>& lowEdgeAndRange_);

  [[nodiscard]] std::string getSummary() const;

protected:
  uint32_t generateHash();

private:
  bool _isMasked_{false};
  bool _isDialUpdateRequested_{true};
  bool _useParameterMirroring_{false};
  std::vector<std::pair<double, double>> _parameterMirrorBounds_{}; // first = lowBound, second = range
  std::vector<double> _buffer_;
  std::vector<std::pair<size_t, size_t>> _inputParameterIndicesList_{};
  std::vector<FitParameterSet>* _parSetRef_{nullptr};

  // for cache
  uint32_t _currentHash_{0};

};


#endif //GUNDAM_DIALINPUTBUFFER_H
