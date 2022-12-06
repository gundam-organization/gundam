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

  [[nodiscard]] bool isMasked() const;
  [[nodiscard]] size_t getBufferSize() const;
  [[nodiscard]] const double* getBuffer() const;
  [[nodiscard]] const uint32_t& getCurrentHash() const;
  [[nodiscard]] const std::vector<std::pair<size_t, size_t>> &getInputParameterIndicesList() const;

  void updateBuffer(const std::vector<FitParameterSet>& parSetList_);
  void addParameterIndices(const std::pair<size_t, size_t>& indices_);
  void addMirrorBounds(const std::pair<double, double>& lowEdgeAndRange_);

  [[nodiscard]] std::string getSummary() const;

protected:
  uint32_t generateHash();

private:
  bool _isMasked_{false};
  bool _useParameterMirroring_{false};
  std::vector<std::pair<double, double>> _parameterMirrorBounds_{}; // first = lowBound, second = range
  std::vector<double> _buffer_;
  std::vector<std::pair<size_t, size_t>> _inputParameterIndicesList_{};

  // for cache
  uint32_t _currentHash_{0};

};


#endif //GUNDAM_DIALINPUTBUFFER_H
