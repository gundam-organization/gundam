//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINPUTBUFFER_H
#define GUNDAM_DIALINPUTBUFFER_H

#include "FitParameter.h"

#include "zlib.h"

#include "vector"


class DialInputBuffer {

public:
  DialInputBuffer() = default;

  [[nodiscard]] size_t getBufferSize() const;
  [[nodiscard]] const double* getBuffer() const;
  [[nodiscard]] const uint32_t& getCurrentHash() const;

  void updateBuffer();
  void addInputParRef(const FitParameter* par_);

protected:
  uint32_t generateHash();

private:
  std::vector<double> _buffer_;
  std::vector<const FitParameter*> _inputParRefList_;

  // for cache
  uint32_t _currentHash_{0};

};


#endif //GUNDAM_DIALINPUTBUFFER_H
