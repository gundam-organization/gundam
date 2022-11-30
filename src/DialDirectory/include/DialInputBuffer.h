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

  void setInputParRefList(const std::vector<FitParameter *> &inputParRefList);

  void udpateBuffer();

  [[nodiscard]] size_t getBufferSize() const;
  [[nodiscard]] const double* getBuffer() const;
  [[nodiscard]] const uint32_t& getCurrentHash() const;

protected:
  uint32_t generateHash();


private:
  std::vector<double> _buffer_;
  std::vector<FitParameter*> _inputParRefList_;

  // for cache
  uint32_t _currentHash_{0};


};


#endif //GUNDAM_DIALINPUTBUFFER_H
