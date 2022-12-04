//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInputBuffer.h"

#include "Logger.h"

#define USE_ZLIB 0

#if USE_ZLIB
#include "zlib.h"
#endif

LoggerInit([]{
  Logger::setUserHeaderStr("[DialInputBuffer]");
});


void DialInputBuffer::updateBuffer(const std::vector<FitParameterSet>& parSetList_){
  double* buffer{_buffer_.data()};
  for( auto& parIndices : _inputParameterIndicesList_ ){
    *buffer = parSetList_[parIndices.first].getParameterList()[parIndices.second].getParameterValue();
    buffer++;
  }
  _currentHash_ = generateHash();
}

void DialInputBuffer::addParameterIndices(const std::pair<size_t, size_t>& indices_){
  _inputParameterIndicesList_.emplace_back(indices_);
  _buffer_.emplace_back(std::nan("unset"));
}

std::string DialInputBuffer::getSummary() const{
  std::stringstream ss;
  ss << "Par indices: " << GenericToolbox::iterableToString(_inputParameterIndicesList_, [](const std::pair<size_t, size_t>& idx_){
    std::stringstream ss; ss << "{" << idx_.first << ", " << idx_.second << "}"; return ss.str();
  }, false);
  return ss.str();
}

size_t DialInputBuffer::getBufferSize() const{
  return _buffer_.size();
}
const double* DialInputBuffer::getBuffer() const{
  return _buffer_.data();
}
const uint32_t& DialInputBuffer::getCurrentHash() const{
  return _currentHash_;
}
const std::vector<std::pair<size_t, size_t>> &DialInputBuffer::getInputParameterIndicesList() const {
  return _inputParameterIndicesList_;
}

uint32_t DialInputBuffer::generateHash(){
#if USE_ZLIB
  uint32_t out = crc32(0L, Z_NULL, 0);
  double* inputPtr = _buffer_.data();
  while( inputPtr < _buffer_.data() + _buffer_.size() ){
    out = crc32( out, (const Bytef*) inputPtr, sizeof(double) );
    inputPtr++;
  }
  return out;
#endif
  return 0;
}

