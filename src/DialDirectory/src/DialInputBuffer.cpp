//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInputBuffer.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialInputBuffer]");
});


void DialInputBuffer::updateBuffer(){
  if(_buffer_.size() != _inputParRefList_.size() ){
    _buffer_.resize(_inputParRefList_.size(), std::nan("unset"));
  }
  double* buffer{_buffer_.data()};
  for( auto* par : _inputParRefList_ ){
    *buffer = par->getParameterValue();
    buffer++;
  }
  _currentHash_ = generateHash();
}
void DialInputBuffer::addInputParRef(const FitParameter* par_){
  _inputParRefList_.emplace_back(par_);
  _buffer_.emplace_back(std::nan("unset"));
}

size_t DialInputBuffer::getBufferSize() const{
  return _inputParRefList_.size();
}
const double* DialInputBuffer::getBuffer() const{
  return _buffer_.data();
}
const uint32_t& DialInputBuffer::getCurrentHash() const{
  return _currentHash_;
}

uint32_t DialInputBuffer::generateHash(){
  uint32_t out = crc32(0L, Z_NULL, 0);
  double* inputPtr = _buffer_.data();
  while( inputPtr < _buffer_.data() + _buffer_.size() ){
    out = crc32( out, (const Bytef*) inputPtr, sizeof(double) );
    inputPtr++;
  }
  return out;
}


