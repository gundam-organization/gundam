//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInputBuffer.h"

#include "Logger.h"

#if USE_ZLIB
#include "zlib.h"
#endif

LoggerInit([]{
  Logger::setUserHeaderStr("[DialInputBuffer]");
});


void DialInputBuffer::setIsMasked(bool isMasked) {
  _isMasked_ = isMasked;
}
void DialInputBuffer::setUseParameterMirroring(bool useParameterMirroring) {
  _useParameterMirroring_ = useParameterMirroring;
}
void DialInputBuffer::setParSetRef(std::vector<FitParameterSet> *parSetRef) {
  _parSetRef_ = parSetRef;
}

bool DialInputBuffer::isMasked() const {
  return _isMasked_;
}
bool DialInputBuffer::isDialUpdateRequested() const {
  return _isDialUpdateRequested_;
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

void DialInputBuffer::updateBuffer(){
  LogThrowIf(_parSetRef_ == nullptr, "parSetRef is not set.");

  _isMasked_ = false;
  double* bufferPtr{_buffer_.data()};
  double buffer;

  // will change if at least one parameter is updated
  _isDialUpdateRequested_ = false;

  for( auto& parIndices : _inputParameterIndicesList_ ){

    if( (*_parSetRef_)[parIndices.first].isMaskedForPropagation() ){
      _isMasked_ = true;
      return;
    }

    buffer = (*_parSetRef_)[parIndices.first].getParameterList()[parIndices.second].getParameterValue();
    if( _useParameterMirroring_ ){
      buffer = std::abs(std::fmod(
          buffer - _parameterMirrorBounds_[std::distance(_buffer_.data(), bufferPtr)].first,
          2 * _parameterMirrorBounds_[std::distance(_buffer_.data(), bufferPtr)].second
      ));

      if(buffer > _parameterMirrorBounds_[std::distance(_buffer_.data(), bufferPtr)].second ){
        // odd pattern  -> mirrored -> decreasing effective X while increasing parameter
        buffer -= 2 * _parameterMirrorBounds_[std::distance(_buffer_.data(), bufferPtr)].second;
        buffer = -buffer;
      }

      // re-apply the offset
      buffer += _parameterMirrorBounds_[std::distance(_buffer_.data(), bufferPtr)].first;
    }
    LogThrowIf(std::isnan(buffer), "NaN while evaluating input buffer of " << (*_parSetRef_)[parIndices.first].getParameterList()[parIndices.second].getTitle());

    if( *bufferPtr != buffer ){
//      LogTrace << "UPT: " << this->getSummary() << ": " << *bufferPtr << " -> " << buffer << std::endl;
      _isDialUpdateRequested_ = true;
    }

    *bufferPtr = buffer;
    bufferPtr++;
  }

  if( not _isDialUpdateRequested_ ){
//    LogDebug << "CACHED: " << this->getSummary() << std::endl;
  }

  _currentHash_ = generateHash();
}
void DialInputBuffer::addParameterIndices(const std::pair<size_t, size_t>& indices_){
  _inputParameterIndicesList_.emplace_back(indices_);
  _buffer_.emplace_back(std::nan("unset"));
}
void DialInputBuffer::addMirrorBounds(const std::pair<double, double>& lowEdgeAndRange_){
  _parameterMirrorBounds_.emplace_back(lowEdgeAndRange_);
}
std::string DialInputBuffer::getSummary() const{
  std::stringstream ss;
  ss << GenericToolbox::iterableToString(_inputParameterIndicesList_, [&](const std::pair<size_t, size_t>& idx_){
    std::stringstream ss;
    if( _parSetRef_ != nullptr ){
      ss << (*_parSetRef_)[idx_.first].getParameterList()[idx_.second].getFullTitle() << "=";
      ss << (*_parSetRef_)[idx_.first].getParameterList()[idx_.second].getParameterValue();
    }
    else{
      ss << "{" << idx_.first << ", " << idx_.second << "}";
    }
    return ss.str();
  }, false);
  if( _useParameterMirroring_ ){
    ss << ", " << GenericToolbox::iterableToString(_parameterMirrorBounds_, [&](const std::pair<double, double>& bounds_){
      std::stringstream ss;
      ss << "mirrorLow=" << bounds_.first << ", mirrorUp=" << bounds_.first + bounds_.second;
      return ss.str();
    }, false);
  }
  return ss.str();
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

