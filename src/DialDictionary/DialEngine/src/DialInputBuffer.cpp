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


const Parameter& DialInputBuffer::getParameter( int i) const {
    return _parSetRef_->at(_inputParameterIndicesList_[i].first).getParameterList().at(_inputParameterIndicesList_[i].second);
}
const ParameterSet& DialInputBuffer::getParameterSet( int i) const {
  return _parSetRef_->at(_inputParameterIndicesList_[i].first);
}

void DialInputBuffer::updateBuffer(){
  LogThrowIf(_parSetRef_ == nullptr, "parSetRef is not set.");

  this->setIsMasked( false );
  double* bufferPtr{_buffer_.data()};
  double buffer;

  // will change if at least one parameter is updated
  _isDialUpdateRequested_ = false;

  for( auto& parIndices : _inputParameterIndicesList_ ){

    if( (*_parSetRef_)[parIndices.first].isMaskedForPropagation() ){
      // in that case, the DialInterface will always return 1.
      this->setIsMasked( true );
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

//  if( not _isDialUpdateRequested_ ){
//    LogDebug << "CACHED: " << this->getSummary() << std::endl;
//  }

  _currentHash_ = generateHash();
}
void DialInputBuffer::addParameterIndices(const std::pair<size_t, size_t>& indices_){
  _inputParameterIndicesList_.emplace_back(indices_);
  _buffer_.emplace_back(std::nan("unset"));
}
void DialInputBuffer::addMirrorBounds(const std::pair<double, double>& lowEdgeAndRange_){
  const int p = _parameterMirrorBounds_.size();
  // Overriding the const to allow the mirroring information to be stored
  Parameter& par = const_cast<Parameter&>(getParameter(p));
  par.setMinMirror(lowEdgeAndRange_.first);
  par.setMaxMirror(lowEdgeAndRange_.first + lowEdgeAndRange_.second);
  _parameterMirrorBounds_.emplace_back(lowEdgeAndRange_);
}
const std::pair<double,double>&
DialInputBuffer::getMirrorBounds(int i) const {
    return _parameterMirrorBounds_.at(i);
}
std::string DialInputBuffer::getSummary() const{
  std::stringstream ss;
  ss << GenericToolbox::toString(_inputParameterIndicesList_, [&](const std::pair<size_t, size_t>& idx_){
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
    ss << ", " << GenericToolbox::toString(_parameterMirrorBounds_, [&](const std::pair<double, double>& bounds_){
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
