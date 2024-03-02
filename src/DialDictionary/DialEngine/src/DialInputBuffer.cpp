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


void DialInputBuffer::initialise(){
  LogThrowIf(_parSetListPtr_ == nullptr, "ParameterSet list pointer not set.");

  // the size won't change anymore
  _inputParameterReferenceList_.shrink_to_fit();

  // this value shouldn't change anymore:
  _inputArraySize_ = int(_inputParameterReferenceList_.size());

  // set the buffer to the proper size
  _inputBuffer_.resize(_inputArraySize_, std::nan("unset"));

  // sanity checks
  for( int iInput = 0 ; iInput < _inputArraySize_ ; iInput++ ){
    LogThrowIf(_inputParameterReferenceList_[iInput].parSetIndex < 0,
               "Parameter set index invalid: " << _inputParameterReferenceList_[iInput].parSetIndex);
    LogThrowIf(_inputParameterReferenceList_[iInput].parSetIndex >= _parSetListPtr_->size(),
               "Parameter set index invalid: " << _inputParameterReferenceList_[iInput].parSetIndex);

    LogThrowIf(_inputParameterReferenceList_[iInput].parIndex < 0,
               "Parameter index invalid: " << _inputParameterReferenceList_[iInput].parIndex);
    LogThrowIf(_inputParameterReferenceList_[iInput].parIndex >= getParameterSet(iInput).getParameterList().size(),
               "Parameter index invalid: " << _inputParameterReferenceList_[iInput].parIndex);
  }

  _isInitialized_ = true;
}

void DialInputBuffer::updateBuffer(){
  this->setIsMasked( false );
  double tempBuffer;

  // will change if at least one parameter is updated
  _isDialUpdateRequested_ = false;

  for( int iInput = 0 ; iInput < _inputArraySize_ ; iInput++ ){

    if( getParameterSet(iInput).isMaskedForPropagation() ){
      // in that case, the DialInterface will always return 1.
      this->setIsMasked( true );
      return;
    }

    // grab the value of the parameter
    tempBuffer = getParameter(iInput).getParameterValue();

    // find the actual parameter value if mirroring is applied
    if( not std::isnan( getMirrorEdges(iInput).minValue ) ){
      tempBuffer = std::abs(std::fmod(
          tempBuffer - getMirrorEdges(iInput).minValue,
          2 * getMirrorEdges(iInput).range
      ));

      if( tempBuffer > getMirrorEdges(iInput).range ){
        // odd pattern  -> mirrored -> decreasing effective X while increasing parameter
        tempBuffer -= 2 * getMirrorEdges(iInput).range;
        tempBuffer = -tempBuffer;
      }

      // re-apply the offset
      tempBuffer += getMirrorEdges(iInput).minValue;
    }

    // has it been updated?
    if( _inputBuffer_[iInput] != tempBuffer ){
      _isDialUpdateRequested_ = true;
      _inputBuffer_[iInput] = tempBuffer;
    }

  }

  _currentHash_ = generateHash();
}
void DialInputBuffer::addParameterReference( const ParameterReference& parReference_){
  LogThrowIf(_isInitialized_, "Can't add parameter index while initialized.");
  _inputParameterReferenceList_.emplace_back(parReference_);
}
std::string DialInputBuffer::getSummary() const{
  std::stringstream ss;
  ss << GenericToolbox::toString(_inputParameterReferenceList_, [&]( const ParameterReference& parRef_){
    std::stringstream ss;
    if( _parSetListPtr_ != nullptr ){
      ss << parRef_.getParameter(_parSetListPtr_).getFullTitle() << "=";
      ss << parRef_.getParameter(_parSetListPtr_).getParameterValue();
    }
    else{
      ss << "{" << parRef_.parSetIndex << ", " << parRef_.parIndex << "}";
    }

    if( not std::isnan(parRef_.mirrorEdges.minValue) ){
      ss << ", mirrorLow=" << parRef_.mirrorEdges.minValue;
      ss << ", mirrorUp=" << parRef_.mirrorEdges.minValue + parRef_.mirrorEdges.range;
    }

    return ss.str();
  }, false);

  return ss.str();
}

#if USE_ZLIB
uint32_t DialInputBuffer::generateHash(){
  uint32_t out = crc32(0L, Z_NULL, 0);
  double* inputPtr = _inputBuffer_.data();
  while( inputPtr < _inputBuffer_.data() + _inputBuffer_.size() ){
    out = crc32( out, (const Bytef*) inputPtr, sizeof(double) );
    inputPtr++;
  }
  return out;
}
#endif
