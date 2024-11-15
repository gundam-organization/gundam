//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInputBuffer.h"

#include "Logger.h"
#include "GundamBacktrace.h"


void DialInputBuffer::invalidateBuffers(){
  // invalidate buffer
  for( auto& buf : _inputBuffer_ ){ buf = std::nan("unset"); }
}

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

void DialInputBuffer::update(){
  // by default consider we have to update
  _isDialUpdateRequested_ = true;

  // look for the parameter values
  double tempBuffer;
  _isDialUpdateRequested_ = false; // if ANY is different, request the update
  for( auto& inputRef : _inputParameterReferenceList_ ){
    // grab the value of the parameter
    tempBuffer = inputRef.getParameter(_parSetListPtr_).getParameterValue();

    // find the actual parameter value if mirroring is applied
    if( not std::isnan( inputRef.mirrorEdges.minValue ) ){
      tempBuffer = std::abs(std::fmod(
          tempBuffer - inputRef.mirrorEdges.minValue,
          2 * inputRef.mirrorEdges.range
      ));

      if( tempBuffer > inputRef.mirrorEdges.range ){
        // odd pattern  -> mirrored -> decreasing effective X while increasing parameter
        tempBuffer -= 2 * inputRef.mirrorEdges.range;
        tempBuffer = -tempBuffer;
      }

      // re-apply the offset
      tempBuffer += inputRef.mirrorEdges.minValue;
    }
    if( std::isnan(tempBuffer) ){
        // LogThrowIf is broken, but OK for real error traps, but this is
        // checking user input it's critical that the error message is
        // properly formated so print an error, a backtrace, and then exit.
        LogError << "NaN while evaluating input buffer of "
                 << inputRef.getParameter(_parSetListPtr_).getTitle()
                 << std::endl;
        LogError << GundamUtils::Backtrace << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // has it been updated?
    if( _inputBuffer_[inputRef.bufferIndex] != tempBuffer ){
      _isDialUpdateRequested_ = true;
      _inputBuffer_[inputRef.bufferIndex] = tempBuffer;
    }
  }
}
void DialInputBuffer::addParameterReference( const ParameterReference& parReference_){
  LogThrowIf(_isInitialized_, "Can't add parameter index while initialized.");
  _inputParameterReferenceList_.emplace_back(parReference_);
  auto& parRef = _inputParameterReferenceList_.back();
  parRef.bufferIndex = int(_inputParameterReferenceList_.size())-1;
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
