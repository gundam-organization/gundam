//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIAL_INPUT_BUFFER_H
#define GUNDAM_DIAL_INPUT_BUFFER_H

#include "ParameterSet.h"

#include <vector>
#include <utility>


class DialInputBuffer {

  /// DialInputBuffer is in charge of keeping the "input" parameter values
  /// for the evaluation of a collection of dials.

  /// The buffer is updated in single thread before the evaluation of
  /// individual dials is done. This means DialInputBuffer is only meant
  /// to handle Parameters. Not Event variables.

public:
  /// definition of data structs
  struct ParameterReference{
    int parSetIndex{-1};
    int parIndex{-1};
    int bufferIndex{-1};

    // it costs less memory to have only one vector with two potentially useless double (2*8bytes)
    // than an empty vector (24 bytes)
    struct MirrorEdges{
      double minValue{std::nan("unset")};
      double range{std::nan("unset")};
    };
    MirrorEdges mirrorEdges{};

    [[nodiscard]] const ParameterSet& getParameterSet(std::vector<ParameterSet>* parSetListPtr_) const {
      return (*parSetListPtr_)[parSetIndex];
    }
    [[nodiscard]] const Parameter& getParameter(std::vector<ParameterSet>* parSetListPtr_) const {
      return this->getParameterSet(parSetListPtr_).getParameterList()[parIndex];
    }
  };

public:
  DialInputBuffer() = default;

  /// Tell the input buffer about the global vector of fit parameter sets.
  /// This is required, so it must be set before the DialInputBuffer can
  /// be used.
  void setParSetRef(std::vector<ParameterSet> *parSetRef_){ _parSetListPtr_ = parSetRef_; }

  // const getters
  [[nodiscard]] bool isDialUpdateRequested() const{ return _isDialUpdateRequested_; }
  [[nodiscard]] bool* isDialUpdateRequestedPtr() { return &_isDialUpdateRequested_; }
  [[nodiscard]] int getBufferSize() const{ return _inputArraySize_; }
  [[nodiscard]] size_t getInputSize() const{ return _inputParameterReferenceList_.size(); }
  [[nodiscard]] const std::vector<double>& getInputBuffer() const { return _inputBuffer_; }
  [[nodiscard]] const std::vector<ParameterReference> &getInputParameterIndicesList() const{ return _inputParameterReferenceList_; }

  // mutable getters

  /// Function that allow to tweak the buffer from the inside. Used for
  /// individual spline evaluation.
  std::vector<double>& getInputBuffer(){ return _inputBuffer_; }
  std::vector<ParameterReference> &getInputParameterIndicesList(){ return _inputParameterReferenceList_; }

  // core
  void invalidateBuffers();

  /// Make sure everything is ready for use
  void initialise();

  /// Update the buffer to flag if any parameter has changed, and apply any
  /// mirroring to the parameter values.
  void update();

  // nested getters
  [[nodiscard]] const ParameterSet& getParameterSet(int iInput_) const{ return _inputParameterReferenceList_[iInput_].getParameterSet(_parSetListPtr_); }
  [[nodiscard]] const Parameter& getParameter(int iInput_) const { return _inputParameterReferenceList_[iInput_].getParameter(_parSetListPtr_); }
  [[nodiscard]] const ParameterReference::MirrorEdges& getMirrorEdges(int iInput_) const{ return _inputParameterReferenceList_[iInput_].mirrorEdges; }

  /// Push the index of a ParameterSet and Parameter in the set onto the
  /// vector of parameters.  This must be used in the order that the dial will
  /// be expecting the parameters (e.g. for a 2D parameter,
  void addParameterReference( const ParameterReference& parReference_);

  /// Simple printout function for debug info on error
  [[nodiscard]] std::string getSummary() const;

private:
  /// Flag if the member can be still edited.
  bool _isInitialized_{false};

  /// Flag for if the dials need to be recalculated.  Dials don't need to
  /// be recalculated if the parameter values have not changed.
  bool _isDialUpdateRequested_{true};

  /// How many inputs are handled
  int _inputArraySize_{0};

  /// A pointer to the "global" vector of parameter sets.
  /// This provides the connection to the parameters.
  std::vector<ParameterSet>* _parSetListPtr_{nullptr};

  /// A calculated buffer of parameter values to be handed to the dial
  /// evaluation.  This is the value after transformations like "mirroring"
  /// have been applied.  For a 1D dial, this has a size of 1.  For a
  /// 2D this has a size of 2, and so on.
  std::vector<double> _inputBuffer_;

  /// There is one entry per parameter used by the dial (e.g. a 1D parameter
  /// has one entry)
  std::vector<ParameterReference> _inputParameterReferenceList_{};

};


#endif //GUNDAM_DIAL_INPUT_BUFFER_H
