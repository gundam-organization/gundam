//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINPUTBUFFER_H
#define GUNDAM_DIALINPUTBUFFER_H

#include "FitParameterSet.h"

#include <vector>
#include <utility>


class DialInputBuffer {

public:
  DialInputBuffer() = default;

  /// Set the IsMasked flag.  This flag is ignored internally, but
  /// can be set internally when a parameter set is masked.  The value
  /// of the flag is checked using isMasked()
  void setIsMasked(bool isMasked){ _isMasked_ = isMasked; }

  /// Flag that the parameters being used in this DialInputBuffer include
  /// mirroring.  If this is set, then mirroring bounds must be provided
  /// for all parameters in the DialInputBuffer.
  void setUseParameterMirroring(bool useParameterMirroring){ _useParameterMirroring_ = useParameterMirroring; }

  /// Tell the input buffer about the global vector of fit parameter sets.
  /// This is required, so it must be set before the DialInputBuffer can
  /// be used.
  void setParSetRef(std::vector<FitParameterSet> *parSetRef){ _parSetRef_ = parSetRef; }

  // Getters
  [[nodiscard]] bool isMasked() const{ return _isMasked_; }
  [[nodiscard]] bool isDialUpdateRequested() const{ return _isDialUpdateRequested_; }
  [[nodiscard]] bool useParameterMirroring() const{ return _useParameterMirroring_; }

  /// Just a wrapper around the vector<double>::size() method.
  [[nodiscard]] size_t getBufferSize() const{ return _buffer_.size(); }

  /// All I can say is "wow".  This is returning a raw pointer the underlying
  /// vector memory.  This is vector<double>::data(), so it returns an array
  /// of parameter values that should be passed to the dial.
  [[nodiscard]] const double* getBuffer() const{ return _buffer_.data(); }

  /// Get a pointer to the FitParameter for a DialInputBuffer entry.
  [[nodiscard]] const FitParameter& getFitParameter(int i=0) const;

  /// Get a pointer to the FitParameterSet for the DialInputBuffer entry.
  [[nodiscard]] const FitParameterSet& getFitParameterSet(int i=0) const;

  /// Get the reference to the hash for the current cache.
  [[nodiscard]] const uint32_t& getCurrentHash() const{ return _currentHash_; }
  [[nodiscard]] const std::vector<std::pair<size_t, size_t>> &getInputParameterIndicesList() const{ return _inputParameterIndicesList_; }

  /// Update the buffer to flag if any parameter has changed, and apply any
  /// mirroring to the parameter values.
  void updateBuffer();

  /// Push the index of a FitParameterSet and FitParameter in the set onto the
  /// vector of parameters.  This must be used in the order that the dial will
  /// be expecting the parameters (e.g. for a 2D parameter,
  void addParameterIndices(const std::pair<size_t, size_t>& indices_);

  /// Push the parameter mirror bounds on to the vector of parameters being
  /// used for the DialInputBuffer.  This must be done in the same order as
  /// the parameters are added by addParameterIndices().  If mirrored
  /// parameters are being used, there must be a mirror bound for every
  /// parameter.
  void addMirrorBounds(const std::pair<double, double>& lowEdgeAndRange_);

  /// Get the vector of mirror bounds (will be empty if the parameter is not
  /// mirrored) If mirrorred parameters have been provided, there must be a
  /// mirror for each parameter.  The first entry in the pair is the lower
  /// bound of the mirrored region, the second entry is the range of the
  /// mirror.  The valid region will be between first, and first+second.
  [[nodiscard]] const std::pair<double,double>& getMirrorBounds(int i) const;

  /// Simple printout function for debug info on error
  [[nodiscard]] std::string getSummary() const;

  /// Function that allow to tweak the buffer from the inside. Used for
  /// individual spline evaluation.
  std::vector<double>& getBufferVector() { return _buffer_; }

protected:
  uint32_t generateHash();

private:
  bool _isMasked_{false};

  /// Flag for if the dials need to be recalculated.  Dials don't need to
  /// be recalculated if the parameter values have not changed.
  bool _isDialUpdateRequested_{true};

  /// Flag if mirroring should be used.  If this is true, then the
  /// parameter mirror bounds must be filled.
  bool _useParameterMirroring_{false};

  /// Should be a struct with real names and documentation: For now,
  /// the pair is holding:
  ///    * first  -- The lower bound of the range to be mirrored.
  ///    * second -- The range to be mirrored (so the upper bound is
  ///         first+second)
  /// If parameter mirroring is used, there must be one entry per parameter
  /// used by the dial (e.g. a 1D parameter has one entry).
  std::vector<std::pair<double, double>> _parameterMirrorBounds_{};

  /// A calculated buffer of parameter values to be handed to the dial
  /// evaluation.  This is the value after transformations like "mirroring"
  /// have been applied.  For a 1D dial, this has a size of 1.  For a
  /// 2D this has a size of 2, and so on.
  std::vector<double> _buffer_;

  /// Should be a struct with real names and documentation: For now,
  /// the pair is holding:
  ///    * first  -- The index of the FitParameterSet in the *_parSetRef_
  ///            vector.  This is the index of the FitParameterSet (e.g.
  ///            "Cross Section", "Cross Section (binned), "Flux", etc)
  ///    * second -- The index of the parameter particular FitParameterSet.
  /// There is one entry per parameter used by the dial (e.g. a 1D parameter
  /// has one entry)
  std::vector<std::pair<size_t, size_t>> _inputParameterIndicesList_{};

  /// A pointer to the "global" vector of fit parameter sets.  This provides
  /// the connection to the fit parameters.
  std::vector<FitParameterSet>* _parSetRef_{nullptr};

  // for cache
  uint32_t _currentHash_{0};

};


#endif //GUNDAM_DIALINPUTBUFFER_H
