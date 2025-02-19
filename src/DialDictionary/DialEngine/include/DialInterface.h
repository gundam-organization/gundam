//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"

#include "SampleSet.h"
#include "ParameterSet.h"

#include <string>
#include <memory>
#include <vector>


/// This class is size critical and should not be used as a base class (no
/// virtual methods.
class DialInterface {

public:
  DialInterface() = default;

  void setDial(const GenericToolbox::PolymorphicObjectWrapper<DialBase>& dial_){ _dial_ = dial_; }
  void setInputBufferRef(DialInputBuffer *inputBufferRef){ _inputBufferPtr_ = inputBufferRef; }
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef){ _responseSupervisorPtr_ = responseSupervisorRef; }
  void setDialBinRef(const Bin *dialBinRef){ _dialBinRef_ = dialBinRef; }

  /// Return the input buffer containing the connection to the Parameter(s)
  /// used by this dial.  The number of Parameters contained in the input
  /// buffer musts mach the number expected by the specialization of the
  /// DialBase.
  [[nodiscard]] const DialInputBuffer *getInputBufferRef() const {return _inputBufferPtr_;}

  /// Get the dial calculation method.  The dial will need one or more
  /// Parameter inputs, and the number *must* match the number and order of
  /// the Parameters in the DialInputBuffer.
  [[nodiscard]] DialBase* getDialBaseRef() const {return _dial_.get();}

  /// Get the DialResponseSupervisor. This conditions the return value of the
  /// dial (normally truncates between zero and the maximum response).
  [[nodiscard]] const DialResponseSupervisor* getResponseSupervisorRef() const {return _responseSupervisorPtr_;}

  /// Get the data bin definition for the dial.
  [[nodiscard]] const Bin* getDialBinRef() const {return _dialBinRef_;}

  [[nodiscard]] double evalResponse() const { return _responseSupervisorPtr_->process(_dial_->evalResponse(*_inputBufferPtr_)); }
  [[nodiscard]] std::string getSummary(bool shallow_=true) const;

private:
  // owner of
  GenericToolbox::PolymorphicObjectWrapper<DialBase> _dial_{};

  // non owner of
  const DialInputBuffer* _inputBufferPtr_{nullptr};
  const DialResponseSupervisor* _responseSupervisorPtr_{nullptr};
  const Bin* _dialBinRef_{nullptr}; // for printout

};


#endif //GUNDAM_DIALINTERFACE_H
