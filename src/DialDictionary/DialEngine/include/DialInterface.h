//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"

#include "FitSampleSet.h"
#include "FitParameterSet.h"

#include "string"
#include "memory"
#include "vector"


/// This class is size critical and should not be used as a base class (no
/// virtual methods.
class DialInterface {

public:
  DialInterface() = default;

  void setDialBaseRef(DialBase *dialBasePtr);
  void setInputBufferRef(DialInputBuffer *inputBufferRef);
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef);
  void setDialBinRef(const DataBin *dialBinRef);

  /// Return the input buffer containing the connection to the FitParameter(s)
  /// used by this dial.  The number of FitParameters contained in the input
  /// buffer musts mach the number expected by the specialization of the
  /// DialBase.
  [[nodiscard]] inline DialInputBuffer *getInputBufferRef() const {return _inputBufferRef_;}

  /// Get the dial calculation method.  The dial will need one or more
  /// FitParameter inputs, and the number *must* match the number and order of
  /// the FitParameters in the DialInputBuffer.
  [[nodiscard]] inline DialBase* getDialBaseRef() const {return _dialBaseRef_;}

  /// Get the DialResponseSupervisor. This conditions the return value of the
  /// dial (normally truncates between zero and the maximum response).
  [[nodiscard]] inline const DialResponseSupervisor* getResponseSupervisorRef() const {return _responseSupervisorRef_;}

  /// Get the data bin definition for the dial.
  [[nodiscard]] inline const DataBin* getDialBinRef() const {return _dialBinRef_;}

  [[nodiscard]] double evalResponse();
  [[nodiscard]] std::string getSummary(bool shallow_=true);

private:
  DialBase* _dialBaseRef_{nullptr}; // should be filled while init
  DialInputBuffer* _inputBufferRef_{nullptr};
  const DialResponseSupervisor* _responseSupervisorRef_{nullptr};
  const DataBin* _dialBinRef_{nullptr}; // for printout

};


#endif //GUNDAM_DIALINTERFACE_H
