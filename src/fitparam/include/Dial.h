//
// Created by Nadrino on 21/05/2021.
//

#ifndef XSLLHFITTER_DIAL_H
#define XSLLHFITTER_DIAL_H

#include "string"
#include "mutex"
#include "memory"

#include "GenericToolbox.h"

#include "DataBin.h"


namespace DialType{
  ENUM_EXPANDER(
    DialType, -1,
    Invalid,
    Normalization, // response = dial
    Spline,        // response = spline(dial)
    Graph,         // response = graphInterpol(dial)
    Other
  );

  DialType toDialType(const std::string& dialStr_);
}



class Dial {

public:
  Dial();
  virtual ~Dial();

  virtual void reset();

  void setApplyConditionBin(const DataBin &applyConditionBin);
  void setAssociatedParameterReference(void *associatedParameterReference);

  virtual void initialize();

  double getDialResponseCache() const;
  const DataBin &getApplyConditionBin() const;
  DialType::DialType getDialType() const;
  void *getAssociatedParameterReference() const;

  virtual std::string getSummary();
  virtual double evalResponse(const double& parameterValue_);
  double evalResponse();

protected:
  virtual void fillResponseCache() = 0;

  // Parameters
  DataBin _applyConditionBin_;
  DialType::DialType _dialType_{DialType::Invalid};
  void* _associatedParameterReference_{nullptr};

  // Internals
  bool _isEditingCache_{false};
  double _dialResponseCache_{};
  double _dialParameterCache_{};

};


#endif //XSLLHFITTER_DIAL_H
