//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_DIAL_H
#define GUNDAM_DIAL_H

#include "string"
#include "mutex"
#include "memory"

#include "TSpline.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Wrappers.h"

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
  void setIsReferenced(bool isReferenced);

  virtual void initialize();

  bool isInitialized() const;

  double getDialResponseCache() const;
  const DataBin &getApplyConditionBin() const;
  DataBin &getApplyConditionBin();
  DialType::DialType getDialType() const;
  void *getAssociatedParameterReference() const;
  bool isReferenced() const;

  virtual std::string getSummary();
  virtual double evalResponse(double parameterValue_);
  double evalResponse();

  void copySplineCache(TSpline3& splineBuffer_);
  virtual void buildResponseSplineCache();

  virtual void fillResponseCache() = 0;

protected:

  // Parameters
  DataBin _applyConditionBin_;
  DialType::DialType _dialType_{DialType::Invalid};
  void* _associatedParameterReference_{nullptr};

  // Internals
  bool _isInitialized_{false};
  GenericToolbox::AtomicWrapper<bool> _isEditingCache_{false};
  bool _isReferenced_{false};
  bool _useMirrorDial_{false}; // TODO
  double _dialResponseCache_{};
  double _dialParameterCache_{};
  std::shared_ptr<TSpline3> _responseSplineCache_{nullptr};

};


#endif //GUNDAM_DIAL_H
