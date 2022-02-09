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
    DialType, -1
    ,Invalid
    ,Normalization // response = dial
    ,Spline        // response = spline(dial)
    ,Graph         // response = graphInterpol(dial)
  )
}



class Dial {

public:
  Dial();
  virtual ~Dial();

  virtual void reset();

  void setApplyConditionBin(const DataBin &applyConditionBin);
  void setAssociatedParameterReference(void *associatedParameterReference);
  void setIsReferenced(bool isReferenced);
  void setUseMirrorDial(bool useMirrorDial);
  void setMirrorLowEdge(double mirrorLowEdge);
  void setMirrorRange(double mirrorRange);
  void setDialType(DialType::DialType dialType);

  void setMinimumDialResponse(double minimumDialResponse);

  void copySplineCache(TSpline3& splineBuffer_);

  virtual void initialize();

  bool isInitialized() const;
  bool isReferenced() const;
  double getDialResponseCache() const;
  const DataBin &getApplyConditionBin() const;
  DataBin &getApplyConditionBin();
  DialType::DialType getDialType() const;
  void *getAssociatedParameterReference() const;

  void updateEffectiveDialParameter();
  double evalResponse();

  virtual double evalResponse(double parameterValue_);
  virtual std::string getSummary();
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
  double _dialResponseCache_{};
  double _dialParameterCache_{};
  double _effectiveDialParameterValue_{}; // take into account internal transformations while using mirrored splines transformations
  std::shared_ptr<TSpline3> _responseSplineCache_{nullptr};

  bool _useMirrorDial_{false};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};

  double _minimumDialResponse_{std::nan("unset")};

};


#endif //GUNDAM_DIAL_H
