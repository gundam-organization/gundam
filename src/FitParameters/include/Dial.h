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

class DialSet;

class Dial {

protected:
  // Not supposed to define a bare Dial. Use the downcast instead
  explicit Dial(DialType::DialType dialType_);
  virtual ~Dial();

public:
  virtual void reset();

  void setApplyConditionBin(const DataBin &applyConditionBin);
  void setAssociatedParameterReference(void *associatedParameterReference);
  void setIsReferenced(bool isReferenced);
  void setUseMirrorDial(bool useMirrorDial);
  void setMirrorLowEdge(double mirrorLowEdge);
  void setMirrorRange(double mirrorRange);
  void setMinDialResponse(double minDialResponse_);
  void setMaxDialResponse(double maxDialResponse_);
  void setOwner(const DialSet* dialSetPtr);

  virtual void initialize();

  bool isInitialized() const;
  bool isReferenced() const;
  double getDialResponseCache() const;
  const DataBin &getApplyConditionBin() const;
  DataBin &getApplyConditionBin();
  DialType::DialType getDialType() const;
  const DialSet* getOwner() const;

  void *getAssociatedParameterReference() const;
  double getAssociatedParameter() const;
  int getAssociatedParameterIndex() const;
  double getMinDialResponse() const {return _minDialResponse_;}
  double getMaxDialResponse() const {return _maxDialResponse_;}
  bool getUseMirrorDial() const {return _useMirrorDial_;}
  double getMirrorLowEdge() const {return _mirrorLowEdge_;}
  double getMirrorRange() const {return _mirrorRange_;}

  void updateEffectiveDialParameter();
  double evalResponse();
  void copySplineCache(TSpline3& splineBuffer_);

  virtual double evalResponse(double parameterValue_);
  virtual std::string getSummary();
  virtual void buildResponseSplineCache();
  virtual void fillResponseCache() = 0;

#ifdef CACHE_MANAGER_SLOW_VALIDATION
  // Debugging.  This is only meaningful when the GPU is filling the spline
  // value cache (only filled during validation).  It's a nullptr otherwise,
  // or not included in the object.
  std::string getGPUCacheName() const {return _GPUCacheName_;}
  void setGPUCacheName(std::string s) {_GPUCacheName_ = s;}
  double* getGPUCachePointer() const {return _GPUCachePointer_;}
  void setGPUCachePointer(double* v) {_GPUCachePointer_=v;}
#endif

protected:
  const DialType::DialType _dialType_;
  // The DialSet that owns this dial.  The dial DOES NOT OWN THIS POINTER
  const DialSet* _ownerDialSetReference_{nullptr};

  // Parameters
  DataBin _applyConditionBin_;
  void* _associatedParameterReference_{nullptr};

  // Internals
  bool _isInitialized_{false};
  std::shared_ptr<std::mutex> _isEditingCache_;
  bool _isReferenced_{false};
  double _dialResponseCache_{};
  double _dialParameterCache_{};
  double _effectiveDialParameterValue_{}; // take into account internal transformations while using mirrored splines transformations

  // Response cap
  double _minDialResponse_{std::nan("unset")};
  double _maxDialResponse_{std::nan("unset")};

  // Dial mirroring
  bool _useMirrorDial_{false};
  double _mirrorLowEdge_{std::nan("unset")};
  double _mirrorRange_{std::nan("unset")};

#ifdef CACHE_MANAGER_SLOW_VALIDATION
  // Debugging.  This is only meaningful when the GPU is filling the spline
  // value cache (only filled during validation).
  std::string _GPUCacheName_{"unset"};
  double* _GPUCachePointer_{nullptr};
#endif

  // Output
  std::shared_ptr<TSpline3> _responseSplineCache_{nullptr}; // dial response as a spline

};
#endif //GUNDAM_DIAL_H
