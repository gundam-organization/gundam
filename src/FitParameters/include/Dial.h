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
#include "GenericToolbox.OrderedLock.h"

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
  void setIsReferenced(bool isReferenced);
  void setOwner(const DialSet* dialSetPtr);

  virtual void initialize();

  bool isReferenced() const;
  double getDialResponseCache() const;
  const DataBin* getApplyConditionBinPtr() const{ return _applyConditionBin_.get(); }
  const DataBin &getApplyConditionBin() const;
  DataBin &getApplyConditionBin();
  DialType::DialType getDialType() const;
  const DialSet* getOwner() const;

  void *getAssociatedParameterReference() const;
  double getAssociatedParameter() const;

  void updateEffectiveDialParameter();
  double evalResponse();
//  void copySplineCache(TSpline3& splineBuffer_);

  virtual double evalResponse(double parameterValue_);
  virtual std::string getSummary();
//  virtual void buildResponseSplineCache();
  virtual void fillResponseCache() = 0;

protected:
  const DialType::DialType _dialType_;
  // The DialSet that owns this dial.  The dial DOES NOT OWN THIS POINTER
  const DialSet* _ownerDialSetReference_{nullptr};

  // Parameters
  std::shared_ptr<DataBin> _applyConditionBin_{nullptr};

  // Internals
  bool _isEditingCache_{false};
  std::shared_ptr<std::mutex> _evalDialLock_{nullptr};
  bool _isReferenced_{false};
  double _dialResponseCache_{};
  double _dialParameterCache_{};
  double _effectiveDialParameterValue_{}; // take into account internal transformations while using mirrored splines transformations

#ifdef GUNDAM_USING_CUDA
  // Debugging.  This is only meaningful when the GPU is filling the spline
  // value cache (only filled during validation).
public:
  void setCacheManagerName(std::string s) {_CacheManagerName_ = s;}
  void setCacheManagerIndex(int i) {_CacheManagerIndex_ = i;}
  void setCacheManagerValuePointer(double* v) {_CacheManagerValue_ = v;}
  void setCacheManagerValidPointer(bool* v) {_CacheManagerValid_ = v;}
  std::string getCacheManagerName() {return _CacheManagerName_;}
  int  getCacheManagerIndex() {return _CacheManagerIndex_;}
  const double* getCacheManagerValuePointer() {return _CacheManagerValue_;}
  const bool* getCacheManagerValidPointer() {return _CacheManagerValid_;}
  void (*getCacheManagerUpdatePointer())() {return _CacheManagerUpdate_;}
private:
  std::string _CacheManagerName_{"unset"};
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _CacheManagerIndex_{-1};
  // A pointer to the cached result.
  double* _CacheManagerValue_{nullptr};
  // A pointer to the cache validity flag.
  bool* _CacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_CacheManagerUpdate_)(){nullptr};
#endif

  // Output
//  std::shared_ptr<TSpline3> _responseSplineCache_{nullptr}; // dial response as a spline

};
#endif //GUNDAM_DIAL_H
