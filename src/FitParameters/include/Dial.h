//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_DIAL_H
#define GUNDAM_DIAL_H


#include "DataBin.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Wrappers.h"

#include "TSpline.h"

#include <mutex>
#include <string>
#include <memory>

#ifdef USE_NEW_DIALS
#define DEPRECATED [[deprecated("Not used with new dial implementation")]]
#else
#define DEPRECATED /*[[deprecated("Not used with new dial implementation")]]*/
#endif

namespace DialType{
  ENUM_EXPANDER(
    DialType, -1
    ,Invalid
    ,Norm          // response = dial
    ,Spline        // response = spline(dial)
    ,Graph         // response = graphInterpol(dial)
  )
}

class DialSet; // owner
class DialWrapper;

class DEPRECATED Dial {

public:
  // Don't check the mask everytime since it is memory delocalized
  static bool enableMaskCheck;
  static bool disableDialCache;
  static bool throwIfResponseIsNegative;

protected:
  // Not supposed to define a bare Dial. Use the downcast instead
  explicit Dial(DialType::DialType dialType_, const DialSet *owner_);

public:
  virtual ~Dial() = default;
  virtual std::unique_ptr<Dial> clone() const = 0;

  void setApplyConditionBin(DataBin *applyConditionBin);
  void setIsReferenced(bool isReferenced);
  void setOwner(const DialSet* dialSetPtr);

  virtual void initialize();

  // const getters
  bool isReferenced() const;
  bool isMasked() const;
  double getDialResponseCache() const;
  double getAssociatedParameter() const;
  DialType::DialType getDialType() const;
  const DataBin* getApplyConditionBinPtr() const;
  const DialSet* getOwner() const;

  // getters
  DataBin* getApplyConditionBinPtr();

  // calc
  double getEffectiveDialParameter(double parameterValue_);
  double capDialResponse(double response_);
  double evalResponse();

  // virtual
  virtual double calcDial(double parameterValue_) = 0;
  virtual double evalResponse(double parameterValue_);
  virtual std::string getSummary();

  // debug
  virtual void writeSpline(const std::string &fileName_) const {}


//  void copySplineCache(TSpline3& splineBuffer_);
//  virtual void buildResponseSplineCache();

protected:
  //! KEEP THE MEMBER AS LIGHT AS POSSIBLE!!
  const DialType::DialType _dialType_; // Defines the
  const DialSet* _owner_{nullptr};

  // Parameters
  DataBin* _applyConditionBin_{nullptr};

  // Internals
  GenericToolbox::NoCopyWrapper<std::mutex> _evalDialLock_;
  bool _isReferenced_{false};
  double _dialResponseCache_{std::nan("unset")};
  double _dialParameterCache_{std::nan("unset")};

#ifdef GUNDAM_USING_CACHE_MANAGER
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
