//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_DIAL_H
#define XSLLHFITTER_DIAL_H

#include "string"

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

  virtual void initialize();

  const DataBin &getApplyConditionBin() const;
  DialType::DialType getDialType() const;

  virtual std::string getSummary();
  double evalResponse(const double& parameterValue_);
  virtual void updateResponseCache(const double& parameterValue_) = 0;

protected:
  // Parameters
  DataBin _applyConditionBin_;
  DialType::DialType _dialType_{DialType::Invalid};

  // Internals
  double _dialResponseCache_{};
  double _dialParameterCache_{};

};


#endif //XSLLHFITTER_DIAL_H
