//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_ROOT_SPLINE_H
#define GUNDAM_ROOT_SPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include "TSpline.h"

class RootSpline : public DialBase {

public:
  RootSpline() = default;

  // mandatory overrides
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootSpline"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides
  void setAllowExtrapolation(bool allowExtrapolation) override{ _allowExtrapolation_ = allowExtrapolation; }

  // setters
  void setSpline(const TSpline3 &spline_){ _spline_ = spline_; }

  // getters
  [[nodiscard]] const TSpline3 &getSpline() const { return _spline_; }

  // methods
  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);

protected:
  bool _allowExtrapolation_{false};
  TSpline3 _spline_{};

};

#endif // GUNDAM_ROOT_SPLINE_H
