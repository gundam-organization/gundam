//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_UNIFORMSPLINE_H
#define GUNDAM_UNIFORMSPLINE_H


#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "SplineUtils.h"

#include <vector>
#include <utility>


class UniformSpline : public DialBase {

public:
  UniformSpline() = default;

  // mandatory overrides
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<UniformSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"UniformSpline"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  void buildDial(const std::vector<SplineUtils::SplinePoint>& splinePointList_);

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  DialUtils::Range _splineBounds_{std::nan("unset"), std::nan("unset")};
};


#endif //GUNDAM_UNIFORMSPLINE_H
