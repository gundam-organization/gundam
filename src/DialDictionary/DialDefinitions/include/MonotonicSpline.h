//
// Created by Adrien Blanchet on 22/01/2023.
//

#ifndef GUNDAM_MONOTONICSPLINE_H
#define GUNDAM_MONOTONICSPLINE_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include "TGraph.h"

#include <vector>
#include <utility>

class MonotonicSpline : public DialBase {

public:
  MonotonicSpline() = default;

  // mandatory overrides
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<MonotonicSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"MonotonicSpline"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides
  void setAllowExtrapolation(bool allowExtrapolation) override{ _allowExtrapolation_ = allowExtrapolation; }
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  // methods
  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);

  [[nodiscard]] const std::vector<double>& getDialData() const override {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  GenericToolbox::Range _splineBounds_{};
};


#endif //GUNDAM_MONOTONICSPLINE_H
