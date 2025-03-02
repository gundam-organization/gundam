#ifndef GUNDAM_BILINEAR_INTERPOLATION_H
#define GUNDAM_BILINEAR_INTERPOLATION_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include <vector>
#include <utility>

/// Manage a bilinear interpolation for a grid of points.
class Bilinear : public DialBase {

public:
  Bilinear() = default;
  ~Bilinear() override = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Bilinear>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override {return {"Bilinear"};}
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  // other overrides
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  void buildDial(const TH2& h2);


protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};

  // The vector of input parameter bounds.
  std::vector<GenericToolbox::Range> _splineBounds_;
};


#endif //GUNDAM_COMPACTSPLINE_H
