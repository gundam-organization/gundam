//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_GENERALSPLINE_H
#define GUNDAM_GENERALSPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "TGraph.h"
#include "TSpline.h"

#include "vector"
#include "utility"


class GeneralSpline : public DialBase {

public:
  GeneralSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<GeneralSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"GeneralSpline"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  [[nodiscard]] bool getAllowExtrapolation() const override;

  /// Pass information to the dial so that it can build it's
  /// internal information.  New build overloads should be
  /// added as we have classes of dials
  /// (e.g. multi-dimensional dials).
  virtual void buildDial(const TGraph& grf, std::string option="") override;
  virtual void buildDial(const TSpline3& spl, std::string option="") override;
  virtual void buildDial(const std::vector<double>& v1,
                         const std::vector<double>& v2,
                         const std::vector<double>& v3,
                         std::string option="") override;

   const std::vector<double>& getDialData() const override {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};
};

typedef CachedDial<GeneralSpline> GeneralSplineCache;


#endif //GUNDAM_GENERALSPLINE_H
