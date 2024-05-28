#ifndef GUNDAM_BILINEAR_INTERPOLATION_H
#define GUNDAM_BILINEAR_INTERPOLATION_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include <vector>
#include <utility>

class Bilinear : public DialBase {

public:
  Bilinear() = default;
  ~Bilinear() override = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Bilinear>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"Bilinear"}; }
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  [[nodiscard]] std::string getSummary() const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  [[nodiscard]] bool getAllowExtrapolation() const override;

  void buildSplineData(TGraph& graph_);
  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

  /// Pass information to the dial so that it can build it's
  /// internal information.  New build overloads should be
  /// added as we have classes of dials
  /// (e.g. multi-dimensional dials).
  virtual void buildDial(const TH2D& grf, const std::string& option_="") override;

  [[nodiscard]] const std::vector<double>& getDialData() const override {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};
};

typedef CachedDial<Bilinear> BilinearCache;


#endif //GUNDAM_COMPACTSPLINE_H
