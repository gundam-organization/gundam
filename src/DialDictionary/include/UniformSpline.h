//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_UNIFORMSPLINE_H
#define GUNDAM_UNIFORMSPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "TGraph.h"
#include "TSpline.h"

#include "vector"
#include "utility"

class UniformSpline : public DialBase {

public:
  UniformSpline() = default;
  virtual ~UniformSpline() = default;

  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<UniformSpline>(*this); }
  [[nodiscard]] std::string getDialTypeName() const override { return {"UniformSpline"}; }
  double evalResponse(const DialInputBuffer& input_) const override;

  void setAllowExtrapolation(bool allowExtrapolation) override;
  bool getAllowExtrapolation() const override;

  /// Pass information to the dial so that it can build it's
  /// internal information.  New build overloads should be
  /// added as we have classes of dials
  /// (e.g. multi-dimensional dials).
  void buildDial(const TGraph& grf, const std::string &option= "") override;
  void buildDial(const TSpline3& spl, const std::string &option_="") override;
  void buildDial(const std::vector<double>& v1,
                 const std::vector<double>& v2,
                 const std::vector<double>& v3,
                 const std::string &option_= "") override;

   const std::vector<double>& getDialData() const override {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};
};


#endif //GUNDAM_UNIFORMSPLINE_H
