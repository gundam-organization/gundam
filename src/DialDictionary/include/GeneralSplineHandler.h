//
// Created by Adrien Blanchet on 23/01/2023.
//

#ifndef GUNDAM_GENERALSPLINEHANDLER_H
#define GUNDAM_GENERALSPLINEHANDLER_H


#include "DialInputBuffer.h"

#include "TGraph.h"

#include "vector"
#include "utility"


class GeneralSplineHandler {

public:
  GeneralSplineHandler() = default;
  virtual ~GeneralSplineHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);

  void buildSplineData(TGraph& graph_);
  void buildSplineData(const TSpline3& sp_);
  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};

};


#endif //GUNDAM_GENERALSPLINEHANDLER_H
