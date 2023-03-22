//
// Created by Adrien Blanchet on 22/01/2023.
//

#ifndef GUNDAM_COMPACTSPLINEHANDLER_H
#define GUNDAM_COMPACTSPLINEHANDLER_H

#include "DialInputBuffer.h"

#include "TGraph.h"

#include "vector"
#include "utility"


class CompactSplineHandler {

public:
  CompactSplineHandler() = default;
  virtual ~CompactSplineHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);

  void buildSplineData(TGraph& graph_);
  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

  bool getIsUniform() const {return true;}
  const std::vector<double>& getSplineData() const {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};

};


#endif //GUNDAM_COMPACTSPLINEHANDLER_H
