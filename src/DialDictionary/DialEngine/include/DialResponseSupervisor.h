//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALRESPONSESUPERVISOR_H
#define GUNDAM_DIALRESPONSESUPERVISOR_H

#include <cmath>
#include <string>
#include <vector>
#include <functional>


class DialResponseSupervisor {

public:
  DialResponseSupervisor() = default;

  // setters
  void setMinResponse(double minResponse_){ _minResponse_ = minResponse_; updateTransitions(); }
  void setMaxResponse(double maxResponse_){ _maxResponse_ = maxResponse_; updateTransitions(); }
  void setSmoothingWidths(double minWidth_, double maxWidth_);

  // const getters
  [[nodiscard]] double getMinResponse() const{ return _minResponse_; }
  [[nodiscard]] double getMaxResponse() const{ return _maxResponse_; }
  [[nodiscard]] double getMinSmoothingWidth() const{ return _minSmoothingWidth_; }
  [[nodiscard]] double getMaxSmoothingWidth() const{ return _maxSmoothingWidth_; }

  [[nodiscard]] double process(double response_) const;
  [[nodiscard]] std::string getSummary(bool shallow_ = true) const;


private:
  void updateTransitions();

  // These equal the hard limits when smoothing is disabled. The ordinary
  // in-range path therefore needs no additional smoothing flag or arithmetic.
  double _minTransitionStart_{std::nan("unset")};
  double _maxTransitionStart_{std::nan("unset")};
  double _minResponse_{std::nan("unset")};
  double _maxResponse_{std::nan("unset")};
  double _minSmoothingWidth_{0.};
  double _maxSmoothingWidth_{0.};
  double _minInverseTransitionWidth_{0.};
  double _maxInverseTransitionWidth_{0.};


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
