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
  void setMinResponse(double minResponse){ _minResponse_ = minResponse; }
  void setMaxResponse(double maxResponse){ _maxResponse_ = maxResponse; }

  // const getters
  [[nodiscard]] double getMinResponse() const{ return _minResponse_; }
  [[nodiscard]] double getMaxResponse() const{ return _maxResponse_; }

  // Compare configuration, treating unset (NaN) limits as equal.
  [[nodiscard]] bool isEquivalentTo(const DialResponseSupervisor& other_) const{
    const auto equal = [](double left_, double right_){
      return left_ == right_ or (std::isnan(left_) and std::isnan(right_));
    };
    return equal(_minResponse_, other_._minResponse_) and equal(_maxResponse_, other_._maxResponse_);
  }

  [[nodiscard]] double process(double reponse_) const;
  [[nodiscard]] std::string getSummary(bool shallow_ = true) const;


private:
  double _minResponse_{std::nan("unset")};
  double _maxResponse_{std::nan("unset")};


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
