//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALRESPONSESUPERVISOR_H
#define GUNDAM_DIALRESPONSESUPERVISOR_H

#include "vector"
#include "functional"
#include "cmath"
#include "string"


class DialResponseSupervisor {

public:
  DialResponseSupervisor() = default;

  void setMinResponse(double minResponse);
  void setMaxResponse(double maxResponse);

  void process(double &output_) const;

  std::string getSummary() const;

private:
  double _minResponse_{std::nan("unset")};
  double _maxResponse_{std::nan("unset")};


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
