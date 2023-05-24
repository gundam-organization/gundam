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

  void setMinResponse(double minResponse);
  void setMaxResponse(double maxResponse);

  [[nodiscard]] double process(double reponse_) const;

  [[nodiscard]] std::string getSummary() const;

private:
  double _minResponse_{std::nan("unset")};
  double _maxResponse_{std::nan("unset")};


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
