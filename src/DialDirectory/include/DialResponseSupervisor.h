//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALRESPONSESUPERVISOR_H
#define GUNDAM_DIALRESPONSESUPERVISOR_H

#include "vector"
#include "functional"


class DialResponseSupervisor {

public:
  DialResponseSupervisor() = default;

  void process(double &output_) const;

private:
  std::vector<std::function<void(double&)>> _functionsList_;


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
