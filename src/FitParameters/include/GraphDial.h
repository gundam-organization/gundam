//
// Created by Adrien BLANCHET on 02/12/2021.
//

#ifndef GUNDAM_GRAPHDIAL_H
#define GUNDAM_GRAPHDIAL_H

#include "TGraph.h"

#include "Dial.h"

class GraphDial : public Dial {

public:
  GraphDial();

  void reset() override;

  void setGraph(const TGraph &graph);

  void initialize() override;
//  double evalResponse(const double &parameterValue_) override;
  void fillResponseCache() override;
  std::string getSummary() override;

private:
  TGraph _graph_;
};


#endif //GUNDAM_GRAPHDIAL_H
