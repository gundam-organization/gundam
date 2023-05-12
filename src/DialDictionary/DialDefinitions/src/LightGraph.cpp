//
// Created by Adrien Blanchet on 19/01/2023.
//

#include "LightGraph.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[LightGraph]");
});

void LightGraph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool LightGraph::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void LightGraph::buildDial(const TGraph &grf, const std::string& option_) {
  LogThrowIf(grf.GetN() == 0, "Invalid input graph");
  TGraph graph(grf);
  graph.Sort();

  nPoints = graph.GetN();

  xPoints.resize(nPoints);
  yPoints.resize(nPoints);

  memcpy(&xPoints[0], graph.GetX(), nPoints * sizeof(double));
  memcpy(&yPoints[0], graph.GetY(), nPoints * sizeof(double));
}

double LightGraph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for LightGraph");
#endif

  LogThrowIf(xPoints.empty(), "No graph point defined.");
  if (nPoints == 1) return yPoints[0];

  if( not _allowExtrapolation_ ){
    if     (dialInput <= xPoints[0])     { return yPoints[0]; }
    else if(dialInput >= xPoints.back()) { return yPoints.back(); }
  }

  //linear interpolation
  //In case x is < xPoints[0] or > xPoints[nPoints-1] return the extrapolated point

  //find points in graph around x assuming points are not sorted
  // (if point are sorted use a binary search)
  auto low = Int_t( TMath::BinarySearch(nPoints, &xPoints[0], dialInput) );
  if (low == -1)  {
    // use first two points for doing an extrapolation
    low = 0;
  }

  Double_t yn;
  if (xPoints[low] == dialInput){
    yn = yPoints[low];
  }
  else{
    if (low == xPoints.size() - 1) low--; // for extrapolating
    Int_t up(low+1);

    if (xPoints[low] == xPoints[up]) return yPoints[low];
    yn = yPoints[up] + (dialInput - xPoints[up]) * (yPoints[low] - yPoints[up]) / (xPoints[low] - xPoints[up]);
  }
  return yn;
}
