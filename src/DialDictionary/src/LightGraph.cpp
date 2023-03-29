//
// Created by Adrien Blanchet on 19/01/2023.
//

#include "LightGraph.h"


void LightGraph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}
void LightGraph::setGraph(TGraph &graph) {
  LogThrowIf(graph.GetN() == 0, "Invalid input graph");
  graph.Sort();

  nPoints = graph.GetN();

  xPoints.resize(nPoints);
  yPoints.resize(nPoints);

  memcpy(&xPoints[0], graph.GetX(), nPoints * sizeof(double));
  memcpy(&yPoints[0], graph.GetY(), nPoints * sizeof(double));
}

double LightGraph::evaluateGraph(const DialInputBuffer& input_) const{
  LogThrowIf(xPoints.empty(), "No graph point defined.");
  if (nPoints == 1) return yPoints[0];

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= xPoints[0])     { return yPoints[0]; }
    else if(input_.getBuffer()[0] >= xPoints.back()) { return yPoints.back(); }
  }

  //linear interpolation
  //In case x is < xPoints[0] or > xPoints[nPoints-1] return the extrapolated point

  //find points in graph around x assuming points are not sorted
  // (if point are sorted use a binary search)
  auto low = Int_t( TMath::BinarySearch(nPoints, &xPoints[0], input_.getBuffer()[0] ) );
  if (low == -1)  {
    // use first two points for doing an extrapolation
    low = 0;
  }

  Double_t yn;
  if (xPoints[low] == input_.getBuffer()[0]){
    yn = yPoints[low];
  }
  else{
    if (low == xPoints.size() - 1) low--; // for extrapolating
    Int_t up(low+1);

    if (xPoints[low] == xPoints[up]) return yPoints[low];
    yn = yPoints[up] + (input_.getBuffer()[0] - xPoints[up]) * (yPoints[low] - yPoints[up]) / (xPoints[low] - xPoints[up]);
  }
  return yn;
}
