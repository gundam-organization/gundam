#include "Histogram.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

TEST(BinningGradient, MultidimensionalFractionsMatchFiniteDifferences){
  for( bool relative : {false, true} ){
    Histogram histogram;
    JsonType axes = {{{"name", "X"}, {relative ? "widthFraction" : "width", .25}},
                     {{"name", "Y"}, {"width", .3}}};
    histogram.build(ConfigReader(JsonType{{"binningDefinition", {
        {{"name", "X"}, {"edges", {0., 1., 2.5}}},
        {{"name", "Y"}, {"edges", {0., 1., 2.}}}}} }));
    histogram.configureBinningResponse(ConfigReader(JsonType{{"isEnabled", true}, {"variables", axes}}));
    const std::vector<std::string> names{"X", "Y"};
    Event event;
    event.getVariables().setVarNameList(names);
    for( const auto& point : {std::make_pair(.9, 1.1), std::make_pair(-.1, .9),
                             std::make_pair(2.6, 2.1), std::make_pair(.5, .5),
                             std::make_pair(-2., -2.), std::make_pair(1., 1.)} ){
      event.getVariables().fetchVariable("X").set(point.first);
      event.getVariables().fetchVariable("Y").set(point.second);
      const auto gradient = histogram.evalBinningGradient(event);
      ASSERT_EQ(2U, gradient.variableIndices.size());
      for( int axis = 0; axis < 2; ++axis ){
        const auto found = std::find(gradient.variableIndices.begin(), gradient.variableIndices.end(), axis);
        ASSERT_NE(gradient.variableIndices.end(), found);
        const auto column = size_t(found - gradient.variableIndices.begin());
        const double coordinate = axis == 0 ? point.first : point.second;
        auto& variable = event.getVariables().fetchVariable(names[axis]);
        variable.set(coordinate + 1E-6);
        const auto upper = histogram.evaluateEventBinningResponse(event);
        variable.set(coordinate - 1E-6);
        const auto lower = histogram.evaluateEventBinningResponse(event);
        variable.set(coordinate);
        const auto getFraction = [](const BinningResponse::EventResponse& response_, int bin_){
          for( const auto& fraction : response_.binFractions ){
            if( fraction.binIndex == bin_ ){ return fraction.fraction; }
          }
          return 0.;
        };
        double totalGradient = 0.;
        for( int bin = 0; bin < histogram.getNbBins(); ++bin ){
          double actual = 0.;
          for( size_t row = 0; row < gradient.response.binFractions.size(); ++row ){
            if( gradient.response.binFractions[row].binIndex == bin ){
              actual = gradient.fractionGradients[row][column];
            }
          }
          EXPECT_NEAR(actual, (getFraction(upper, bin) - getFraction(lower, bin)) / 2E-6, 1E-8);
          totalGradient += actual;
        }
        // The outside fraction closes the partition, including domain edges.
        EXPECT_NEAR(-totalGradient, (upper.outsideFraction - lower.outsideFraction) / 2E-6, 1E-8);
      }
    }
  }
}

TEST(BinningGradient, MovingHardAxesAreRejected){
  Histogram histogram;
  histogram.build(ConfigReader(JsonType{{"binningDefinition", {
      {{"name", "X"}, {"edges", {0., 1., 2.}}}, {{"name", "Y"}, {"edges", {0., 1.}}}}} }));
  histogram.configureBinningResponse(ConfigReader(JsonType{{"isEnabled", true}, {"variables", {
      {{"name", "X"}, {"width", .2}}}}}));
  EXPECT_NO_THROW(histogram.validateGradientVariable("X"));
  EXPECT_NO_THROW(histogram.validateGradientVariable("unbinnedVariable"));
  EXPECT_THROW(histogram.validateGradientVariable("Y"), std::runtime_error);
  Histogram hard;
  hard.build(ConfigReader(JsonType{{"binningDefinition", {{{"name", "X"}, {"edges", {0., 1.}}}}}}));
  EXPECT_THROW(hard.validateGradientVariable("X"), std::runtime_error);
}
