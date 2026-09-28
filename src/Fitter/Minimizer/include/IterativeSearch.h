//
// Created by Haowei Zheng on 19/08/2026.
//

#ifndef GUNDAM_ITERATIVE_SEARCH_H
#define GUNDAM_ITERATIVE_SEARCH_H

// IterativeSearch is a minimizer that iterates over a set of "search
// parameters", and at every point minimizes the remaining parameters with a
// ROOT::Math::Minimizer. One reduced entry per point is written out.
//
// The points are the cartesian product of the values given for each search
// parameter, visited in a walk where consecutive points are neighbours. A job
// can be told to run only a slice of the walk, so the work can be shared.
//
// Output, under postFit/ of the fit directory:
//   pointList          one entry per visited point: llh, convergence, the profiled values
//                      and (optionally) the post-fit values of the other parameters
//   problemDefinition  one entry: which parameters were profiled over which values, the
//                      key for the arrays in pointList, and the best point of this file

#include "ParameterSet.h"
#include "MinimizerBase.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TDirectory.h"
#include "TTree.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>


class IterativeSearch : public MinimizerBase {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  struct SearchParameter {
    std::string parameterSetName{};
    std::string parameterName{};
    std::vector<double> values{};

    // filled by resolveSearchParameters()
    Parameter* parPtr{nullptr};
  };

  struct SearchPoint {
    int index{-1};                // position in the walk, what firstPoint/pointList refer to
    std::vector<double> values{}; // one value per search parameter, same order as the list
  };

  // overrides
  void minimize() override;
  [[nodiscard]] bool isErrorCalcEnabled() const override { return false; }

  // c-tor
  explicit IterativeSearch(FitterEngine* owner_): MinimizerBase(owner_) {}

  // const getters
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer>& getMinimizer() const{ return _rootMinimizer_; }
  [[nodiscard]] const std::vector<SearchParameter>& getSearchParameterList() const{ return _searchParameterList_; }
  [[nodiscard]] const std::vector<SearchPoint>& getSearchPointList() const{ return _searchPointList_; }
  [[nodiscard]] const std::vector<int>& getRunList() const{ return _runList_; }

protected:
  // name -> Parameter*, checks, setIsFixed(true)
  void resolveSearchParameters();
  // erase the search parameters from getMinimizerFitParameterPtr()
  void stripSearchParametersFromList();
  // Clear() + SetVariable loop. Used at init and for any restart from a given point.
  void resetMinimizer(const std::vector<double>& startValues_);
  // cartesian product of the search parameter values, in walk order
  void buildSearchPointList();
  // firstPoint/nbPoints or pointList -> _runList_
  void buildRunList();
  // pointList tree: one entry per visited point. Nothing is booked without an output file.
  void bookPointList();
  // problemDefinition tree: one entry, the key for ProfiledValues[] and
  // PostFitParameterValues[], plus the best point of this file
  void writeProblemDefinition(int bestPoint_, double bestLlh_);

private:
  // config
  std::vector<SearchParameter> _searchParameterList_{};

  // which points this job runs: pointList if given, otherwise nbPoints from firstPoint
  int _firstPoint_{0};
  int _nbPoints_{-1};                 // <0: all points from _firstPoint_
  std::vector<int> _runList_{};
  // reverse every other row of the walk so consecutive points are neighbours
  bool _serpentine_{true};
  // start each point from the previous point's best fit instead of the prefit point
  bool _warmStart_{true};
  // redo a point from the prefit point if its llh is above the previous one by more than this. nan: never
  double _coldRefitThreshold_{std::nan("unset")};
  // write PostFitParameterValues[] in the pointList tree
  bool _saveParameterVector_{true};

  int _strategy_{0}; // 0: fewest gradient cycles, enough without Hesse
  int _printLevel_{0};
  double _tolerance_{1.};
  unsigned int _maxIterations_{500};
  unsigned int _maxFcnCalls_{1000000000};
  std::string _minimizerType_{"Minuit2"};
  std::string _minimizerAlgo_{"Migrad"};

  // internals
  std::vector<SearchPoint> _searchPointList_{};

  // owned by the output directory
  TTree* _pointListTree_{nullptr};

  // branch buffers, addresses have to outlive the fill loop
  int _outPoint_{0};
  int _outNbCalls_{0};
  bool _outConverged_{false};
  double _outLlh_{0};
  double _outLlhStat_{0};
  double _outLlhPenalty_{0};
  double _outEdm_{0};
  std::vector<double> _outProfiledValues_{};
  std::vector<double> _outParValues_{};

  ROOT::Math::Functor _functor_{};
  std::unique_ptr<ROOT::Math::Minimizer> _rootMinimizer_{nullptr};

  // starting point in fit space, replayed by a cold start
  std::vector<double> _prefitValues_{};
  std::vector<double> _prefitSteps_{};

};

#endif //GUNDAM_ITERATIVE_SEARCH_H
