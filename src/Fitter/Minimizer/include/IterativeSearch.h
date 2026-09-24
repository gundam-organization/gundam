//
// Created by Haowei Zheng on 19/08/2026.
//

#ifndef GUNDAM_ITERATIVE_SEARCH_H
#define GUNDAM_ITERATIVE_SEARCH_H

// IterativeSearch is a minimizer that iterates over a set of "search
// parameters", and at every point minimizes the remaining parameters with a
// ROOT::Math::Minimizer. One reduced entry per point is written out.
//
// Work in progress: the iteration logic is being migrated step by step. For
// now this class resolves the search parameters and sets up the underlying
// ROOT minimizer on the remaining ones.

#include "ParameterSet.h"
#include "MinimizerBase.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"

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

  // overrides
  void minimize() override;
  [[nodiscard]] bool isErrorCalcEnabled() const override { return false; }

  // c-tor
  explicit IterativeSearch(FitterEngine* owner_): MinimizerBase(owner_) {}

  // const getters
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer>& getMinimizer() const{ return _rootMinimizer_; }
  [[nodiscard]] const std::vector<SearchParameter>& getSearchParameterList() const{ return _searchParameterList_; }

protected:
  // name -> Parameter*, checks, setIsFixed(true)
  void resolveSearchParameters();
  // erase the search parameters from getMinimizerFitParameterPtr()
  void stripSearchParametersFromList();
  // Clear() + SetVariable loop. Used at init and for any restart from a given point.
  void resetMinimizer(const std::vector<double>& startValues_);

private:
  // config
  std::vector<SearchParameter> _searchParameterList_{};

  int _strategy_{0}; // 0: fewest gradient cycles, enough without Hesse
  int _printLevel_{0};
  double _tolerance_{1.};
  unsigned int _maxIterations_{500};
  unsigned int _maxFcnCalls_{1000000000};
  std::string _minimizerType_{"Minuit2"};
  std::string _minimizerAlgo_{"Migrad"};

  // internals
  ROOT::Math::Functor _functor_{};
  std::unique_ptr<ROOT::Math::Minimizer> _rootMinimizer_{nullptr};

  // starting point in fit space, replayed by a cold start
  std::vector<double> _prefitValues_{};
  std::vector<double> _prefitSteps_{};

};

#endif //GUNDAM_ITERATIVE_SEARCH_H
