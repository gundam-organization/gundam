//
// Created by Haowei Zheng on 19/08/2026.
//

#ifndef GUNDAM_GRID_SEARCH_H
#define GUNDAM_GRID_SEARCH_H

// GridSearch is a minimizer that scans a set of "grid parameters" over a
// grid and, at every node, minimizes the remaining parameters with a
// ROOT::Math::Minimizer. One reduced entry per node is written out.
//
// Work in progress: the grid logic is being migrated step by step. For now
// this class only sets up the underlying ROOT minimizer.

#include "MinimizerBase.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"

#include <memory>
#include <string>


class GridSearch : public MinimizerBase {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  // overrides
  void minimize() override;
  [[nodiscard]] bool isErrorCalcEnabled() const override { return false; }

  // c-tor
  explicit GridSearch(FitterEngine* owner_): MinimizerBase(owner_) {}

  // const getters
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer>& getMinimizer() const{ return _rootMinimizer_; }

private:
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

};

#endif //GUNDAM_GRID_SEARCH_H
