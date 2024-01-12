//
// Created by Clark McGrew on 25/01/2023.
//

#ifndef GUNDAM_MinimizerBase_h
#define GUNDAM_MinimizerBase_h

#include "Propagator.h"
#include "Parameter.h"
#include "JsonBaseClass.h"
#include "LikelihoodInterface.h"

#include "GenericToolbox.Utils.h"

#include "TDirectory.h"

#include <vector>
#include <string>


class MinimizerBase : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  MinimizerBase() = default;

  /// Local RTTI
  [[nodiscard]] virtual std::string getMinimizerTypeName() const { return "MinimizerBase"; };
  [[nodiscard]] virtual bool isFitHasConverged() const = 0;

  virtual void minimize() = 0;
  virtual void calcErrors() = 0;

  virtual void scanParameters(TDirectory* saveDir_);

  /// Set if the calcErrors method should be called by the FitterEngine.
  void setEnablePostFitErrorEval(bool enablePostFitErrorEval_) {_enablePostFitErrorEval_ = enablePostFitErrorEval_;}
  [[nodiscard]] bool isEnablePostFitErrorEval() const {return _enablePostFitErrorEval_;}

protected:

  [[nodiscard]] const LikelihoodInterface& getLikelihood() const;

  // Get the convergence monitor that is maintained by the likelihood
  // interface.  A local convenience function to get the convergence monitor.
  // The monitor actually lives in the likelihood).
  GenericToolbox::VariablesMonitor &getConvergenceMonitor();

  // Get the vector of parameters being fitted.  This is a local convenience
  // function to get the vector of fit parameter pointers.  The actual vector
  // lives in the likelihood.
  std::vector<Parameter *> &getMinimizerFitParameterPtr();

  // Print a table of the fitting parameters.
  void printMinimizerFitParameters();

private:
  /// Save a copy of the address of the engine that owns this object.
  FitterEngine* _owner_{nullptr};

  bool _enablePostFitErrorEval_{true};

};

#endif //GUNDAM_MinimizerBase_h

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
