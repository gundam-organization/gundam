//
// Created by Nadrino on 24/06/2025.
//

#ifndef FITSEQUENCER_H
#define FITSEQUENCER_H

#include "FitTask.h"

#include "ConfigUtils.h"

#include <vector>

class FitSequencer : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

private:
  std::vector<FitTask> _taskList_{};

  // shared among the tasks
  std::unique_ptr<MinimizerBase> _minimizer_{};

};



#endif //FITSEQUENCER_H
