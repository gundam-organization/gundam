//
// Created by Nadrino on 24/06/2025.
//

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "MinimizerBase.h"

#include "ConfigUtils.h"

#include <memory>


class FitTask : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  void run();

private:
  std::unique_ptr<MinimizerBase> _minimizer_{nullptr};

};



#endif //SEQUENCE_H
