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
  // config
  bool _isEnabled_{true};
  std::string _name_{};
  std::string _outputFolder_{};
  std::vector<std::string> _actionList_{};

};



#endif //SEQUENCE_H
