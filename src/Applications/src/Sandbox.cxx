//
// Created by Nadrino on 17/06/2021.
//

#include "Logger.h"

#include "string"
#include "vector"
#include "memory"

LoggerInit([]{
  Logger::setUserHeaderStr("[Sandbox]");
})

class Base{};
class Derivative1: public Base{};
class Derivative2: public Base{};

class Host{
public: std::vector<std::shared_ptr<Base>> _derivativeList_{};
};

class Owner{
public: std::vector<Host> _hostList_;
};

int main(int argc, char** argv){
  Owner o;
  o._hostList_.resize(10);

  Host& h = o._hostList_[0];
  h._derivativeList_.emplace_back(std::make_shared<Derivative1>());
}


