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

class Base{
public:
  virtual ~Base() = default;
  Base() = default;
  virtual std::unique_ptr<Base> clone() = 0;
  virtual std::string name(){ return "B"; }
};
class Derivative1: public Base{
public:
  ~Derivative1() override = default;
  Derivative1() = default;
  std::unique_ptr<Base> clone() override { return std::make_unique<Derivative1>(*this); }
  std::string name() override { return "D1"; }
};
class Derivative2: public Base{
public:
  ~Derivative2() override = default;
  Derivative2() = default;
  std::unique_ptr<Base> clone() override { return std::make_unique<Derivative2>(*this); }
  std::string name() override { return "D2"; }
};

class Wrapper{
public:
  ~Wrapper() = default;
  Wrapper() = default;
  explicit Wrapper(std::unique_ptr<Base> def): m{std::move(def)} {};
  Wrapper(const Wrapper& src_): m{src_.m->clone()} {  }

  Base* operator->() const { return m.get(); }

  std::unique_ptr<Base> m;
};

class Host{
public: std::vector<Wrapper> _derivativeList_{};
};

class Owner{
public: std::vector<Host> _hostList_;
};

int main(int argc, char** argv){
  Owner o;
  o._hostList_.resize(10);

  Host& h = o._hostList_[0];
  h._derivativeList_.emplace_back(std::make_unique<Derivative1>());

  Wrapper w = h._derivativeList_[0];

  std::cout << "NAME IS " << w->name() << std::endl;

  Owner o2(o);
}


