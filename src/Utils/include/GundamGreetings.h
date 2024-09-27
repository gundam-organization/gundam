//
// Created by Adrien BLANCHET on 15/12/2021.
//

#ifndef GUNDAM_GUNDAMGREETINGS_H
#define GUNDAM_GUNDAMGREETINGS_H

#include <string>


class GundamGreetings {

public:
  GundamGreetings() = default;
  virtual ~GundamGreetings() = default;

  void setAppName(const std::string &appName){ _appName_ = appName; }

  void hello();
  void goodbye();

private:
  std::string _appName_{};

};


#endif //GUNDAM_GUNDAMGREETINGS_H
