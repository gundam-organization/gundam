//
// Created by Adrien BLANCHET on 15/12/2021.
//

#ifndef GUNDAM_GUNDAMGREETINGS_H
#define GUNDAM_GUNDAMGREETINGS_H


#include <string>


class GundamGreetings {

public:
  GundamGreetings();
  virtual ~GundamGreetings();

  void setAppName(const std::string &appName);

  void hello();
  void goodbye();

  static bool isNewerOrEqualVersion(const std::string &appName);

private:
  std::string _appName_{};

};


#endif //GUNDAM_GUNDAMGREETINGS_H
