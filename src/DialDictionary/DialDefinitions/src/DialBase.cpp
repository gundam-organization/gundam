//
// Created by Adrien BLANCHET on 13/04/2022.
//

#include "DialBase.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialBase]");
});

const std::vector<double>& DialBase::getDialData() const {
    throw std::runtime_error("getDialData not implemented for "
                             + this->getDialTypeName());
    static const std::vector<double> dummy;
    return dummy;
}

std::string DialBase::getDialTypeName() const { return {"DialBase"}; }
