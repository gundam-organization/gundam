//
// Created by Adrien BLANCHET on 13/04/2022.
//

#include "DialBase.h"

#include "Logger.h"



const std::vector<double>& DialBase::getDialData() const {
    LogError << "getDialData not implemented for "
             << this->getDialTypeName()
             << std::endl;
#ifdef NDEBUG
    throw std::runtime_error("DialBase::getDialData not implemented for "
                             + this->getDialTypeName());
#endif
    static const std::vector<double> dummy;
    return dummy;
}

std::string DialBase::getDialTypeName() const { return {"DialBase"}; }
