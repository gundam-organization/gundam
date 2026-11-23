//
// Created by Adrien BLANCHET on 13/04/2022.
//

/// @file DialBase.cpp
/// @brief Implements the default DialBase::getDialData(), the only
///        out-of-line member of the dial base class.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in DialBase.h. The
/// comments here describe how each method does its job.

#include "DialBase.h"

#include "Logger.h"


/// @par Implementation
/// Logs an error naming the dial type (through the virtual
/// getDialTypeName()). With NDEBUG defined it then throws. Otherwise it
/// returns a function-local static empty vector, so the returned reference
/// stays valid for the whole run.
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
