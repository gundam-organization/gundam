//
// Created by Adrien BLANCHET on 13/04/2022.
//

#include "DialBase.h"

#include "Logger.h"

#include <cmath>
#include <limits>


double DialBase::evalGradient(const DialInputBuffer& input_, int iInput_) const {
    LogThrowIf(iInput_ < 0 or iInput_ >= input_.getInputSize(),
               "Invalid input index " << iInput_ << " for " << this->getDialTypeName()
                                      << " with " << input_.getInputSize() << " inputs.");

    auto lowerInput{input_};
    auto upperInput{input_};
    const double x{input_.getInputBuffer()[iInput_]};
    const double step{std::sqrt(std::numeric_limits<double>::epsilon()) * (std::abs(x) + 1.)};

    lowerInput.getInputBuffer()[iInput_] = x - step;
    upperInput.getInputBuffer()[iInput_] = x + step;

    return (this->evalResponse(upperInput) - this->evalResponse(lowerInput)) / (2. * step);
}


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
