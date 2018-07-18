#ifndef COLOROUTPUT_HH
#define COLOROUTPUT_HH

#include <iostream>
#include <sstream>
#include <string>

namespace color
{

enum code
{
    RESET      = 0,
    BOLD       = 1,
    FG_RED     = 31,
    FG_GREEN   = 32,
    FG_YELLOW  = 33,
    FG_BLUE    = 34,
    FG_MAGENTA = 35,
    FG_CYAN    = 36,
    FG_DEFAULT = 39
};

const std::string RESET_STR("\033[0m");
const std::string BOLD_STR("\033[1m");
const std::string RED_STR("\033[31m");
const std::string GREEN_STR("\033[32m");
const std::string YELLOW_STR("\033[33m");
const std::string BLUE_STR("\033[34m");
const std::string MAGENTA_STR("\033[35m");
const std::string CYAN_STR("\033[36m");

template<typename T>
std::string ColorText(const T& text, unsigned int c)
{
    std::stringstream ss;
    ss << "\033[" << c << "m" << text << RESET_STR;
    return ss.str();
}
} // namespace color
#endif
