#ifndef COLOROUTPUT_HH
#define COLOROUTPUT_HH

#include <cmath>
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

const unsigned int rainbow_size = 6;
const unsigned int rainbow_code[rainbow_size] = {31, 32, 33, 34, 35, 36};

#ifndef NOCOLOR
const std::string RESET_STR("\033[0m");
const std::string BOLD_STR("\033[1m");
const std::string RED_STR("\033[31m");
const std::string GREEN_STR("\033[32m");
const std::string YELLOW_STR("\033[33m");
const std::string BLUE_STR("\033[34m");
const std::string MAGENTA_STR("\033[35m");
const std::string CYAN_STR("\033[36m");
#else
const std::string RESET_STR("");
const std::string BOLD_STR("");
const std::string RED_STR("");
const std::string GREEN_STR("");
const std::string YELLOW_STR("");
const std::string BLUE_STR("");
const std::string MAGENTA_STR("");
const std::string CYAN_STR("");
#endif

template<typename T>
std::string ColorText(const T& text, unsigned int c)
{
    std::stringstream ss;
#ifndef NOCOLOR
    ss << "\033[" << c;
#endif
    ss << "m" << text;
#ifndef NOCOLOR
    ss << RESET_STR;
#endif
    return ss.str();
}

template<typename T>
std::string RainbowText(const T& text)
{
    std::stringstream ss;
    ss << text;

    std::string text_str = ss.str();

    ss.str("");
    unsigned int i = 0;
    unsigned int char_count = 0;
    unsigned int char_per_color = std::ceil(text_str.length() / (rainbow_size * 1.0));
    for(const auto& c : text_str)
    {
#ifndef NOCOLOR
        if(char_count == char_per_color || char_count == 0)
        {
            ss << BOLD_STR << "\033[" << rainbow_code[i++] << "m";
            char_count = 0;
        }
#endif

        ss << c;
        char_count++;
    }
#ifndef NOCOLOR
    ss << RESET_STR;
#endif

    return ss.str();
}
} // namespace color
#endif
