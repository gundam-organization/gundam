#ifndef PROGRESSBAR_HH
#define PROGRESSBAR_HH

#include <iostream>
#include <string>

#ifdef TTYCHECK
#include <unistd.h>
#endif

#include "ColorOutput.hh"

class ProgressBar
{
private:
    bool is_rainbow;
    double max_progress;
    unsigned int bar_size;
    std::string bar_char;
    std::string bar_prefix;

public:
    ProgressBar()
        : ProgressBar(60, std::string("#"))
    {
    }

    ProgressBar(unsigned int len)
        : ProgressBar(len, std::string("#"))
    {
    }

    ProgressBar(unsigned int len, const char* ch)
        : ProgressBar(len, std::string(ch))
    {
    }

    ProgressBar(unsigned int len, const std::string& ch)
        : is_rainbow(false)
        , max_progress(1.0)
        , bar_size(len)
        , bar_char(ch)
        , bar_prefix("")
    {
    }

    void SetBarLength(unsigned int len) { bar_size = len; }
    void SetBarChar(const std::string& ch) { bar_char = ch; }
    void SetBarChar(const char* ch) { bar_char = ch; }
    void SetRainbow(bool flag = true) { is_rainbow = flag; }
    void SetPrefix(const std::string& prefix) { bar_prefix = prefix; }
    void SetMaxProgress(const double val) { max_progress = val; }

    template<typename T>
    void Print(T current)
    {
        Print(bar_prefix, current, max_progress);
    }

    template<typename T>
    void Print(const std::string& prefix, T current)
    {
        Print(prefix, current, max_progress);
    }

    template<typename T, typename U>
    void Print(T current, U limit)
    {
        Print(bar_prefix, current, limit);
    }

    template<typename T, typename U>
    void Print(const std::string& prefix, T current, U limit)
    {
        std::string bar;
        float progress = static_cast<float>(current) / limit;
        int pos        = bar_size * progress;
        for(int b = 0; b < bar_size; ++b)
        {
            if(b < pos)
                bar.replace(b, 1, bar_char);
            else
                bar.replace(b, 1, " ");
        }
        
        #ifdef TTYCHECK
        if(isatty(fileno(stdout)))
        {
            std::cout << "\r" << prefix << "[" << (is_rainbow ? color::RainbowText(bar) : bar) << "] "
                      << int(progress * 100.0) << "%";
        }
        else
        {
            std::cout << " " << int(progress * 100.0) << "%";
        }
        #else
            std::cout << "\r" << prefix << "[" << (is_rainbow ? color::RainbowText(bar) : bar) << "] "
                      << int(progress * 100.0) << "%";
        #endif


        if(current == limit)
            std::cout << std::endl;
        std::cout.flush();
    }
};

#endif
