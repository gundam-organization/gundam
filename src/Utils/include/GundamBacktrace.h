#ifndef GUNDAM_Backtrace_h_seen
#define GUNDAM_Backtrace_h_seen

#include <ostream>

// Backtrace depends on execinfo.h being available.  This is only available on
// linux.
//
// Note: Rumor has it that this can be tested using CMake (FindBacktrace), but
// that has not been well tested (remove this comment if it gets implemented)!
//
// Note: Rumor has it that Macs have it after Mac OS X 10.5, but I can't test
// that so for now it's not included.
//
// Note: This is a candidate for the generic tool kit, or simple cpp logger.
#if defined(__linux__)
#include <execinfo.h>
#include <cstdlib>
#else
// Backtrace is not available on non-linux builds
#endif

namespace GundamUtils {
    // When possible, print a backtrace for the current call stack to the
    // standard output stream.
    template<class CharT, class Traits>
    std::basic_ostream<CharT,Traits>&
    Backtrace(std::basic_ostream<CharT, Traits>& out) {
#if defined(__linux__)
        void *buffer[100];
        int nptrs = backtrace(buffer,100);
        out << "Backtrace of " << nptrs << " calls" << std::endl;

        // This uses malloc to allocate memory for strings, and passes
        // ownership to the caller.
        char **strings = backtrace_symbols(buffer,nptrs);
        if (strings == nullptr) {
            out << "Error generating backtrace" << std::endl;
            return out;
        }

        for (int i=0; i< nptrs; ++i) out << strings[i] << std::endl;

        // Free the strings pointer that was allocated with malloc.
        std::free(strings);
#else
        out << "Backtrace not available" << std::endl;
#endif
        return out;
    }
};

#endif
