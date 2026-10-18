#ifndef GUNDAM_UTILS_EXPORT_H
#define GUNDAM_UTILS_EXPORT_H

// CMake defines GundamUtils_EXPORTS while building the DLL, and propagates
// GUNDAM_UTILS_STATIC_DEFINE to consumers when building a static library.
#if defined(_WIN32) && !defined(GUNDAM_UTILS_STATIC_DEFINE)
#if defined(GundamUtils_EXPORTS)
#define GUNDAM_UTILS_EXPORT __declspec(dllexport)
#else
#define GUNDAM_UTILS_EXPORT __declspec(dllimport)
#endif
#else
#define GUNDAM_UTILS_EXPORT
#endif

#endif // GUNDAM_UTILS_EXPORT_H
