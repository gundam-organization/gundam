#ifndef GUNDAM_CACHE_MANAGER_EXPORT_H
#define GUNDAM_CACHE_MANAGER_EXPORT_H

#if defined(_WIN32) && !defined(GUNDAM_CACHE_MANAGER_STATIC_DEFINE)
#if defined(GundamCacheManager_EXPORTS)
#define GUNDAM_CACHE_MANAGER_EXPORT __declspec(dllexport)
#else
#define GUNDAM_CACHE_MANAGER_EXPORT __declspec(dllimport)
#endif
#else
#define GUNDAM_CACHE_MANAGER_EXPORT
#endif

#endif
