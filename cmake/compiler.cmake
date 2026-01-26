

message("")
cmessage( WARNING "Configuring compiler options...")

# Changes default install path to be a subdirectory of the build dir.
# Should set the installation dir at configure time with
# -DCMAKE_INSTALL_PREFIX=/install/path
if(NOT DEFINED CMAKE_INSTALL_PREFIX
    OR CMAKE_INSTALL_PREFIX STREQUAL ""
    OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

cmessage( STATUS "CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")


# disable RPATH handling by CMake.
# Binaries and libs should be set manually (check the tutorial)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.5)
    cmessage(STATUS "Detected GCC version: ${CMAKE_CXX_COMPILER_VERSION}")
    cmessage(FATAL_ERROR "GCC version must be at least 5.0")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    cmessage( STATUS "Detected Clang version: ${CMAKE_CXX_COMPILER_VERSION}" )
    cmessage( FATAL_ERROR "Clang version must be at least 3.3" )
  endif()
else()
  cmessage( ALERT "You are using an untested compiler." )
endif()

# CXX standard is required and must match the version ROOT was compiled with.
set( CMAKE_CXX_STANDARD_REQUIRED ON )

set(IS_ROOT_CXX_STANDARD_SET OFF)
get_target_property(root_features ROOT::Core INTERFACE_COMPILE_FEATURES)
foreach(cxxstd IN ITEMS 23 20 17 14 11)
  if("cxx_std_${cxxstd}" IN_LIST root_features)
    cmessage(STATUS "ROOT compiled with C++${cxxstd}")
    set(IS_ROOT_CXX_STANDARD_SET ON)
    set(CMAKE_CXX_STANDARD ${cxxstd})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    break()
  endif()
endforeach()

if(!IS_ROOT_CXX_STANDARD_SET)
  # !! This is an old routine that use to work in 2025 !!
  # Explicitly set the compiler version so that it will match the
  # compiler that was used to compile ROOT.  Recent ROOT documentation
  # explicitly notes that the appliation needs to use the same C++
  # standard as ROOT.
  if ( ROOT_cxx14_FOUND )
    cmessage(STATUS "ROOT compiled with C++14")
    set(CMAKE_CXX_STANDARD 14)
  elseif ( ROOT_cxx17_FOUND )
    cmessage(STATUS "ROOT compiled with C++17")
    set(CMAKE_CXX_STANDARD 17)
  elseif ( ROOT_cxx20_FOUND )
    cmessage(STATUS "ROOT compiled with C++20")
    set(CMAKE_CXX_STANDARD 20)
  else ( ROOT_cxx14_FOUND )
    cmessage( ALERT "ROOT C++ standard not set, use ROOT minimum (C++14)")
    set(CMAKE_CXX_STANDARD 14)
  endif ( ROOT_cxx14_FOUND)
endif()

if(CXX_MARCH_FLAG)
endif()

if(CXX_WARNINGS)
  cmessage( STATUS "Enable CXX warnings" )
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>)
  # add_compile_options(-Wformat=0)
  # add_compile_options(-Wno-reorder)
endif()

if( NOT ENABLE_COLOR_OUTPUT )
  add_definitions( -D NOCOLOR )
endif()

if(ENABLE_TTY_CHECK)
  add_definitions(-D ENABLE_TTY_CHECK)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -O1 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3 -g")

if( CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
  # Try to enable AddressSanitizer automatically if the toolchain supports it

  include(CheckCXXSourceCompiles)

  # Save CMake internals we are about to modify
  set(_saved_req_flags "${CMAKE_REQUIRED_FLAGS}")
  set(_saved_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")

  # We want to test a real executable (so linking must succeed, including libasan)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fsanitize=address")

  check_cxx_source_compiles(
      "int main() { return 0; }"
      HAVE_WORKING_ASAN
  )

  # Restore internals
  set(CMAKE_REQUIRED_FLAGS "${_saved_req_flags}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_saved_try_type}")

  if(HAVE_WORKING_ASAN)
    message(STATUS "AddressSanitizer detected, enabling for Debug builds.")

    # Enable globally (only in Debug, you can tweak the condition if you want)
    add_compile_options(
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    add_link_options(
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
  else()
    message(STATUS "AddressSanitizer not usable on this toolchain, keeping it disabled.")
  endif()

endif()

################################################################################
# CMake Generated
###############################################################################

cmessage( STATUS "C++ Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}" )
cmessage( STATUS "C++ Flags         : ${CMAKE_CXX_FLAGS}")
cmessage( STATUS "C++ Standard      : ${CMAKE_CXX_STANDARD}" )
cmessage( STATUS "C++ Release flags : ${CMAKE_CXX_FLAGS_RELEASE}" )
cmessage( STATUS "C++ Debug flags   : ${CMAKE_CXX_FLAGS_DEBUG}" )
cmessage( STATUS "Build type        : ${CMAKE_BUILD_TYPE}" )

