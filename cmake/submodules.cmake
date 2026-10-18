
# SubModules: These are just adding the code directly, as stand-alone projects.

message("")
cmessage( WARNING "Checking submodules..." )

function( checkSubmodule )

  list( GET ARGV 0 SELECTED_SUBMODULE )
  cmessage( STATUS "Checking submodule: ${SELECTED_SUBMODULE}" )

  file( GLOB FILES_IN_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/submodules/${SELECTED_SUBMODULE}/*")

  if( FILES_IN_DIRECTORY )
    cmessage( STATUS "Git submodule ${SELECTED_SUBMODULE} is present" )
  else()
    cmessage( ERROR "Git submodule ${SELECTED_SUBMODULE} is not present, please checkout: \"git submodule update --init --remote --recursive\"" )
    cmessage( FATAL_ERROR "CMake fatal error." )
  endif()

endfunction( checkSubmodule )

checkSubmodule( cpp-generic-toolbox )
checkSubmodule( simple-cpp-logger )
checkSubmodule( simple-cpp-cmd-line-parser )

## Add the CmdLineParser
# Reproduce needed parts of the simple-cpp-cmd-line-parser CMakeLists.txt
include_directories(submodules/simple-cpp-cmd-line-parser/include)
add_definitions( -D CMDLINEPARSER_YAML_CPP_ENABLED=1 )

## Configure the GenericToolbox
if(NOT MSVC)
  add_definitions( "-DPROGRESS_BAR_FILL_TAG=\"GUNDAM\"" )
endif()
if (ENABLE_COLOR_OUTPUT)
  add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=1 )
else (ENABLE_COLOR_OUTPUT)
  # add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=0 )
  add_definitions( -D CPP_GENERIC_TOOLBOX_NOCOLOR )
endif (ENABLE_COLOR_OUTPUT)
if( ENABLE_BATCH_MODE )
  add_definitions( -D CPP_GENERIC_TOOLBOX_BATCH )
endif( ENABLE_BATCH_MODE )

## Add the Logger
add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
add_definitions( -D LOGGER_PREFIX_LEVEL=3 )

if(MSVC)
  # Keep string literals with spaces out of MSVC's /D arguments.
  # _DEBUG in the header selects the actual Visual Studio configuration.
  set(MSVC_LOGGER_DEV_MODE 0)
  if(ENABLE_DEV_MODE)
    set(MSVC_LOGGER_DEV_MODE 1)
  endif()
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/MsvcCompat.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/MsvcCompat.h"
    @ONLY
  )
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_BINARY_DIR}/generated/MsvcCompat.h>")
else()
  add_definitions( "-DLOGGER_TIME_FORMAT=\"%Y.%m.%d %H:%M:%S\"" )
  if( ${CMAKE_BUILD_TYPE} MATCHES DEBUG OR ${ENABLE_DEV_MODE} )
    cmessage( STATUS "Logger set in DEBUG mode." )
    add_definitions( "-DLOGGER_PREFIX_FORMAT=\"{TIME} {SEVERITY} {FILELINE}\"" )
  else()
    cmessage( STATUS "Logger set in RELEASE mode." )
    add_definitions( "-DLOGGER_PREFIX_FORMAT=\"{TIME} {SEVERITY} {FILENAME}\"" )
  endif()
endif()

if(NOT ENABLE_COLOR_OUTPUT)
  cmessage( STATUS "Color output is disabled." )
  add_definitions( -D LOGGER_ENABLE_COLORS=0 )
  add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
else()
  add_definitions( -D LOGGER_ENABLE_COLORS=1 )
  add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
endif()


# GUNDAM uses the compiled engine; the facade retains per-source-file settings.
set(SIMPLE_CPP_LOGGER_COMPILED ON)
if(USE_STATIC_LINKS)
  set(SIMPLE_CPP_LOGGER_BUILD_SHARED OFF)
else()
  set(SIMPLE_CPP_LOGGER_BUILD_SHARED ON)
endif()
set(SIMPLE_CPP_LOGGER_BUILD_TESTS OFF)
add_subdirectory(submodules/simple-cpp-logger)

# Build independent toolbox modules and enable the extensions used by GUNDAM.
set(GENERIC_TOOLBOX_COMPILED ON)
set(GENERIC_TOOLBOX_BUILD_SHARED ON)
if(USE_STATIC_LINKS)
  set(GENERIC_TOOLBOX_BUILD_SHARED OFF)
endif()
set(GENERIC_TOOLBOX_WITH_ROOT ON)
set(GENERIC_TOOLBOX_WITH_JSON ON)
set(GENERIC_TOOLBOX_WITH_YAML ON)
set(GENERIC_TOOLBOX_INSTALL_PACKAGE OFF)
set(GENERIC_TOOLBOX_BUILD_EXAMPLES OFF)
set(GENERIC_TOOLBOX_BUILD_TESTS OFF)
add_subdirectory(submodules/cpp-generic-toolbox)

# ROOT also ships nlohmann headers. Prefer GUNDAM's selected JSON package
# when compiling the toolbox so its exported functions use the same ABI.
target_include_directories(GenericToolboxJson BEFORE PUBLIC
  "$<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>")
