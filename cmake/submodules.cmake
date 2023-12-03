
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
if(yaml-cpp_FOUND)
  add_definitions( -DCMDLINEPARSER_YAML_CPP_ENABLED=1 )
endif()

## Add the GenericToolbox²
# Reproduce needed parts of the cpp-generic-toolbox CMakeLists.txt
include_directories(submodules/cpp-generic-toolbox/include)

#file( GLOB CPP_GENERIC_TOOLBOX_HEADERS ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/*.h )
#file( GLOB CPP_GENERIC_TOOLBOX_HEADERS_IMPL ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/implementation/*.h )
#install(FILES ${CPP_GENERIC_TOOLBOX_HEADERS} DESTINATION include)
#install(FILES ${CPP_GENERIC_TOOLBOX_HEADERS_IMPL} DESTINATION include/implementation)

add_definitions( -D PROGRESS_BAR_FILL_TAG="\\\"GUNDAM"\\\" )
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
# Reproduce needed parts of the simple-cpp-logger CMakeLists.txt
include_directories(submodules/simple-cpp-logger/include)
add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
add_definitions( -D LOGGER_TIME_FORMAT="\\\"%d/%m/%Y %H:%M:%S"\\\" )

if(${CMAKE_BUILD_TYPE} MATCHES Debug OR ${ENABLE_DEV_MODE})
  cmessage( STATUS "Logger set in dev mode." )
  add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER} {FILELINE}"\\\" )
else()
  cmessage( STATUS "Logger set in release mode." )
  add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER}"\\\" )
endif()

if(NOT ENABLE_COLOR_OUTPUT)
  cmessage( STATUS "Color output is disabled." )
  add_definitions( -D LOGGER_ENABLE_COLORS=0 )
  add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
else()
  add_definitions( -D LOGGER_ENABLE_COLORS=1 )
  add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
endif()


