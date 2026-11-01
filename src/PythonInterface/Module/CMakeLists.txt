set(Python3_EXECUTABLE "${pythonCandidate}")
find_package(Python3 ${pythonVersion} EXACT QUIET COMPONENTS Interpreter Development.Module)
if(NOT Python3_FOUND)
  if(GUNDAM_PYTHON_VERSIONS STREQUAL "AUTO")
    cmessage(STATUS "Skipping Python ${pythonVersion} (${pythonCandidate}): matching development files unavailable.")
    return()
  endif()
  cmessage(FATAL_ERROR "Python ${pythonVersion} (${pythonCandidate}) requires matching development files: ${Python3_NOT_FOUND_MESSAGE}")
endif()

set(pythonTarget "GUNDAM_python${pythonVersionTag}")
Python3_add_library(${pythonTarget} MODULE WITH_SOABI ../src/PythonInterface.cpp)
target_link_libraries(${pythonTarget} PRIVATE pybind11::headers GundamFitter)
target_include_directories(${pythonTarget} BEFORE PRIVATE ../include)
target_compile_definitions(${pythonTarget} PRIVATE VERSION_INFO=${GUNDAM_FULL_VERSION_STR})
set_target_properties(${pythonTarget} PROPERTIES
    OUTPUT_NAME GUNDAM
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
add_dependencies(GUNDAM ${pythonTarget})
install(TARGETS ${pythonTarget} DESTINATION lib)
list(APPEND pythonBuiltVersions "${pythonVersion}")
set(pythonBuiltVersions "${pythonBuiltVersions}" PARENT_SCOPE)
cmessage(STATUS "Building ${pythonTarget} with ${Python3_EXECUTABLE}")
