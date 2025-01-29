
set( LIB_NAME "PythonBinder" )

set( SRC_FILE_LIST
    src/PythonBinder.cpp
)

set( PYBIND11_PYTHON_VERSION 3.11.6 )

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module( ${LIB_NAME} ${SRC_FILE_LIST} )

target_include_directories( ${LIB_NAME} PUBLIC
    ${PYBIND11_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions( ${LIB_NAME} PUBLIC
    VERSION_INFO=${GUNDAM_FULL_VERSION_STR}
)

target_link_libraries( ${LIB_NAME} PUBLIC
    GundamFitter
)

#target_include_directories(${LIB_NAME} PUBLIC
#    ${CMAKE_CURRENT_SOURCE_DIR}/include)
#
#target_link_libraries( ${LIB_NAME} PUBLIC
#    ${ROOT_LIBRARIES}
#)
#
install( TARGETS ${LIB_NAME} DESTINATION lib )



