
set( LIB_NAME "PythonBinder" )

set( SRCFILES
    src/PythonBinder.cpp
)

set( PYBIND11_PYTHON_VERSION 3.11.6 )

find_package(pybind11 REQUIRED)

pybind11_add_module(${LIB_NAME} ${SRCFILES})

target_include_directories(${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(${LIB_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${LIB_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

#target_include_directories(${LIB_NAME} PUBLIC
#    ${CMAKE_CURRENT_SOURCE_DIR}/include)
#
#target_link_libraries( ${LIB_NAME} PUBLIC
#    ${ROOT_LIBRARIES}
#)
#
#install( TARGETS ${LIB_NAME} DESTINATION lib )



