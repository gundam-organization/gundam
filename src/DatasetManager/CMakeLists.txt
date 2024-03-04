#
# Define the data set manager lib
# This lib handles how data is loaded from the input root files to the samples and parameters
#
set( LIB_NAME GundamDatasetManager )

set( SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DataSetManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DatasetDefinition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EventTreeWriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DataDispenser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DataDispenserUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EventVarTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EventVarTransformLib.cpp
    )

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DataSetManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DatasetDefinition.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EventTreeWriter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DataDispenser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DataDispenserUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EventVarTransform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EventVarTransformLib.h
)

if( USE_STATIC_LINKS )
  add_library( ${LIB_NAME} STATIC ${SRCFILES} )
else()
  add_library( ${LIB_NAME} SHARED ${SRCFILES} )
endif()

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( ${LIB_NAME}
    PUBLIC
    GundamPropagator
)

#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install( TARGETS ${LIB_NAME} DESTINATION lib )
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

