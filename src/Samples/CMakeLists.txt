# Samples
set( LIB_NAME GundamSamples )

set( SRCFILES
        src/SampleElement.cpp
    src/SampleSet.cpp
    src/Sample.cpp
        src/PhysicsEvent.cpp
        src/PlotGenerator.cpp
        src/JointProbability.cpp
        src/EventVarTransform.cpp
        src/EventVarTransformLib.cpp
        )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(
        ${LIB_NAME} PUBLIC
        GundamUtils
        GundamParameters
        ${ROOT_LIBRARIES}
        -ldl
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)

