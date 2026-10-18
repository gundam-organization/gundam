set( LIB_NAME GundamFitter )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/FitterEngine.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/MinimizerBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/SimpleMcmc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/RootMinimizer.cpp
  )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME}
    BEFORE
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/include
)

if( WITH_ROOT_DICT_BUILD )
  set(CMAKE_CXX_FLAGS_DICT "-Wno-nonportable-include-path")
  ROOT_GENERATE_DICTIONARY(
    SimpleMcmc_Dict
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/include/SimpleMcmc.h
    MODULE ${LIB_NAME}
    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/SimpleMcmc_LinkDef.h
    OPTIONS "${CMAKE_CXX_FLAGS_DICT}"
  )

  # ROOT looks for its dictionary beside the library. Multi-configuration
  # generators put the DLL in a configuration subdirectory.
  if(CMAKE_CONFIGURATION_TYPES)
    add_custom_command(TARGET ${LIB_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter_rdict.pcm"
        "${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter.rootmap"
        "$<TARGET_FILE_DIR:${LIB_NAME}>"
      VERBATIM
    )
  endif()

  target_link_libraries(
      ${LIB_NAME}
      PUBLIC
      GundamStatisticalInference
      ${DICT_SIMPLEMCMC}
  )

  # Install the ROOT pcm and rootmap files
  install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter_rdict.pcm
      ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter.rootmap
      DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib
      COMPONENT
      libraries
  )
else()
  target_link_libraries(
      ${LIB_NAME}
      PUBLIC
      GundamStatisticalInference
  )
endif()

install(TARGETS ${LIB_NAME} DESTINATION lib)
