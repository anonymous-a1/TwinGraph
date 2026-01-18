CMAKE_MINIMUM_REQUIRED(VERSION 3.25)
SET(CMAKE_REQUIRED_QUIET TRUE)
PROJECT("search")
SET(CMAKE_CXX_STANDARD 20)

# Define configurable variables with default values
SET(CMAKE_MAX_DIM 8192 CACHE STRING "Maximum dimension for compilation")
SET(CMAKE_NO_PREFETCH OFF CACHE STRING "Disable all prefetch optimizations")
SET(CMAKE_DEBUG_VERBOSE OFF CACHE STRING "Print more information at runtime")

ADD_DEFINITIONS(-DMAXDIM=${CMAKE_MAX_DIM})
MESSAGE(STATUS "MAX DIM is set to ${CMAKE_MAX_DIM}")

IF(CMAKE_NO_PREFETCH)
    ADD_DEFINITIONS(-DNOPREFETCH)
    MESSAGE(STATUS "CMAKE_NO_PREFETCH: all prefetching disabled")
ENDIF()
IF(CMAKE_DEBUG_VERBOSE)
    ADD_DEFINITIONS(-DDEBUG_VERBOSE)
    MESSAGE(STATUS "CMAKE_DEBUG_VERBOSE: using debug verbose")
ENDIF()

# Detect the architecture
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    MESSAGE(STATUS "Detected ARM architecture")

    # Check for SVE support
    EXECUTE_PROCESS(
        COMMAND lscpu
        OUTPUT_VARIABLE lscpu_output
        ERROR_VARIABLE lscpu_error
        RESULT_VARIABLE lscpu_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    IF(lscpu_result EQUAL 0 AND lscpu_output MATCHES "sve")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve")
        MESSAGE(STATUS "SVE support detected, using -march=armv8-a+sve")
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
        MESSAGE(STATUS "SVE not supported, using -march=armv8-a")
    ENDIF()  
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386")
    MESSAGE(STATUS "Detected x86 architecture")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ELSE()
    MESSAGE(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

# Include Boost
IF(POLICY CMP0167)
    CMAKE_POLICY(SET CMP0167 NEW)
ENDIF()
FIND_PACKAGE(Boost 1.71 REQUIRED COMPONENTS timer chrono system program_options)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES("../include")

# Set C++ Compiler Flags  
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -lrt -fPIC -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -ftemplate-depth=16384 -fpermissive -fconcepts -w")

# Executables and Linking
FILE(GLOB CPP_FILES "*.cpp")
FOREACH(src ${CPP_FILES})
  GET_FILENAME_COMPONENT(execName ${src} NAME_WE)
  ADD_EXECUTABLE(${execName} ${src})
  TARGET_LINK_LIBRARIES(${execName} ${Boost_LIBRARIES})
ENDFOREACH()
