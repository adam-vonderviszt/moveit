cmake_minimum_required(VERSION 2.8.3)
project(trajopt)

add_compile_options(-std=c++11 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  trajopt_sco
  trajopt_utils
  roscpp
  moveit_core
)

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system python thread program_options REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    trajopt_sco
    trajopt_utils
    roscpp
  DEPENDS
    EIGEN3
    Boost
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  SYSTEM ${EIGEN3_INCLUDE_DIRS}
  SYSTEM ${Boost_INCLUDE_DIRS}
  SYSTEM ${JSONCPP_INCLUDE_DIRS}
)

set(TRAJOPT_SOURCE_FILES
    src/trajectory_costs.cpp
    src/kinematic_terms.cpp
    src/problem_description.cpp
    src/collision_terms.cpp
    src/utils.cpp
)

add_library(${PROJECT_NAME} ${TRAJOPT_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${Boost_SYSTEM_LIBRARY} ${JSONCPP_LIBRARIES} ${catkin_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hxx"
 )
