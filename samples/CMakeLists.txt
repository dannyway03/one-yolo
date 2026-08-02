
add_executable(det  "det.cpp")
target_link_libraries(det  ${PROJECT_NAME})

add_executable(cls  "cls.cpp")
target_link_libraries(cls  ${PROJECT_NAME})

add_executable(seg  "seg.cpp")
target_link_libraries(seg  ${PROJECT_NAME})

add_executable(pose "pose.cpp")
target_link_libraries(pose ${PROJECT_NAME})

add_executable(obb  "obb.cpp")
target_link_libraries(obb  ${PROJECT_NAME})
