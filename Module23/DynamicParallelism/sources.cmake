
add_lab("DynamicParallelism")
add_lab_template("DynamicParallelism" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("DynamicParallelism" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("DynamicParallelism" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
