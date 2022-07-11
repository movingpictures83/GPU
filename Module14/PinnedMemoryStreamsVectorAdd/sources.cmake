add_lab("PinnedMemoryStreamsVectorAdd")
add_lab_template("PinnedMemoryStreamsVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("PinnedMemoryStreamsVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("PinnedMemoryStreamsVectorAdd" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
