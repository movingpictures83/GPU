
add_lab("HeatEquationLibs")
add_lab_template("HeatEquationLibs" ${CMAKE_CURRENT_LIST_DIR}/template.cu)
add_lab_solution("HeatEquationLibs" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_lab_template_libs("HeatEquationLibs" cublas)
add_lab_template_libs("HeatEquationLibs" cusparse)
add_lab_solution_libs("HeatEquationLibs" cublas)
add_lab_solution_libs("HeatEquationLibs" cusparse)
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/solution.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -UGPUTK_USE_CUDA")
