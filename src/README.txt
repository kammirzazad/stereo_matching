Software Reference for MEMOCODE 2013 Design Contest.

Eriko Nurvitadhi, Intel Science and Technology Center for Embedded Computing.

http://www.memocode-conference.com

This package contains the following:

- lbp.cpp: functional loopy belief propagation software reference 

- vdata_in.txt: test input vertex data(data cost) from tsukuba image

- ref_output_labels.txt: output labels from running compiled lbp.cpp 
  using vdata_in.txt as input. Provided here for reference.

- eval_results.cpp: tool to evaluate accuracy of resulting depth 
  labels relative to ground truth 

- ground_truth_labels.txt: tsukuba's ground truth depth labels

- Makefile
    "make" compiles lbp.cpp and eval_results.cpp 
    "make run-lbp" runs lbp using vdata_in.txt and produces output_labels.txt
    "make run-eval" calculates accuracy of output_labels.txt
    "make clean" removes lbp and eval_results executables


