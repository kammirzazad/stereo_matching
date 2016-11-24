// ----------------------------------------------------------------------------------------
//    Tool to Evaluate Resulting Depth Labels for MEMOCODE Design Contest 2013
//
//    It reads output_labels.txt and ground_truth_labels.txt files.
//    These files contain depth labels produced by lbp.cpp and
//    the ground truth (reference) depth labels, respectivelly. 
//    The code calculates the number of label mismatches. 
// ----------------------------------------------------------------------------------------

#include <stdio.h>
#include <assert.h>

int main() {
    FILE *fp_ref;
    FILE *fp_test;

    int ref_lnum, test_lnum;
    int ref_label, test_label;
    int mismatch_cnt; 

    // Open files
    if( ( fp_ref = fopen( "ground_truth_labels.txt", "r" ) ) == NULL ) {
        printf("Can't open file\n"); 
        assert(0);
    } 
    if( ( fp_test = fopen( "output_labels.txt", "r" ) ) == NULL ) {
        printf("Can't open file\n"); 
        assert(0);
    } 
   
    // Read number of labels to compare
    fscanf(fp_ref, "%d", &ref_lnum);
    fscanf(fp_test, "%d", &test_lnum);
    assert(ref_lnum == test_lnum);

    // Count mismatches
    mismatch_cnt=0;
    for(int i=0; i<ref_lnum; i++) {
       fscanf(fp_ref, "%d", &ref_label);
       fscanf(fp_test, "%d", &test_label);

       if(ref_label != test_label) {
          mismatch_cnt++;
          //printf( "%d\n" , test_label-ref_label );
       } 
    }

    printf("There are %d mismatch out of %d labels\n", 
           mismatch_cnt, ref_lnum);

    return 0;
}

