GXX = g++
LIBS = #-lopencv_core -lopencv_highgui

all: general

general:
	$(GXX) -O3 -fopenmp lbp_MKSF.cpp -o lbp $(LIBS)
	$(GXX) eval_results.cpp -o eval_results

run-lbp:
	./lbp

run-eval:
	./eval_results

clean:
	rm lbp eval_results
