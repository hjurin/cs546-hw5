.PHONY: default all clean mrproper strats

default: build

build: serial cuda

serial:
	gcc matrixNorm.c -o serial

cuda:
	nvcc matrixNorm_1.cu -o cuda1
	nvcc matrixNorm_2.cu -o cuda2
	nvcc matrixNorm_3.cu -o cuda3

test:
	@chmod +x *.sh
	./test.sh

mrproper:
	rm -rf cuda*
	rm -rf serial*
	rm -rf *~
	rm -rf \#*
