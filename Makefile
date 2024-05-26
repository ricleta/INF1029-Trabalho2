CC = nvcc 

SRCS = matrix_lib.cu matrix_lib_test.cu
HEADERS = $(shell find . -name '.ccls-cache' -type d -prune -o -type f -name '*.h' -print)

main: $(SRCS) $(HEADERS)
	$(CC) $(SRCS) -o matrix_lib_test

clean:
	rm -f main