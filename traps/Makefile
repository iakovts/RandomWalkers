OUT_FILE=traps

build:
	@g++ -g  -fsanitize=address -Wall -fopenmp -o $(OUT_FILE) traps.cpp

run: build
	@for number in 0.01 0.001; do \
		./$(OUT_FILE) -C $$number; \
		done
clean:
	@rm -f survival*.txt
	@rm -f $(OUT_FILE)
