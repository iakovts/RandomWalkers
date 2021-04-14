OUT_FILE=fast_walker

build:
	@g++ -g  -fsanitize=address -Wall -fopenmp -o $(OUT_FILE) fast_walker.cpp

run: build
	@for number in 1 2 3; do \
		./$(OUT_FILE) -D $$number; \
		done
clean:
	@rm -f walk_mean_*.txt
	@rm -f $(OUT_FILE)

