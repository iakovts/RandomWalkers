#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <random>
#include <set>

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Set = std::set<T>;

Vec<int> walker(int dims);
Vec<int> random_choice(int dims);
void add_elementwise(Vec<int> &curr, std::vector<int> &rand_v);
int bound_elementwise(Vec<int> vec, int bound);

int main(int argc, char **argv) {
  int j = 0, N = 10000, dims = 1;
  Vec<Vec<int>> all_walks(N);

  // Get number of dimensions from user input.
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'D':
          sscanf(argv[i + 1], "%d", &dims);
      }
    }
  }

  srand(4385);
  // std::mt19937 gen(4385);
  double fTimeStart = omp_get_wtime();
#pragma omp parallel shared(all_walks)
  {
#pragma omp for
    for (j = 0; j < N; j++) {
      all_walks[j] = walker(dims);
    }
  }
  Vec<double> sum_walk;

  // Mean value for `N` runs of the walker
  double curr_sum;
  for (int k = 0; k < 10; k++) {
    curr_sum = 0;
    for (int f = 0; f < N; f++) {
      curr_sum += all_walks[f][k];
      // std::cout << curr_sum << " ";
    }
    sum_walk.push_back(curr_sum / N);
  }

  double fTimeEnd = omp_get_wtime();
  std::cout << "Dims: " << dims << " Time Elapsed : " << (fTimeEnd - fTimeStart) << "s  Mean per 100: ";
  // Write output file.
  std::ostringstream filename;
  filename << "walk_mean_" << dims << ".txt";
  std::ofstream fil;
  fil.open(filename.str());
  for (auto ele : sum_walk) {
    fil << ele << " ";
    std::cout << ele << ", ";
  }
  std::cout << "\n";
  fil.close();
}

Vec<int> random_choice(int dims) {
  int r;
  int choices[2] = {-1, 1};
  Vec<int> res(dims);
  for (int x = 0; x < dims; x++) {
    r = rand() % 2;
    res[x] = choices[r];
  }
  return res;
}

void add_elementwise(Vec<int> &curr, Vec<int> &rand_v) {
  // Adds two vector of the same size element-wise.
  std::transform(curr.begin(), curr.end(), rand_v.begin(), curr.begin(),
                 std::plus<int>());
}

int bound_elementwise(Vec<int> vec, int bound) {
  // Checks if walker has reached bounds on any dimension
  return std::find(vec.begin(), vec.end(), bound) == vec.end();
}

Vec<int> walker(int dims) {
  /**
     A N-dimensional walker. This walker will move in any
     direction according to `dims`, and adds coordinates
     which were visited at least once to the `walk_total`
     set. Does nothing when revisiting a coord already
     counted.
   */
  static int N = 500;  // Grid Size
  Set<Vec<int>> walk_total;
  Vec<int> sum_grid(10);

  Vec<int> walk_sum(dims, N / 2);  // Start from origin.
  walk_total.insert(walk_sum);  // Account for origin.

  Vec<int> new_walk_sum(dims, N / 2);
  Vec<int> rc;
  int j = 0;
  for (int i = 0; i <= 1000; i++) {
    rc = random_choice(dims);
    add_elementwise(new_walk_sum, rc);

    // Boundary conditions checks.
    if (bound_elementwise(new_walk_sum, N) &&
        bound_elementwise(new_walk_sum, 0)) {
      walk_sum = new_walk_sum;
      // If the new vector is not already in the walk set, add it.
      walk_total.insert(walk_sum);
    } else {
      new_walk_sum = walk_sum;
    }

    // Every 100 steps count the total number of unique coords
    if (i % 100 == 0 && i > 0) {
      sum_grid[j] = walk_total.size();
      j++;
    }
  }
  return sum_grid;
}
