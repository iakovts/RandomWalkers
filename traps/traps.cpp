#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <vector>

template <typename T>
using Vec = std::vector<T>;

template <typename T>
using Set = std::set<T>;
// std::mt19937 gen(4385);

Vec<int> random_choice(int dims);
void add_elementwise(Vec<int> &curr, std::vector<int> &rand_v);
int bound_elementwise(Vec<int> vec, int bound);
double calc_S_2D(int N);
void print_vec(Vec<int> &vec);
template <class RNG>
void set_traps(int dims, int traps_number, RNG &gen);
template <class RNG>
Vec<int> set_random(int dims, RNG &gen);
template <class RNG>
int walker(int dims, double c, int grid_size, RNG &gen);

int main(int argc, char **argv) {
  const int N = 100000, dims = 2, grid_size = 500;
  int i = 0, nthreads;
  double c = 0.001;
  std::array<int, N> output;
  std::array<double, N> Phi;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'C':
          sscanf(argv[i + 1], "%le", &c);
      }
    }
  }
  double fTimeStart = omp_get_wtime();
#pragma omp parallel
  { nthreads = omp_get_num_threads(); }
  std::mt19937 generator[nthreads];
  for (int j = 0; j < nthreads; j++) {
    generator[j] = std::mt19937(4385 * j);  // Each thread gets a generator
  }

  int tid;
#pragma omp parallel shared(output)
  {
    tid = omp_get_thread_num();
#pragma omp for
    for (i = 0; i < N; i++) {
      output[i] = walker(dims, c, 500, generator[tid]);
    }
  }
  for (int x = 0; x < (int)output.size(); x++) {
    Phi[x] = (double)output[x] / pow(grid_size, 2);
  }
  std::ostringstream filename;
  int ccc;
  if (c == 0.01) {
    ccc = 1;
  } else {
    ccc = 2;
  }
  double fTimeEnd = omp_get_wtime();
  filename << "survival" << ccc << ".txt";
  std::cout << "c = " << c <<
    ". Writing results to file: " << filename.str() <<
    ". Time Elapsed : " << (fTimeEnd - fTimeStart) << " s";
  std::ofstream fil;
  fil.open(filename.str());
  for (auto ele : Phi) {
    fil << ele << " ";
  }
  std::cout << "\n";
  fil.close();
}

double calc_S_2D(int N) {
  double S;
  S = (M_PI * N) / (logf(N));
  return S;
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

template <class RNG>
Vec<int> random_choice(int dims, RNG &gen) {
  std::discrete_distribution<int> distrib{1, 0, 1};
  Vec<int> res(dims);
  for (int x = 0; x < (int)dims; x++) {
    // Distrib range is [0, 1, 2] so subtract one with weights {1, 0, 1} to get
    // -1 or 1;
    res[x] = distrib(gen) - 1;
  }
  return res;
}

template <class RNG>
Vec<int> set_random(int dims, RNG &gen) {
  std::uniform_int_distribution<> distrib(1, 500);
  Vec<int> res(dims);
  for (int x = 0; x < dims; x++) {
    res[x] = distrib(gen);
    // std::cout << res[x] << " ";
  }
  return res;
};

void print_vec(Vec<int> &vec) {
  for (auto x : vec) {
    std::cout << x << " ";
  }
  std::cout << "\n";
}

template <class RNG>
int walker(int dims, double c, int grid_size, RNG &gen) {
  static int traps_number = int(c * pow(grid_size, 2));
  static int max_sanity = pow(grid_size, 2);  // Break loop condition
  int counter = 0;
  Set<Vec<int>> traps;
  Vec<int> curr_pos = set_random(2, gen);  // Initial pos is random
  Vec<int> rc(2);
  Vec<int> new_pos = curr_pos;
  while ((int)traps.size() < traps_number) {
    traps.insert(set_random(2, gen));
  }
  while (counter < max_sanity) {
    rc = random_choice(2, gen);
    add_elementwise(new_pos, rc);
    if (bound_elementwise(new_pos, 0) &&
        bound_elementwise(new_pos, grid_size)) {
      curr_pos = new_pos;
      if (traps.find(new_pos) != traps.end()) {
        break;
      } else {
        counter++;
      }
    } else {
      new_pos = curr_pos;
    }
  }
  return counter;
}
