//
//
// Implementation of the iterative Jacobi method.
//
// Given a known, diagonally dominant matrix A and a known vector b, we aim to
// to find the vector x that satisfies the following equation:
//
//     Ax = b
//
// We first split the matrix A into the diagonal D and the remainder R:
//
//     (D + R)x = b
//
// We then rearrange to form an iterative solution:
//
//     x' = (b - Rx) / D
//
// More information:
// -> https://en.wikipedia.org/wiki/Jacobi_method
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

static int N;
static int MAX_ITERATIONS;
static int SEED;
static float CONVERGENCE_THRESHOLD;

#define SEPARATOR "------------------------------------\n"

// Return the current time in seconds since the Epoch
double get_timestamp();

// Parse command line arguments to set solver parameters
void parse_arguments(int argc, char *argv[]);

// Run the Jacobi solver
// Returns the number of iterations performed
int run(float *A, float *b, float *x, float *xtmp)
{
  int itr;
  int row, col;
  float diff;
  float sqdiff;
  float *ptrtmp;
  float thresh = CONVERGENCE_THRESHOLD*CONVERGENCE_THRESHOLD;
  // Loop until converged or maximum iterations reached
  //change cores

  
  itr = 0;
  do
  {
    sqdiff = 0.0;
    #pragma omp parallel for private(row,col)
    for (row = 0; row < N; row++)
    {
      float dot = 0.0;
      for (col = 0; col < N; col++)
      {
          dot += A[row*N + col] * x[col];
      }
      dot -= A[row*N + row] * x[row];
      xtmp[row] = (b[row] - dot) / A[row + row*N];
    }
    
    // Swap pointers
    ptrtmp = x;
    x      = xtmp;
    xtmp   = ptrtmp;
    //discuss increase solution error, try change to doubles
    #pragma omp parallel for private(diff, row) reduction(+:sqdiff)
    for (row = 0; row < N; row++) {
      diff    = xtmp[row] - x[row];
      sqdiff += diff * diff;
    }

    itr++;
  } while ((itr < MAX_ITERATIONS) && (sqdiff > thresh));


  return itr;
}

int main(int argc, char *argv[])
{
  parse_arguments(argc, argv);

  float *A    = malloc(N*N*sizeof(float));
  float *b    = malloc(N*sizeof(float));
  float *x    = malloc(N*sizeof(float));
  float *xtmp = malloc(N*sizeof(float));

  printf(SEPARATOR);
  printf("Matrix size:            %dx%d\n", N, N);
  printf("Maximum iterations:     %d\n", MAX_ITERATIONS);
  printf("Convergence threshold:  %lf\n", CONVERGENCE_THRESHOLD);
  printf(SEPARATOR);

  double total_start = get_timestamp();
  int row;
  int col;
  float *randomA = malloc(N*N*sizeof(float));
  float *randomB = malloc(N*sizeof(float));

  for(int row = 0; row < N; row++){
      for(int col = 0; col < N; col++){
	 randomA[row*N + col] = rand()/(float)RAND_MAX;
      }
      randomB[row] = rand()/(float)RAND_MAX;
}
  // Initialize data
  srand(SEED);
  #pragma omp parallel for private(row,col)
  for (row = 0; row < N; row++)
  {
    float rowsum = 0.0;
    for (col = 0; col < N; col++)
    {
      float value = randomA[row*N + col];
      A[row*N + col] = value;
      rowsum += value;
    }
    A[row + row*N] += rowsum;
    b[row] = randomB[row];
    x[row] = 0.0;
  }

  // Run Jacobi solver
  double solve_start = get_timestamp();
  int itr = run(A, b, x, xtmp);
  double solve_end = get_timestamp();

  // Check error of final solution
  float err = 0.0;
  #pragma omp parallel for private(row,col) reduction(+:err)
  for (row = 0; row < N; row++)
  {
    float tmp = 0.0;
    for (col = 0; col < N; col++)
    {
      tmp += A[row*N + col] * x[col];
    }
    tmp = b[row] - tmp;
    err += tmp*tmp;
  }
  err = sqrt(err);

  double total_end = get_timestamp();

  printf("Solution error = %lf\n", err);
  printf("Iterations     = %d\n", itr);
  printf("Total runtime  = %lf seconds\n", (total_end-total_start));
  printf("Solver runtime = %lf seconds\n", (solve_end-solve_start));
  if (itr == MAX_ITERATIONS)
    printf("WARNING: solution did not converge\n");
  printf(SEPARATOR);

  free(A);
  free(b);
  free(x);
  free(xtmp);

  return 0;
}

double get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}

int parse_int(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

float parse_float(const char *str)
{
  char *next;
  float value = strtod(str, &next);
  return strlen(next) ? -1 : value;
}

void parse_arguments(int argc, char *argv[])
{
  // Set default values
  N = 1000;
  MAX_ITERATIONS = 20000;
  CONVERGENCE_THRESHOLD = 0.0001;
  SEED = 0;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--convergence") || !strcmp(argv[i], "-c"))
    {
      if (++i >= argc || (CONVERGENCE_THRESHOLD = parse_float(argv[i])) < 0)
      {
        printf("Invalid convergence threshold\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (MAX_ITERATIONS = parse_int(argv[i])) < 0)
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--norder") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (N = parse_int(argv[i])) < 0)
      {
        printf("Invalid matrix order\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--seed") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || (SEED = parse_int(argv[i])) < 0)
      {
        printf("Invalid seed\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./jacobi [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print this message\n");
      printf("  -c  --convergence  C     Set convergence threshold\n");
      printf("  -i  --iterations   I     Set maximum number of iterations\n");
      printf("  -n  --norder       N     Set maxtrix order\n");
      printf("  -s  --seed         S     Set random number seed\n");
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }
}
