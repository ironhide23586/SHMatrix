#include <vector>
#include <iostream>

#include "SHMatrix.h"

using namespace std;

int main() {
  cublasHandle_t cublasHandle;
  CublasSafeCall(cublasCreate_v2(&cublasHandle));

  SHMatrix a(cublasHandle, std::vector<int> { 3, 5 }, GPU);
  a.GaussianInit(); //Initializing a 3x5 matrix with random numbers from gaussian distribution.
  a.Print();

  SHMatrix b(cublasHandle, std::vector<int> { 3, 5 }, GPU);
  b.UniformInit(); //Initializing a 3x5 matrix with random numbers from uniform distribution.
  b.Print();
  b.T(); //Performing in-place lazy-transpose to change dimensions to 5x3.

  SHMatrix c(cublasHandle, std::vector<int> { 3, 3 }, GPU); //SHMatrix to store dot-product results.

  SHMatrix::Dot(cublasHandle, a, b, c); //Performing dot-product on GPU.
  c.Print();

  c.Move2CPU();
  SHMatrix::Dot(cublasHandle, a, b, c); //Performing dot-product on CPU.
  c.Print();

  b.T(); //Changing dimensions to 3x5 for element-wise operations with a.

  a += b; //In-place matrix-matrix add operation (b is added to a) on GPU.
  a.Print();

  a.Move2CPU();
  a += b; //In-place matrix-matrix add operation (b is added to a) on CPU.
  a.Print();
}