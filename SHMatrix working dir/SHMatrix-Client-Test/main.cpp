#include <vector>
#include <iostream>

#include "SHMatrix.h"

using namespace std;

int main() {
  cublasHandle_t cublasHandle;
  CublasSafeCall(cublasCreate_v2(&cublasHandle));
  
  SHMatrix a(cublasHandle, std::vector<int> { 3, 5 }, GPU);
  a.GaussianInit();
  a.Print();

  SHMatrix b(cublasHandle, std::vector<int> { 7, 5 }, GPU);
  b.UniformInit();
  b.Print();

  SHMatrix c(cublasHandle, std::vector<int> { 3, 3 }, GPU);

  SHMatrix::Dot(cublasHandle, a, b.T(), c);
  c.Print();

  c.Move2CPU();
  SHMatrix::Dot(cublasHandle, a, b, c);
  c.Print();

}

