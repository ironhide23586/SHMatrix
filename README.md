# SHMatrix

![alt-text]:https://fontmeme.com/permalink/170623/a7c127d73d853b8b11305b59b7042df9.png

A neat C++ custom Matrix class to perform super-fast GPU (or CPU) powered Matrix/Vector computations with minimal code, leveraging the power of cuBLAS where applicable (Python Interface in the works).

Relevant DLL files (etc.) contained in the "package" folder for respective platforms. More details coming soon...

### Example usage for performing a Matrix-Matrix Dot Product & an element-wise add on the GPU- 
This only shows the += operation, similar pattern can be used for +, *, - operations between SHMatrix objects or floats (any way) on CPU or GPU.

```c++
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
```

### 1. Class Methods -
#### 1.1 __Constructors-__
  1.1.1
  ```c++
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           float *mat_data, std::vector<int> &dims,
           mem_location = GPU);
  ```
  1.1.2
  ```c++
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           std::vector<int> &dims, mem_location = GPU,
           bool default_init = false, float init_val = 0.0f);
  ```
  1.1.3
  ```c++
  SHMatrix(const cublasHandle_t &cublas_handle_arg,
           SHMatrix &src_shmatrix, mem_location = GPU);
  ```
  
#### 1.2 __Non-static Public Methods__
  1.2.1
  ```c++
  void Equate(SHMatrix &src_shmatrix);
  ```
  1.2.2
  ```c++
  void Reallocate(std::vector<int> &dims, mem_location mem_loc = GPU,
                  bool copy_original = false, bool default_init = false,
                  float init_val = 0.0f);
  ```
  1.2.3
  ```c++
  void Print(bool print_elem = true);
  ```
  1.2.4
  ```c++
  void Move2GPU();
  ```
  1.2.5
  ```c++
  void Move2CPU();
  ```
  1.2.6
  ```c++
  Clear();
  ```
  1.2.7
  ```c++
  void GaussianInit(float mean = 0.0f, float stddev = 0.1f);
  ```
  1.2.8
  ```c++
  void UniformInit(float lower = -0.5f, float higher = 0.5f);
  ```
  1.2.9
  ```c++
  SHMatrix& T();
  ```
  1.2.10
  ```c++
  SHMatrix& Scale(float scale_arg);
  ```
  1.2.11
  ```c++
  void CommitUnaryOps();
  ```
  1.2.12
  ```c++
  void CommitTranspose();
  ```
  1.2.13
  ```c++
  void CommitScale();
  ```

#### 1.3 __Static Public Methods__
  1.3.1
  ```c++
  void Dot(cublasHandle_t cublas_handle, SHMatrix &A, SHMatrix &B, SHMatrix &C);
  ```
  1.3.2
  ```c++
  float* DataPointerAtLoc(SHMatrix& arg, mem_location desired_loc);
  ```
  1.3.3
  ```c++
  float GetGaussianNum(float mean, float stddev);
  ```
  1.3.4
  ```c++
  float GetUniformNum(float lower, float higher);
  ```
  
#### 1.4 __Public Properties__
  1.4.1
  ```c++
  float *data;
  ```
  1.4.2
  ```c++
  float scalar;
  ```
  1.4.3
  ```c++
  mem_location data_loc;
  ```
  1.4.4
  ```c++
  std::vector<int> data_dims;
  ```
  1.4.5
  ```c++
  std::string name;
  ```
  1.4.6
  ```c++
  float mean;
  ```
  1.4.7
  ```c++
  float mini;
  ```
  1.4.8
  ```c++
  float maxi;
  ```
  1.4.9
  ```c++
  bool allocated;
  ```
  1.4.10
  ```c++
  int rows;
  ```
  1.4.11
  ```c++
  int cols;
  ```
  1.4.12
  ```c++
  int num_elems;
  ```
  1.4.13
  ```c++
  float mini_idx;
  ```
  1.4.14
  ```c++
  float maxi_idx;
  ```
