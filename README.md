# CUDA 100 days Learning Journey

This repo is an attempt to learn CUDA and parallel programming by following PMPP book and doing 100 days challenge https://github.com/hkproj/100-days-of-gpu?tab=readme-ov-file

---

## Day 1
### File: `vectadd.cu`
**Summary:**  
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.  

**Learned:**  
- Basics of writing a CUDA kernel.
- Understanding of grid, block, and thread hierarchy in CUDA.  
- How to allocate and manage device (GPU) memory using `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.  

### Reading:  
- Read **Chapter 1** of the PMPP book.  
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model.

## Day 2
### File: `matrix_add.cu`
**Summary:**  
Implemented matrix addition.

**Learned:**  
- How matrix is linearly stored in memory and how to access it using linear index.
- `dim3` type for grid, block, and thread.

### Reading:  
- Read **Chapter 2** and **Chapter 3** of the PMPP book.
- Learned how threads perform complex operations in parallel rather than simple operations.
- Next steps: implement matrix multiplication, colorToGrayscaleConversion, and imageBlurring.

## Day 3
### File: `matrix_mult.cu`
**Summary:**  
Implemented matrix multiplication.

### Reading:  
- Read **Chapter 3** (continued) of the PMPP book.
- Next steps: use shared memory to speed up matrix multiplication, implement colorToGrayscaleConversion, and imageBlurring.


---
Template borrowed from https://github.com/a-hamdi/cuda/tree/main
