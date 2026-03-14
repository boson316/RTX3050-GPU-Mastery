/*
 * C++ 純 CPU 版 Vector Add（對照用）
 * C[i] = A[i] + B[i]，一個 for 迴圈依序做
 * 編譯 (VS): cl /EHsc vector_add_cpu.cpp
 * 編譯 (g++): g++ -o vector_add_cpu vector_add_cpu.cpp
 */

#include <stdio.h>
#include <stdlib.h>

// C++：一個迴圈，一個 thread 依序處理每個 i
void vectorAdd_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

int main() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    vectorAdd_cpu(A, B, C, N);

    printf("Result C[0]: %f\n", C[0]);  // 應為 0

    free(A);
    free(B);
    free(C);
    return 0;
}
