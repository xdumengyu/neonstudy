//============================================================================
// Name        : transpose.cpp
// Author      : mengyu
// Version     :
// Copyright   : Your copyright notice
// Description : 
//============================================================================

#include <iostream>
#include <arm_neon.h>
#include <cstdio>
#include "macro.h"

using namespace std;
inline void Transpose32x4x4(float32x4_t& q0, float32x4_t &q1, float32x4_t &q2, float32x4_t& q3)
{
//  origin:
//  00 01 02 03
//  10 11 12 13
//  20 21 22 23
//  30 31 32 33
	float32x4x2_t q01 = vtrnq_f32(q0, q1);
	float32x4x2_t q23 = vtrnq_f32(q2, q3);
//	q01 = (00, 10, 02, 12), (01, 11, 03, 13)
//	q23 = (20, 30, 22, 32), (21, 31, 23, 33)

//	result:
//	00 10 20 30
//	01 11 21 31
//	02 12 22 32
//	03 13 23 33
	q0 = vcombine_f32(vget_low_f32(q01.val[0]), vget_low_f32(q23.val[0]));
	q1 = vcombine_f32(vget_low_f32(q01.val[1]), vget_low_f32(q23.val[1]));
	q2 = vcombine_f32(vget_high_f32(q01.val[0]), vget_high_f32(q23.val[0]));
	q3 = vcombine_f32(vget_high_f32(q01.val[1]), vget_high_f32(q23.val[1]));
	return ;
}
namespace {
	template<typename T>
	void Print2DArray(const T* const p, int rows, int cols) {
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				printf("%4d ", p[i * cols + j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	int SimpleCheck(const float * const A, const float * const B, int nElement)
	{
		int ret = 1;
		for (int i = 0; i < nElement; ++i)
		{
			if (A[i] != B[i]) {
				ret = 0;
				break;
			}
		}
		return ret;
	}
}

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	float32_t a[] = {
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16
	};
	float32_t ans_pred[] = {
				1, 5, 9, 13,
				2, 6, 10, 14,
				3, 7, 11, 15,
				4, 8, 12, 16
	};
	float32x4_t q0 = vld1q_f32(a);
	float32x4_t q1 = vld1q_f32(a + 4);
	float32x4_t q2 = vld1q_f32(a + 8);
	float32x4_t q3 = vld1q_f32(a + 12);
	Transpose32x4x4(q0, q1, q2, q3);
	vst1q_f32(a, q0);
	vst1q_f32(a + 4, q1);
	vst1q_f32(a + 8, q2);
	vst1q_f32(a + 12, q3);
	if (SimpleCheck(a, ans_pred, 16)) {
		//pass
		printf("Success\n");
	} else {
		//failed
		printf("Failed\n");
		printf("Expect:\n");
		Print2DArray(ans_pred, 4, 4);
		printf("Got:\n");
		Print2DArray(a, 4, 4);
	}
	return 0;
}
