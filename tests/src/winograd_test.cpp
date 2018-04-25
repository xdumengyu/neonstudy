#include "gtest/gtest.h"
#include "winograd_2x2_3x3.h"
#include <cmath>
#include <arm_neon.h>
#include <iostream>
TEST(testWino_f2_k3, inputTransform) 
{
	const float in[16] =
	{
		1,	2,	3,	4,
		5,	6,	7,	8,
		9,	10,	11,	12,
		13,	14,	15,	16
	};	
	float out[16];
	float out_expected[16] = 
	{
		0,	-4,	0,	0,
		-16, 34, 8, 16,
		0, 2, 0, 0,
		0, 4, 0, 0
	};
	float32x4_t q0, q1, q2, q3;

	q0 = vld1q_f32(in);
	q1 = vld1q_f32(in + 4);
	q2 = vld1q_f32(in + 8);
	q3 = vld1q_f32(in + 12);
	winograd_f2k3_input_transform_inplace(
					&q0,
					&q1,
					&q2,
					&q3
					);
	vst1q_f32(out, q0);
	vst1q_f32(out + 4, q1);
	vst1q_f32(out + 8, q2);
	vst1q_f32(out + 12, q3);
  	float max_diff = 0;	
	for (int i = 0; i < 16; ++i) {
//		max_diff = std::max(max_diff, fabsf(out[i] - out_expected[i]));
		EXPECT_NEAR(out[i], out_expected[i], 1e-5);	
	}

}
TEST(testWino_f2_k3, kernelTransform) 
{
	const float in[9] =
	{
		8,	1,	6,
		3,	5,	7,
		4,	9,	2
	};	
	float out[16];
	float out_expected[16] = 
	{
		8, 7.5, 4.5, 4,
		7.5, 11.25, 3.75, 7.5,
		6.5, 3.75, 1.25, -1.5,
		6, 7.5, 0.5, 2
	};
	float32x4_t q0, q1, q2, q3;

	q0 = vld1q_f32(in);
	q1 = vld1q_f32(in + 3);
	q2 = vld1q_f32(in + 5);
	q2 = vextq_f32(q2, q2, 1);
	winograd_f2k3_kernel_transform_inplace(
					&q0,
					&q1,
					&q2,
					&q3
					);
	vst1q_f32(out, q0);
	vst1q_f32(out + 4, q1);
	vst1q_f32(out + 8, q2);
	vst1q_f32(out + 12, q3);
  	float max_diff = 0;	
	for (int i = 0; i < 16; ++i) {
//		max_diff = std::max(max_diff, fabsf(out[i] - out_expected[i]));
		EXPECT_NEAR(out[i], out_expected[i], 1e-5);	
	}

}
TEST(testWino_f2_k3, outputTransform) 
{
	const float in[16] =
	{
		16, 2, 3, 13,
		5, 11, 10, 8,
		9, 7, 6, 12,
		4, 14, 15, 1
	};	
	float out[2][4];
	float out_expected[2][4] = 
	{
		69, 37, 0, 0,
		34, -4, 0, 0
	};
	float32x4_t q0, q1, q2, q3;
	q0 = vld1q_f32(in);
	q1 = vld1q_f32(in + 4);
	q2 = vld1q_f32(in + 8);
	q3 = vld1q_f32(in + 12);
	winograd_f2k3_out_transform_inplace(
					&q0,
					&q1,
					&q2,
					&q3
					);
	vst1q_f32(out[0], q0);
	vst1q_f32(out[1], q1);
  	float max_diff = 0;	
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
//		max_diff = std::max(max_diff, fabsf(out[i] - out_expected[i]));
			EXPECT_NEAR(out[i][j], out_expected[i][j], 1e-5);	
		}
	}

}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
