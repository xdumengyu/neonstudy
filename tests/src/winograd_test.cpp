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
int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
