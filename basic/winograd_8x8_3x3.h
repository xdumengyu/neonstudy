#include <arm_neon.h>
#include "macro.h"
//  const float ktm[8][3] = {
//        {   1.0f,     0.0f,     0.0f},
//        {-2.0f/9,  -2.0f/9,  -2.0f/9},
//        {-2.0f/9,   2.0f/9,  -2.0f/9},
//        {1.0f/90,  1.0f/45,  2.0f/45},
//        {1.0f/90, -1.0f/45,  2.0f/45},
//        {1.0f/45,  1.0f/90, 1.0f/180},
//        {1.0f/45, -1.0f/90, 1.0f/180},
//        {   0.0f,     0.0f,     1.0f}
//    };
static INLINE void winograd8x8_3x3_kernel_transform(
		const float32x4_t g0,
		const float32x4_t g1,
		const float32x4_t g2,
		float32x4_t __restrict *d0,
		float32x4_t __restrict *d1,
		float32x4_t __restrict *d2,
		float32x4_t __restrict *d3,
		float32x4_t __restrict *d4,
		float32x4_t __restrict *d5,
		float32x4_t __restrict *d6,
		float32x4_t __restrict *d7,
		bool rescale_coefficients
		)
{
	const float32x4_t const_4 = vdupq_n_f32(4);
	const float32x4_t two_g1 = g1 + g1;

	float32x4_t a02 = g0 + g2;

	float32x4_t w3 = vmlaq_f32(g0, g2, const_4);

	float32x4_t w5  = vmlaq_f32(g2, g0, const_4);


	//d1 = ((g0 + g2) + g1) * -2.0/9

	//d2 = ((g0 + g2) - g1) * -2.0/9

	*d1 = a02 + g1;

	*d2 = a02 - g1;



	//d3 = ((g2 * 4 + g0) + 2 * g1) * 1 / 90 = ((g0 + g2 + 2 * g1) + 3 * g2) * 1.0 / 90

	//d4 = ((g2 * 4 + g0) - 2 * g1) * 1 / 90 = ((g0 + g2 - 2 * g1) + 3 * g2) * 1.0 / 90

	//d5 = ((g2 + g0 * 4) + 2 * g1) * 1 / 180 = ((g0 + g2 + 2 * g1) + 3 * g0) * 1.0 / 90

	//d5 = ((g2 + g0 * 4) - 2 * g1) * 1 / 180 = ((g0 + g2 - 2 * g1) + 3 * g0) * 1.0 / 90

	*d3 = w3 + two_g1;

	*d4 = w3 - two_g1;

	*d5 = w5 + two_g1;

	*d6 = w5 - two_g1;

	*d0 = g0;
	if (rescale_coefficients) {
		
		const float32x4_t minus2D9 = vdupq_n_f32(-2.0 / 9);
		
		*d1 = vmulq_f32(*d1, minus2D9);

		*d2 = vmulq_f32(*d2, minus2D9);

		const float32x4_t rev90 = vdupq_n_f32(1.0 / 90);

		*d3 = vmulq_f32(*d3, rev90);

		*d4 = vmulq_f32(*d4, rev90);

		const float32x4_t rev180 = vdupq_n_f32(1.0 / 180);
		
		*d5 = vmulq_f32(*d5, rev180);

		*d6 = vmulq_f32(*d6, rev180);
	}
	*d7 = g2;
}

static INLINE void winograd8x8_3x3_input_transform_inplace(
		float32x4_t __restrict *q0,
		float32x4_t __restrict *q1,
		float32x4_t __restrict *q2,
		float32x4_t __restrict *q3,
		float32x4_t __restrict *q4,
		float32x4_t __restrict *q5,
		float32x4_t __restrict *q6,
		float32x4_t __restrict *q7
		)
{

// rewrite formulation to [A + B * C]


//	float32x4_t a26 = d2 + d6;
//
//	//a23s4 = d4 * -4.25 + a26
//	float32x4_t a23S4 = vmlaq_f32(a26, vdupq_n_f32(-4.25), d4);
//	float32x4_t a1S3  = vmlaq_f32(d1,  vdupq_n_f32(-4.25), d3);

	//         const float itm[8][8] = {
	//				0		1		2		3		4		5		6	7
	//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
	//
	//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
	//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
	//
	//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
	//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
	//
	//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
	//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
	//
	//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
	//         };
    // 0 = r00 - r06 + (r04 - r02) * 5.25
    // 7 = r07 - r01 + (r03 - r05) * 5.25

    // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
    // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)
    // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)

//	(r01 * 0.5 - r03 * 2.5 + r05 * 2) = 2 * ((r05 - r03 * 1.25) + r01 * 0.25)
    // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

    // reuse r04 * 1.25
    // reuse r03 * 2.5
    // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
//	 (r01 * 2 - r03 * 2.5 + r05 * 0.5) = 2 * (r01 - r03 * 1.25 + r05 * 0.25)
    // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)
		const float32x4_t const_0_25 = vdupq_n_f32(0.25f);

	const float32x4_t const_1_25 = vdupq_n_f32(1.25f);

	const float32x4_t const_4_25 = vdupq_n_f32(4.25f);

	const float32x4_t const_5_25 = vdupq_n_f32(5.25f);

	float32x4_t wq0 = vmlaq_f32(*q0 - *q6, *q4 - *q2, const_5_25);

	float32x4_t wq7 = vmlaq_f32(*q7 - *q1, *q3 - *q5, const_5_25);




//	const float32x4_t *q2_times_0_25 = vmulq_f32(*q2, const_0_25);

//	const float32x4_t *q1_times_0_25 = vmulq_f32(*q1, const_0_25);

	const float32x4_t q4_times_1_25 = vmulq_f32(*q4, const_1_25);

	const float32x4_t q3_times_1_25 = vmulq_f32(*q3, const_1_25);


	float32x4_t wq1 = vmlsq_f32(*q2 + *q6, *q4, const_4_25);

	float32x4_t wq2 = vmlsq_f32(*q1 + *q5, *q3, const_4_25);

	float32x4_t wq3 = *q6 - q4_times_1_25;

	float32x4_t wq4 = *q5 - q3_times_1_25;

	float32x4_t wq5 = *q2 - q4_times_1_25;

	float32x4_t wq6 = *q1 - q3_times_1_25;

	wq4 = vmlaq_f32(wq4, *q1, const_0_25);

	wq6 = vmlaq_f32(wq6, *q5, const_0_25);

	wq3 = vmlaq_f32(wq3, *q2, const_0_25);

	wq5 = vmlaq_f32(*q6, wq5, vdupq_n_f32(4.0f));



	const float32x4_t const_2 = vdupq_n_f32(2.0f);

	wq4 = vmulq_f32(wq4, const_2);

	wq6 = vmulq_f32(wq6, const_2);





	*q0 = wq0;

	*q1 = wq1 + wq2;

	*q2 = wq1 - wq2;

	*q3 = wq3 + wq4;

	*q4 = wq3 - wq4;

	*q5 = wq5 + wq6;

	*q6 = wq5 - wq6;

	*q7 = wq7;


}

//         const float otm[6][8] = {
//				0		1		2		3		4		5	  6		 7
//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
//         };

static INLINE void winograd8x8_3x3_output_transform_inplace(
		float32x4_t *q0,
		float32x4_t *q1,
		float32x4_t *q2,
		float32x4_t *q3,
		float32x4_t *q4,
		float32x4_t *q5,
		float32x4_t *q6,
		float32x4_t *q7
				)
{
	const float32x4_t q1Aq2 = *q1 + *q2;
	const float32x4_t q1Sq2 = *q1 - *q2;
	const float32x4_t q3Aq4 = *q3 + *q4;
	const float32x4_t q3Sq4 = *q3 - *q4;
	const float32x4_t q5Aq6 = *q5 + *q6;
	const float32x4_t q5Sq6 = *q5 - *q6;
	/*
	 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
	 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
	 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
	 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
	 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
	 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
	 */
	*q0 = *q0 + q1Aq2;
	*q5 = *q7 + q1Sq2;
	const float32x4_t const_16 = vdupq_n_f32(16);
	*q1 = vmlaq_f32(q1Sq2, q5Sq6, const_16);
	*q4 = vmlaq_f32(q1Aq2, q3Aq4, const_16);
	const float32x4_t const_8 = vdupq_n_f32(8);
	*q2 = vmlaq_f32(q1Aq2, q5Aq6, const_8);
	*q3 = vmlaq_f32(q1Sq2, q3Sq4, const_8);
	const float32x4_t const_32 = vdupq_n_f32(32);
	*q0 = vmlaq_f32(*q0, q5Aq6, const_32);
	*q5 = vmlaq_f32(*q5, q3Sq4, const_32);
	const float32x4_t const_2 = vdupq_n_f32(2);
	*q1 = vmlaq_f32(*q1, q3Sq4, const_2);
	*q4 = vmlaq_f32(*q4, q5Aq6, const_2);
	const float32x4_t const_4 = vdupq_n_f32(4);
	*q2 = vmlaq_f32(*q2, q3Aq4, const_4);
	*q3 = vmlaq_f32(*q3, q5Sq6, const_4);
	*q0 = *q0 + q3Aq4;
	*q5 = *q5 + q5Sq6;
}
