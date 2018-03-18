#include <arm_neon.h>
#include "macro.h"

//  


//input 4x4 transform
//transpoed answer
//G*trans(G*g) = G*gT*GT
static INLINE void winograd_f2k3_input_transform_inplace(
				float32x4_t __restrict *q0,
				float32x4_t __restrict *q1,
				float32x4_t __restrict *q2,
				float32x4_t __restrict *q3
				) 
{
	float32x4_t wq0 = *q0 - *q2;
	float32x4_t wq1 = *q1 + *q2;
	float32x4_t wq2 = *q2 - *q1;
	float32x4_t wq3 = *q3 - *q1;
	//transpose
	float32x4x2_t wq01 = vtrnq_f32(wq0, wq1);
	float32x4x2_t wq23 = vtrnq_f32(wq2, wq3);
	float32x4_t nq0 = vcombine_f32(vget_low_f32(wq01.val[0]), vget_low_f32(wq23.val[0]));
	float32x4_t nq1 = vcombine_f32(vget_low_f32(wq01.val[1]), vget_low_f32(wq23.val[1]));
	float32x4_t nq2 = vcombine_f32(vget_high_f32(wq01.val[0]), vget_high_f32(wq23.val[0]));
	float32x4_t nq3 = vcombine_f32(vget_high_f32(wq01.val[1]), vget_high_f32(wq23.val[1]));
	//transposed result
	*q0 = nq0 - nq2;
	*q1 = nq1 + nq2;
	*q2 = nq2 - nq1;
	*q3 = nq3 - nq1;
	return ;
}
//kernel 3x3 transform
//transpoed answer
//BT*trans(BT*d) = BT * dT *B
static INLINE void winograd_f2k3_kernel_transform_inplace(
				float32x4_t __restrict *q0,
				float32x4_t __restrict *q1,
				float32x4_t __restrict *q2,
				//q4 for output
				float32x4_t __restrict *q3
				)
{
	float32x4_t q0_add_q2 = *q0 + *q2;
	const float32x4_t const_0_5 = vdupq_n_f32(0.5f);
	float32x4_t wq0 = *q0;
	float32x4_t wq1 = q0_add_q2 + *q1;
	float32x4_t wq2 = q0_add_q2 - *q1;
	float32x4_t wq3 = *q1;
	wq1 = vmulq_f32(wq1, const_0_5);
	wq2 = vmulq_f32(wq2, const_0_5);
	//transpose
	float32x4x2_t wq01 = vtrnq_f32(wq0, wq1);
	float32x4x2_t wq23 = vtrnq_f32(wq2, wq3);
	float32x4_t nq0 = vcombine_f32(vget_low_f32(wq01.val[0]), vget_low_f32(wq23.val[0]));
	float32x4_t nq1 = vcombine_f32(vget_low_f32(wq01.val[1]), vget_low_f32(wq23.val[1]));
	float32x4_t nq2 = vcombine_f32(vget_high_f32(wq01.val[0]), vget_high_f32(wq23.val[0]));
	float32x4_t nq3 = vcombine_f32(vget_high_f32(wq01.val[1]), vget_high_f32(wq23.val[1]));
	q0_add_q2 = nq0 + nq2;
	//transposed result
	*q0 = nq0;
	*q1 = q0_add_q2 + nq1;
	*q2 = q0_add_q2 - nq1;
	*q3 = nq1;
	*q1 = vmulq_f32(*q1, const_0_5);
	*q2 = vmulq_f32(*q2, const_0_5);
	return ;	
}
// dot = U .* V
// AT * trans(AT * dotT) = AT * dot * A = AT * (U .* V) *A
// input transposed dot
static INLINE void winograd_f2k3_out_transform_inplace(
				float32x4_t __restrict *d0,
				float32x4_t __restrict *d1,
				float32x4_t __restrict *d2,
				float32x4_t __restrict *d3
				) 
{
	float32x4_t wq0 = *d0 + *d1 + *d2;
	float32x4_t wq1 = 	    *d1 - *d2 + *d3;
	//(1,2,3,4),(5,6,7,8) => (1,5,3,7),(2,6,4,8)
	float32x4x2_t q01 = vtrnq_f32(wq0, wq1);  
	float32x4_t q2 = vcombine_f32(vget_high_f32(q01.val[0]), vdup_n_f32(0)); // (3, 7, 0, 0)
	float32x4_t q3 = vcombine_f32(vget_high_f32(q01.val[1]), vdup_n_f32(0)); // (4, 8, 0, 0)
	*d0 = q01.val[0] + q01.val[1] + q2;
	*d2 = 			   q01.val[1] - q2 + q3;
}
