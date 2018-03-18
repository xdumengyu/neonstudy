#include "winograd_2x2_3x3.h"
void conv3x3s1_winograd(
				float *dst,
				int dst_stride,
			   	const float *src,
				int src_stride,
				int rows,
				int cols,
				const float* kernel) 
{
	//TODO: copy make border
	// current impl just ignore some point
	const int row_count = (rows - 4) / 2;
	const int col_count = (cols - 4) / 2;
	float gt[4][4];
	float32x4_t g0 = vld1q_f32(kernel);
	float32x4_t g1 = vld1q_f32(kernel + 3);
	float32x4_t g2 = vld1q_f32(kernel + 5);
	float32x4_t g3;
	g2 = vextq_f32(g2, g2, 1);
	//kernel transform
	winograd_f2k3_kernel_transform_inplace(
					&g0,
					&g1,
					&g2,
					&g3
					);
	vst1q_f32(gt[0], g0);
	vst1q_f32(gt[1], g1);
	vst1q_f32(gt[2], g2);
	vst1q_f32(gt[3], g3);
	for (int i = 0; i < row_count; ++i)
	{
		const float* row_base = src + i * 2 * src_stride;
		float* const dst_base = dst + i * 2 * dst_stride;
		for (int j = 0; j < col_count; ++j) 
		{
			const float *row0 = row_base + j * 2;
			const float *row1 = row0 + src_stride;
			const float *row2 = row0 + src_stride * 2;
			const float *row3 = row0 + src_stride * 3;
			//tile
			float32x4_t r0 = vld1q_f32(row0);
			float32x4_t r1 = vld1q_f32(row1);
			float32x4_t r2 = vld1q_f32(row2);
			float32x4_t r3 = vld1q_f32(row3);
			// data transform
			winograd_f2k3_input_transform_inplace(
							&r0,
							&r1,
							&r2,
							&r3
							);	
			float32x4_t gt0 = vld1q_f32(gt[0]);
			float32x4_t gt1 = vld1q_f32(gt[1]);
			float32x4_t gt2 = vld1q_f32(gt[2]);
			float32x4_t gt3 = vld1q_f32(gt[3]);
			// dot product 
			r0 = vmulq_f32(r0, gt0);
			r1 = vmulq_f32(r1, gt1);
			r2 = vmulq_f32(r2, gt2);
			r3 = vmulq_f32(r3, gt3);

			// output transform
			winograd_f2k3_out_transform_inplace(
							&r0,
							&r1,
							&r2,
							&r3
							);
			float ans[2][4];
			vst1q_f32(ans[0], r0);
			vst1q_f32(ans[1], r1);
			dst_base[j * 2] = ans[0][0];
			dst_base[j * 2 + 1] = ans[0][1];
			dst_base[j * 2 + dst_stride] = ans[1][0];
			dst_base[j * 2 + dst_stride + 1] = ans[1][1];
		}
	}
	// TODO: copy cut border with copy make border
	return ;
}
