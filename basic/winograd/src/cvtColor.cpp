/*
 * cvtColor.cpp
 *
 *  Created on: 2018Äê7ÔÂ12ÈÕ
 *      Author: mengyu1
 */
#include <iostream>
#include <arm_neon.h>
//#include "macro.h"
using namespace std;
namespace ncv {
namespace {
typedef enum {
	GRAY2RGB,
	RGB2GRAY
} CVT_TYPE;
typedef unsigned char uchar;
enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899, // == R2YF*16384
    G2Y = 9617, // == G2YF*16384
    B2Y = 1868, // == B2YF*16384
    BLOCK_SIZE = 256
};
}
void
rgb2gray( const uchar* srcarr, uchar* dstarr, int nElement ) {
// nElement = width * height
//#ifdef __aarch64__
	int si = 0;
	int dj = 0;
	uint16x4_t wr, wg, wb;
	wr = vdup_n_u16(R2Y);
	wg = vdup_n_u16(G2Y);
	wb = vdup_n_u16(B2Y);
	for (; dj < nElement; si += 48, dj += 16) {
		uint8x16x3_t bgr = vld3q_u8(srcarr + si);
		uint16x8_t b0123, b4567, g0123, g4567, r0123, r4567;
		b0123 = vmovl_u8(vget_low_u8(bgr.val[0]));
		b4567 = vmovl_u8(vget_high_u8(bgr.val[0]));
		g0123 = vmovl_u8(vget_low_u8(bgr.val[1]));
		g4567 = vmovl_u8(vget_high_u8(bgr.val[1]));
		r0123 = vmovl_u8(vget_low_u8(bgr.val[2]));
		r4567 = vmovl_u8(vget_high_u8(bgr.val[2]));

		uint32x4_t b01 = vmull_u16(vget_low_u16(b0123), wb);
		uint32x4_t b23 = vmull_u16(vget_high_u16(b0123), wb);
		uint32x4_t g01 = vmull_u16(vget_low_u16(g0123), wg);
		uint32x4_t g23 = vmull_u16(vget_high_u16(g0123), wg);
		uint32x4_t r01 = vmull_u16(vget_low_u16(r0123), wr);
		uint32x4_t r23 = vmull_u16(vget_high_u16(r0123), wr);


		uint32x4_t b45 = vmull_u16(vget_low_u16(b4567), wb);
		uint32x4_t b67 = vmull_u16(vget_high_u16(b4567), wb);
		uint32x4_t g45 = vmull_u16(vget_low_u16(g4567), wg);
		uint32x4_t g67 = vmull_u16(vget_high_u16(g4567), wg);
		uint32x4_t r45 = vmull_u16(vget_low_u16(r4567), wr);
		uint32x4_t r67 = vmull_u16(vget_high_u16(r4567), wr);

		b01 = vaddq_u32(b01, vaddq_u32(g01, r01));
		b23 = vaddq_u32(b23, vaddq_u32(g23, r23));
		b45 = vaddq_u32(b45, vaddq_u32(g45, r45));
		b67 = vaddq_u32(b67, vaddq_u32(g67, r67));

		uint8x8_t ans0123 = vqmovn_u16(vcombine_u16(vshrn_n_u32(b01, yuv_shift), vshrn_n_u32(b23, yuv_shift)));
		uint8x8_t ans4567 = vqmovn_u16(vcombine_u16(vshrn_n_u32(b45, yuv_shift), vshrn_n_u32(b67, yuv_shift)));
		vst1q_u8(dstarr + dj, vcombine_u8(ans0123, ans4567));
	}
	for (; dj < nElement; ++si, ++dj) {
		int b = *(srcarr + si);
		int g = *(srcarr + si + 1);
		int r = *(srcarr + si + 2);
		int gray = b * B2Y + g * G2Y + r * R2Y;
		gray >>= yuv_shift;
		*(dstarr + dj) = gray;
	}
//#else
	;
//#endif
	return ;
}
};
//int main()
//{
//	typedef unsigned char uchar;
//	const int height = 512;
//	const int width = 512;
//	const int channel = 3;
//	uchar *src = new uchar[height * width * channel];
//	uchar *dst = new uchar[height * width];
//	int nElements = height * width;
//	ncv::rgb2gray(src, dst, nElements);
//	delete[] src;
//	delete[] dst;
//}


