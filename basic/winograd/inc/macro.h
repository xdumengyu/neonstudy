#include <arm_neon.h>
#if defined(__GNUC__)
	#define INLINE __attribute__((__always_inline__))
#else
	#define INLINE inline
#endif
