#include <opencv2/core/internal.hpp>

void memcpy_float_sse(float* dest, float* src, const int size)
{
	int x=0;
	for(x=0;x<=size-4;x+=4)
	{
		_mm_storeu_ps(dest+x,_mm_loadu_ps(src+x));
	}
	for(;x<size;x++)
	{
		dest[x]=src[x];
	}
}

//broadcast
/*
__m128 xxxx = _mm_shuffle_ps(first, first, 0x00); // _MM_SHUFFLE(0, 0, 0, 0)
__m128 yyyy = _mm_shuffle_ps(first, first, 0x55); // _MM_SHUFFLE(1, 1, 1, 1)
__m128 zzzz = _mm_shuffle_ps(first, first, 0xAA); // _MM_SHUFFLE(2, 2, 2, 2)
__m128 wwww = _mm_shuffle_ps(first, first, 0xFF); // _MM_SHUFFLE(3, 3, 3, 3)
*/