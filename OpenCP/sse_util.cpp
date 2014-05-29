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