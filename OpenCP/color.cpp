#include "opencp.hpp"

void cvtColorBGR2PLANE_8u( const Mat& src, Mat& dest)
{
	dest.create(Size(src.cols,src.rows*3),CV_8U);

	const int size = src.size().area();
	const int ssesize = 3*size-((48-(3*size)%48)%48);
	const int ssecount = ssesize/48;
	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(src.rows);
	uchar* R = dest.ptr<uchar>(2*src.rows);

	//BGR BGR BGR BGR BGR B	
	//GR BGR BGR BGR BGR BG
	//R BGR BGR BGR BGR BGR
	//BBBBBBGGGGGRRRRR shuffle
	const __m128i mask1 = _mm_setr_epi8(0,3,6,9,12,15,1,4,7,10,13,2,5,8,11,14);
	//GGGGGBBBBBBRRRRR shuffle
	const __m128i smask1 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask1 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	//GGGGGGBBBBBRRRRR shuffle
	const __m128i mask2 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);
	//const __m128i smask2 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask2 = _mm_setr_epi8(0,1,2,3,4,11,12,13,14,15,5,6,7,8,9,10);

	//RRRRRRGGGGGBBBBB shuffle -> same mask2
	//__m128i mask3 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);

	//const __m128i smask3 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,10);
	//const __m128i ssmask3 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	const __m128i bmask1 = _mm_setr_epi8
		(255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask2 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0);

	const __m128i bmask3 = _mm_setr_epi8
		(255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask4 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0);	

	__m128i a,b,c;

	for(int i=0;i<ssecount;i++)
	{
		a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s)),mask1);
		b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+16)),mask2);
		c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+32)),mask2);
		_mm_storeu_si128((__m128i*)(B),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask1),bmask2));
		a = _mm_shuffle_epi8(a,smask1);
		b = _mm_shuffle_epi8(b,smask1);
		c = _mm_shuffle_epi8(c,ssmask1);
		_mm_storeu_si128((__m128i*)(G),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask2));

		a = _mm_shuffle_epi8(a,ssmask1);
		c = _mm_shuffle_epi8(c,ssmask1);
		b = _mm_shuffle_epi8(b,ssmask2);

		_mm_storeu_si128((__m128i*)(R),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask4));

		s+=48;
		R+=16;
		G+=16;
		B+=16;
	}
	for(int i=ssesize;i<3*size;i+=3)
	{
		B[0]=s[0];
		G[0]=s[1];
		R[0]=s[2];
		s+=3,R++,G++,B++;
	}
}

void cvtColorBGR2PLANE_32f( const Mat& src, Mat& dest)
{
	dest.create(Size(src.cols,src.rows*3),CV_32F);

	const int size = src.size().area();
	const int ssesize = 3*size-((12-(3*size)%12)%12);
	const int ssecount = ssesize/12;
	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(src.rows);
	float* R = dest.ptr<float>(2*src.rows);

	for(int i=0;i<ssecount;i++)
	{
		__m128 a = _mm_load_ps(s);
		__m128 b = _mm_load_ps(s+4);
		__m128 c = _mm_load_ps(s+8);

		__m128 aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,2,3,0));
		aa=_mm_blend_ps(aa,b,4);
		__m128 cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,3,2,0));
		aa=_mm_blend_ps(aa,cc,8);
		_mm_storeu_ps((B),aa);

		aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,2,0,1));
		__m128 bb = _mm_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
		bb=_mm_blend_ps(bb,aa,1);
		cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,1,0));
		bb=_mm_blend_ps(bb,cc,8);
		_mm_storeu_ps((G),bb);

		aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
		bb=_mm_blend_ps(aa,b,2);
		cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(3,0,1,2));
		cc=_mm_blend_ps(bb,cc,12);
		_mm_storeu_ps((R),cc);

		s+=12;
		R+=4;
		G+=4;
		B+=4;
	}
	for(int i=ssesize;i<3*size;i+=3)
	{
		B[0]=s[0];
		G[0]=s[1];
		R[0]=s[2];
		s+=3,R++,G++,B++;
	}
}

template <class T>
void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
{
	vector<Mat> v(3);
	split(src,v);
	dest.create(Size(src.cols, src.rows*3),depth);

	memcpy(dest.data,                    v[0].data,src.size().area()*sizeof(T));
	memcpy(dest.data+src.size().area()*sizeof(T),  v[1].data,src.size().area()*sizeof(T));
	memcpy(dest.data+2*src.size().area()*sizeof(T),v[2].data,src.size().area()*sizeof(T));
}

void cvtColorBGR2PLANE(const Mat& src, Mat& dest)
{
	if(src.channels()!=3)printf("input image must have 3 channels\n");

	if(src.depth()==CV_8U)
	{
		//cvtColorBGR2PLANE_<uchar>(src, dest, CV_8U);
		//Mat d2;
		cvtColorBGR2PLANE_8u(src, dest);

	}
	else if(src.depth()==CV_16U)
	{
		cvtColorBGR2PLANE_<ushort>(src, dest, CV_16U);
	}
	if(src.depth()==CV_16S)
	{
		cvtColorBGR2PLANE_<short>(src, dest, CV_16S);
	}
	if(src.depth()==CV_32S)
	{
		cvtColorBGR2PLANE_<int>(src, dest, CV_32S);
	}
	if(src.depth()==CV_32F)
	{
		cvtColorBGR2PLANE_32f(src, dest);
		//cvtColorBGR2PLANE_<float>(src, dest, CV_32F);
	}
	if(src.depth()==CV_64F)
	{
		cvtColorBGR2PLANE_<double>(src, dest, CV_64F);
	}
}