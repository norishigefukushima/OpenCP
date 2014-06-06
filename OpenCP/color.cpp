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


//8u
void splitBGRLineInterleave_8u( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_8U);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(1);
	uchar* R = dest.ptr<uchar>(2);

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

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=16)
		{
			a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i)),mask1);
			b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+16)),mask2);
			c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+32)),mask2);
			_mm_stream_si128((__m128i*)(B+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask1),bmask2));

			a = _mm_shuffle_epi8(a,smask1);
			b = _mm_shuffle_epi8(b,smask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			_mm_stream_si128((__m128i*)(G+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask2));

			a = _mm_shuffle_epi8(a,ssmask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			b = _mm_shuffle_epi8(b,ssmask2);

			_mm_stream_si128((__m128i*)(R+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask4));
		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}

void splitBGRLineInterleave_32f( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_32F);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=4)
		{
			__m128 a = _mm_load_ps((s+3*i));
			__m128 b = _mm_load_ps((s+3*i+4));
			__m128 c = _mm_load_ps((s+3*i+8));

			__m128 aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,2,3,0));
			aa=_mm_blend_ps(aa,b,4);
			__m128 cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,3,2,0));
			aa=_mm_blend_ps(aa,cc,8);
			_mm_stream_ps((B+i),aa);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,2,0,1));
			__m128 bb = _mm_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
			bb=_mm_blend_ps(bb,aa,1);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,1,0));
			bb=_mm_blend_ps(bb,cc,8);
			_mm_stream_ps((G+i),bb);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
			bb=_mm_blend_ps(aa,b,2);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(3,0,1,2));
			cc=_mm_blend_ps(bb,cc,12);
			_mm_stream_ps((R+i),cc);

		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}


template <class T>
void cvtColorPLANE2BGR_(const Mat& src, Mat& dest, int depth)
{
	int width = src.cols;
	int height = src.rows/3;
	T* b = (T*)src.ptr<T>(0);
	T* g = (T*)src.ptr<T>(height);
	T* r = (T*)src.ptr<T>(2*height);

	Mat B(height, width, src.type(),b);
	Mat G(height, width, src.type(),g);
	Mat R(height, width, src.type(),r);
	vector<Mat> v(3);
	v[0]=B;
	v[1]=G;
	v[2]=R;
	merge(v,dest);
}

void cvtColorPLANE2BGR_8u_align(const Mat& src, Mat& dest)
{
	int width = src.cols;
	int height = src.rows/3;

	if(dest.empty()) dest.create(Size(width,height),CV_8UC3);
	else if(width!=dest.cols || height!=dest.rows) dest.create(Size(width,height),CV_8UC3);
	else if(dest.type()!=CV_8UC3) dest.create(Size(width,height),CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2*height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	int ssecount = width*height*3/48;

	const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	const __m128i bmask1 = _mm_setr_epi8(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);
	const __m128i bmask2 = _mm_setr_epi8(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

	for(int i=ssecount;i--;)
	{
		__m128i a = _mm_load_si128((const __m128i*)B);
		__m128i b = _mm_load_si128((const __m128i*)G);
		__m128i c = _mm_load_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a,mask1);
		b = _mm_shuffle_epi8(b,mask2);
		c = _mm_shuffle_epi8(c,mask3);
		_mm_stream_si128((__m128i*)(D),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
		_mm_stream_si128((__m128i*)(D+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
		_mm_stream_si128((__m128i*)(D+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));

		D+=48;
		B+=16;
		G+=16;
		R+=16;
	}
}

void cvtColorPLANE2BGR_8u(const Mat& src, Mat& dest)
{
	int width = src.cols;
	int height = src.rows/3;

	if(dest.empty()) dest.create(Size(width,height),CV_8UC3);
	else if(width!=dest.cols || height!=dest.rows) dest.create(Size(width,height),CV_8UC3);
	else if(dest.type()!=CV_8UC3) dest.create(Size(width,height),CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2*height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	int ssecount = width*height*3/48;
	int rem = width*height*3-ssecount*48;

	const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	const __m128i bmask1 = _mm_setr_epi8(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);
	const __m128i bmask2 = _mm_setr_epi8(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

	for(int i=ssecount;i--;)
	{
		__m128i a = _mm_loadu_si128((const __m128i*)B);
		__m128i b = _mm_loadu_si128((const __m128i*)G);
		__m128i c = _mm_loadu_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a,mask1);
		b = _mm_shuffle_epi8(b,mask2);
		c = _mm_shuffle_epi8(c,mask3);

		_mm_storeu_si128((__m128i*)(D),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
		_mm_storeu_si128((__m128i*)(D+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
		_mm_storeu_si128((__m128i*)(D+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));

		D+=48;
		B+=16;
		G+=16;
		R+=16;
	}
	for(int i=rem;i--;)
	{
		D[0]=*B;
		D[1]=*G;
		D[2]=*R;
		D+=3;
		B++,G++,R++;
	}
}

void cvtColorPLANE2BGR(const Mat& src, Mat& dest)
{
	if(src.depth()==CV_8U)
	{
		//cvtColorPLANE2BGR_<uchar>(src, dest, CV_8U);	
		if(src.cols%16==0)
			cvtColorPLANE2BGR_8u_align(src, dest);	
		else
			cvtColorPLANE2BGR_8u(src, dest);	
	}
	else if(src.depth()==CV_16U)
	{
		cvtColorPLANE2BGR_<ushort>(src, dest, CV_16U);
	}
	if(src.depth()==CV_16S)
	{
		cvtColorPLANE2BGR_<short>(src, dest, CV_16S);
	}
	if(src.depth()==CV_32S)
	{
		cvtColorPLANE2BGR_<int>(src, dest, CV_32S);
	}
	if(src.depth()==CV_32F)
	{
		cvtColorPLANE2BGR_<float>(src, dest, CV_32F);
	}
	if(src.depth()==CV_64F)
	{
		cvtColorPLANE2BGR_<double>(src, dest, CV_64F);
	}
}


void splitBGRLineInterleave_32fcast( const Mat& src, Mat& dest)
{
	Mat a,b;
	src.convertTo(a,CV_32F);
	splitBGRLineInterleave_32f(a,b);
	b.convertTo(dest,src.type());
}

void splitBGRLineInterleave( const Mat& src, Mat& dest)
{
	if(src.type()==CV_MAKE_TYPE(CV_8U,3))
	{
		CV_Assert(src.cols%16==0);
		splitBGRLineInterleave_8u(src,dest);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,3))
	{
		CV_Assert(src.cols%4==0);
		splitBGRLineInterleave_32f(src,dest);
	}
	else
	{
		CV_Assert(src.cols%4==0);
		splitBGRLineInterleave_32fcast(src,dest);
	}
}


void cvtColorBGR2BGRA(const Mat& src, Mat& dest, const uchar alpha)
{
	if(dest.empty())dest.create(src.size(),CV_8UC4);

	int size = src.size().area();
	uchar* s = (uchar*)src.ptr<uchar>(0);
	uchar* d = dest.ptr<uchar>(0);
	
	for(int i=0;i<size;i++)
	{
		*d++ = *s++;
		*d++ = *s++;
		*d++ = *s++;
		*d++ = alpha;
	}
}

void cvtColorBGRA2BGR(const Mat& src, Mat& dest)
{
	if(dest.empty())dest.create(src.size(),CV_8UC3);

	int size = src.size().area();
	uchar* s = (uchar*)src.ptr<uchar>(0);
	uchar* d = dest.ptr<uchar>(0);
	
	for(int i=0;i<size;i++)
	{
		*d++ = *s++;
		*d++ = *s++;
		*d++ = *s++;
		*s++;
	}
}


void makemultichannel(Mat& gray, Mat& color)
{
	Mat src;
	int channel = 9;
	color.create(gray.size(),CV_8UC(channel));
	copyMakeBorder(gray,src,1,1,1,1,BORDER_REPLICATE);
	//copyMakeBorder(gray,src,0,0,0,0,BORDER_REPLICATE);

	for(int j=0;j<gray.rows;j++)
	{
		uchar* s = src.ptr(j+1);s++;
		uchar* d = color.ptr(j);

		for(int i=0;i<gray.cols;i++)
		{
			//d[channel*i+0]=s[i];
			//d[channel*i+1]=s[i-1];
			//d[channel*i+2]=s[i+1];

			//d[channel*i+3]=s[i-src.cols];
			//d[channel*i+4]=s[i-src.cols-1];
			//d[channel*i+5]=s[i-src.cols+1];

			//d[channel*i+6]=s[i+src.cols];
			//d[channel*i+7]=s[i+src.cols-1];
			//d[channel*i+8]=s[i+src.cols+1];

			d[channel*i+0]=s[i];
			d[channel*i+1]=abs(s[i]-s[i-1]);
			d[channel*i+2]=abs(s[i]-s[i+1]);

			d[channel*i+3]=abs(s[i]-s[i-src.cols]);
			d[channel*i+4]=abs(s[i]-s[i-src.cols-1]);
			d[channel*i+5]=abs(s[i]-s[i-src.cols+1]);

			d[channel*i+6]=abs(s[i]-s[i+src.cols]);
			d[channel*i+7]=abs(s[i]-s[i+src.cols-1]);
			d[channel*i+8]=abs(s[i]-s[i+src.cols+1]);
		}
	}	
}