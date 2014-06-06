#include "opencp.hpp"

template <class T>
void splitToGrid_(const Mat& src, vector<Mat>& dest, Size gridNum, int borderRadius)
{
	int w = (src.cols%gridNum.width == 0) ? src.cols/gridNum.width : src.cols/gridNum.width+1;
	int h = (src.rows%gridNum.height == 0) ? src.rows/gridNum.height : src.cols/gridNum.height+1;
	Size grid = Size(w,h); 

	int remW = w*gridNum.width -src.cols;
	int remH = h*gridNum.height -src.rows;

	Mat im;
	if(remH==0 && remW==0 && borderRadius==0)
		im = (Mat)src;
	else
		copyMakeBorder(src,im, borderRadius, borderRadius+remH, borderRadius, borderRadius+remW, BORDER_REPLICATE);

	const int width = im.cols;
	const int height = im.rows;
	const int channels = im.channels();
	const int step = width*channels;

	//grid num info
	const int gnumwidth = gridNum.width;
	const int gnumheight = gridNum.height;
	const int gnumsize = gridNum.area();

	//grid info
	const int gwidth = grid.width;
	const int gheight = grid.height;
	const int gwstep = gwidth*channels;
	const int ghstep = gheight*step;

	//grid + border info
	const int bwidth = gwidth + 2*borderRadius;
	const int bheight = gheight + 2*borderRadius;
	const int bwstep = bwidth*im.channels();

	//create dest mat
	dest.resize(gnumsize);
	for(int i=0;i<gnumsize;i++)
		dest[i].create(Size(bwidth, bheight),src.type());

	//copy
	T* sptr = (T*)im.ptr<T>(0);
	for(int j=0;j<gnumheight;j++)
	{
		for(int i=0;i<gnumwidth;i++)
		{
			const int idx = gnumwidth*j+i;
			T* dst = dest[idx].ptr<T>(0);
			T* s = sptr + j*ghstep + i*gwstep;

			for(int k=0;k<bheight;k++)
			{
				memcpy(dst, s, sizeof(T)*bwstep);
				dst+=bwstep;
				s+=step;
			}
		}
	}
}

void splitToGrid_8u(const Mat& src, vector<Mat>& dest, Size gridNum, int borderRadius)
{
	int w = (src.cols%gridNum.width == 0) ? src.cols/gridNum.width : src.cols/gridNum.width+1;
	int h = (src.rows%gridNum.height == 0) ? src.rows/gridNum.height : src.rows/gridNum.height+1;
	Size grid = Size(w,h); 

	int remW = w*gridNum.width -src.cols;
	int remH = h*gridNum.height -src.rows;

	Mat im;
	if(remH==0 && remW==0 && borderRadius==0)
		im = (Mat)src;
	else
		copyMakeBorder(src,im, borderRadius, borderRadius+remH, borderRadius, borderRadius+remW, BORDER_REPLICATE);

	const int width = im.cols;
	const int height = im.rows;
	const int channels = im.channels();
	const int step = width*channels;

	//grid num info
	const int gnumwidth = gridNum.width;
	const int gnumheight = gridNum.height;
	const int gnumsize = gridNum.area();

	//grid info
	const int gwidth = grid.width;
	const int gheight = grid.height;
	const int gwstep = gwidth*channels;
	const int ghstep = gheight*step;

	//grid + border info
	const int bwidth = gwidth + 2*borderRadius;
	const int bheight = gheight + 2*borderRadius;
	const int bwstep = bwidth*im.channels();

	//create dest mat
	dest.resize(gnumsize);
	for(int i=0;i<gnumsize;i++)
		dest[i].create(Size(bwidth, bheight),src.type());


	//copy
	uchar* sptr = (uchar*)im.ptr<uchar>(0);
#ifdef CV_SSE2
	const int ssestep = bwstep/16;
	const int sserem = bwstep-ssestep*16;
#endif

	for(int j=0;j<gnumheight;j++)
	{
		for(int i=0;i<gnumwidth;i++)
		{
			const int idx = gnumwidth*j+i;
			uchar* dst = dest[idx].data;
			uchar* ss = sptr + j*ghstep + i*gwstep;

			for(int k=0;k<bheight;k++)
			{		
#ifdef CV_SSE2
				uchar* s = ss;
				uchar* d = dst;
				for(int i=0;i<ssestep;i++)
				{
					_mm_storeu_si128((__m128i*) d, _mm_loadu_si128((const __m128i*)s));
					d+=16,s+=16;
				}
				for(int i=0;i<sserem;i++)
				{
					*d = *s;
					s++,d++;
				}
#else if 
				memcpy(dst, ss, bwstep);
#endif
				dst+=bwstep;
				ss+=step;
			}
		}
	}
}

/*
Ex: 4x4 grid
|    |   | 0 1 2 3|
|    |   | 4 5 6 7|
|    | = | 8 91011|
|    |   |12131415|
*/
void splitToGrid(const Mat& src, vector<Mat>& dest, Size grid, int borderRadius)
{
	if(src.depth()==CV_8U)
	{
		//splitToGrid_<uchar>(src,dest,grid,borderRadius);
		splitToGrid_8u(src,dest,grid,borderRadius);
	}
	else if(src.depth()==CV_16U)
	{
		splitToGrid_<ushort>(src,dest,grid,borderRadius);
	}
	else if(src.depth()==CV_16S)
	{
		splitToGrid_<short>(src,dest,grid,borderRadius);
	}
	else if(src.depth()==CV_32F)
	{
		splitToGrid_<float>(src,dest,grid,borderRadius);
	}
	else if(src.depth()==CV_64F)
	{
		splitToGrid_<double>(src,dest,grid,borderRadius);
	}
}

using namespace std;
template <class T>
void mergeFromGrid_(vector<Mat>& src, Size beforeSize, Mat& dest, Size grid, int borderRadius)
{	
	const int width = beforeSize.width;
	const int height = beforeSize.height;

	const int channels = src[0].channels();
	

	if(dest.empty()) dest.create(Size(width,height),src[0].type());
	else if(dest.cols!=width ||dest.rows!=height) dest.create(Size(width,height),src[0].type());
	else if(dest.channels() !=channels) dest.create(Size(width,height),src[0].type());

	//grid num info
	const int gnumwidth = grid.width;
	const int gnumheight = grid.height;

	//grid info
	const int gwidth = src[0].cols-2*borderRadius;
	const int gheight = src[0].rows-2*borderRadius;
	const int gwstep = gwidth*channels;
	

	const int bwidth = src[0].cols;
	const int bwstep = bwidth*channels;

	const int soffset = channels*borderRadius;
	//copy
	


	
	Mat dbuff = Mat::zeros(Size(beforeSize.width+beforeSize.width%grid.width,
		beforeSize.height+beforeSize.height%grid.height),src[0].type());

	T* dptr = (T*)dbuff.ptr<T>(0);
	const int step = dbuff.cols*channels;
	const int ghstep = gheight*step;

	for(int j=0;j<gnumheight;j++)
	{
		for(int i=0;i<gnumwidth;i++)
		{
			const int idx = gnumwidth*j+i;
			T* ss = src[idx].ptr<T>(borderRadius) + soffset;
			T* dst = dptr + j*ghstep + i*gwstep;

			for(int k=0;k<gheight;k++)
			{
				memcpy(dst, ss, sizeof(T)*gwstep);
				//memset(dst, 128, sizeof(T)*gwstep);
				ss+=bwstep;
				dst+=step;
			}
		}
		// change copy order
		//for(int k=0;k<gheight;k++)
		//{
		//	T* dst = dptr + j*ghstep+ k*step;
		//	for(int i=0;i<gnumwidth;i++)
		//	{
		//		const int idx = gnumwidth*j+i;
		//		T* ss = src[idx].ptr<T>(borderRadius) + soffset + k*bwstep;
		//		//T* dst = dptr + j*ghstep + i*gwstep + k*step;
		//		memcpy(dst, ss, sizeof(T)*gwstep);
		//		dst+=gwstep;
		//	}	
		//}
	}
	
	Mat(dbuff(Rect(0,0,beforeSize.width, beforeSize.height))).copyTo(dest);
}

void mergeFromGrid_8u(vector<Mat>& src, Mat& dest, Size grid, int borderRadius)
{
	const int width = (src[0].cols-2*borderRadius)*grid.width;
	const int height = (src[0].rows-2*borderRadius)*grid.height;
	const int channels = src[0].channels();
	const int step = width*channels;

	if(dest.empty()) dest.create(Size(width,height),src[0].type());
	else if(dest.cols!=width ||dest.rows!=height) dest.create(Size(width,height),src[0].type());
	else if(dest.channels() !=channels) dest.create(Size(width,height),src[0].type());

	//grid num info
	const int gnumwidth = grid.width;
	const int gnumheight = grid.height;

	//grid info
	const int gwidth = src[0].cols-2*borderRadius;
	const int gheight = src[0].rows-2*borderRadius;
	const int gwstep = gwidth*channels;
	const int ghstep = gheight*step;

	const int bwidth = src[0].cols;
	const int bwstep = bwidth*channels;

	const int soffset = channels*borderRadius;
	const int svoffset = borderRadius*bwstep;
	//copy
	uchar* dptr = dest.ptr<uchar>(0);

#ifdef CV_SSE2
	const int ssesize = gwstep/16;
	const int sserem = gwstep - ssesize*16;
#endif
	for(int j=0;j<gnumheight;j++)
	{
		for(int i=0;i<gnumwidth;i++)
		{
			const int idx = gnumwidth*j+i;
			uchar* ss = src[idx].data + svoffset + soffset;
			uchar* dst = dptr + j*ghstep + i*gwstep;

			for(int k=gheight;k--;)
			{
#ifdef CV_SSE2
				uchar* s = ss;
				uchar* d = dst;
				for(int n=ssesize;n--;s+=16,d+=16)
				{
					_mm_storeu_si128((__m128i*) d, _mm_loadu_si128((const __m128i*)s));
				}
				for(int n=sserem;n--;s++,d++)
				{
					*d = *s;
				}
#else
				memcpy(dst, ss, gwstep);
#endif
				ss+=bwstep;
				dst+=step;
			}
		}
	}
}

void mergeFromGrid(vector<Mat>& src,  Size beforeSize, Mat& dest, Size grid, int borderRadius)
{
	int depth = src[0].depth();
	if(depth==CV_8U)
	{
		//mergeFromGrid_<uchar>(src,beforeSize,dest,grid,borderRadius);
		mergeFromGrid_8u(src,dest,grid,borderRadius);
	}
	else if(depth==CV_16U)
	{
		mergeFromGrid_<ushort>(src,beforeSize,dest,grid,borderRadius);
	}
	else if(depth==CV_16S)
	{
		mergeFromGrid_<short>(src,beforeSize,dest,grid,borderRadius);
	}
	else if(depth==CV_32F)
	{
		mergeFromGrid_<float>(src,beforeSize,dest,grid,borderRadius);
	}
	else if(depth==CV_64F)
	{
		mergeFromGrid_<double>(src,beforeSize,dest,grid,borderRadius);
	}
}