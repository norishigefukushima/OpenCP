//#include <opencv2/core/internal.hpp>

#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <limits>
#include <math.h>

#include "opencp.hpp"
using namespace cv;
using namespace std;

namespace cp
{
void slic_segment_base (int* segmentation,
						float const * image,
						unsigned int width,
						unsigned int height,
						unsigned int numChannels,
						unsigned int regionSize,
						float regularization,
						unsigned int minRegionSize,
						unsigned int const maxNumIterations
						)
{
	int i, x, y, u, v, k, region ;
	unsigned int iter ;
	unsigned int const numRegionsX = (unsigned int) ceil((double) width / regionSize) ;
	unsigned int const numRegionsY = (unsigned int) ceil((double) height / regionSize) ;
	unsigned int const numRegions = numRegionsX * numRegionsY ;
	unsigned int const numPixels = width * height ;
	float * centers ;
	float * edgeMap ;
	float previousEnergy = FLT_MAX;//VL_INFINITY_F ;
	float startingEnergy ;
	unsigned int * masses ;

	assert(segmentation) ;
	assert(image) ;
	assert(width >= 1) ;
	assert(height >= 1) ;
	assert(numChannels >= 1) ;
	assert(regionSize >= 1) ;
	assert(regularization >= 0) ;

#define atimage(x,y,k) image[(x)+(y)*width+(k)*width*height]
#define atEdgeMap(x,y) edgeMap[(x)+(y)*width]

	{
		//CalcTime t("edge");
		edgeMap = (float*)calloc(numPixels, sizeof(float)) ;
		masses = (unsigned int*)malloc(sizeof(unsigned int) * numPixels) ;
		centers = (float*)malloc(sizeof(float) * (2 + numChannels) * numRegions) ;

		/* compute edge map (gradient strength) */
		for (k = 0 ; k < (signed)numChannels ; ++k) {
			for (y = 1 ; y < (signed)height-1 ; ++y) {
				for (x = 1 ; x < (signed)width-1 ; ++x) {
					float a = atimage(x-1,y,k) ;
					float b = atimage(x+1,y,k) ;
					float c = atimage(x,y+1,k) ;
					float d = atimage(x,y-1,k) ;
					atEdgeMap(x,y) += (a - b)  * (a - b) + (c - d) * (c - d) ;
				}
			}
		}
	}

	{
		//CalcTime t("kmean init");
		/* initialize K-means centers */
		i = 0 ;
		for (v = 0 ; v < (signed)numRegionsY ; ++v) {
			for (u = 0 ; u < (signed)numRegionsX ; ++u) {
				int xp ;
				int yp ;
				int centerx ;
				int centery ;
				float minEdgeValue = FLT_MAX;//VL_INFINITY_F ;

				x = cvRound(regionSize * (u + 0.5)) ;
				y = cvRound(regionSize * (v + 0.5)) ;

				x = MAX(MIN(x, (signed)width-1),0) ;
				y = MAX(MIN(y, (signed)height-1),0) ;

				/* search in a 3x3 neighbourhood the smallest edge response */
				for (yp = MAX(0, y-1) ; yp <= MIN((signed)height-1, y+1) ; ++ yp) {
					for (xp = MAX(0, x-1) ; xp <= MIN((signed)width-1, x+1) ; ++ xp) {
						float thisEdgeValue = atEdgeMap(xp,yp) ;
						if (thisEdgeValue < minEdgeValue) {
							minEdgeValue = thisEdgeValue ;
							centerx = xp ;
							centery = yp ;
						}
					}
				}

				/* initialize the new center at this location */
				centers[i++] = (float) centerx ;
				centers[i++] = (float) centery ;
				for (k  = 0 ; k < (signed)numChannels ; ++k) {
					centers[i++] = atimage(centerx,centery,k) ;
				}
			}
		}

	}
	{

		/* run k-means iterations */
		//CalcTime t("iter kmn");
		const float iregionSize = 1.f/(float)regionSize;
		for (iter = 0 ; iter < maxNumIterations ; ++iter)
		{	
			float factor = regularization;
			float energy = 0.f ;

			/* assign pixels to centers */
			for (y = 0 ; y < (signed)height ; ++y)
			{
				for (x = 0 ; x < (signed)width ; ++x)
				{

					int u = cvFloor((float)x * iregionSize - 0.5f) ;
					int v = cvFloor((float)y * iregionSize - 0.5f) ;
					/*int u = floor((float)x /regionSize - 0.5f) ;
					int v = floor((float)y /regionSize - 0.5f) ;*/
					int up, vp ;
					float minDistance = FLT_MAX;

					for (vp = MAX(0, v) ; vp <= MIN((signed)numRegionsY-1, v+1) ; ++vp)
					{
						for (up = MAX(0, u) ; up <= MIN((signed)numRegionsX-1, u+1) ; ++up)
						{
							int region = up  + vp * numRegionsX ;
							float centerx = centers[(2 + numChannels) * region + 0]  ;
							float centery = centers[(2 + numChannels) * region + 1] ;
							float spatial = (x - centerx) * (x - centerx) + (y - centery) * (y - centery) ;
							float appearance = 0.f ;
							float distance ;
							for (k = 0 ; k < (signed)numChannels ; ++k)
							{
								float centerz = centers[(2 + numChannels) * region + k + 2]  ;
								float z = atimage(x,y,k) ;
								appearance += (z - centerz) * (z - centerz) ;
							}
							distance = appearance + factor * spatial ;
							if (minDistance > distance)
							{
								minDistance = distance ;
								segmentation[x + y * width] = (int)region ;
							}
						}
					}
					energy += minDistance ;
				}
			}
			/*
			VL_PRINTF("vl:slic: iter %d: energy: %g\n", iter, energy) ;
			*/

			/* check energy termination conditions */
			if (iter == 0)
			{
				startingEnergy = energy ;
			} 
			else 
			{
				if ((previousEnergy - energy) < 1e-5 * (startingEnergy - energy))
				{
					break ;
				}
			}
			previousEnergy = energy ;

			/* recompute centers */
			memset(masses, 0, sizeof(unsigned int) * width * height) ;
			memset(centers, 0, sizeof(float) * (2 + numChannels) * numRegions) ;

			for (y = 0 ; y < (signed)height ; ++y)
			{
				for (x = 0 ; x < (signed)width ; ++x)
				{
					int pixel = x + y * width ;
					int region = segmentation[pixel] ;
					masses[region] ++ ;
					centers[region * (2 + numChannels) + 0] += x ;
					centers[region * (2 + numChannels) + 1] += y ;
					for (k = 0 ; k < (signed)numChannels ; ++k) 
					{
						centers[region * (2 + numChannels) + k + 2] += atimage(x,y,k) ;
					}
				}
			}

			for (region = 0 ; region < (signed)numRegions ; ++region)
			{
				float mass = (float)MAX(masses[region], 1e-8) ;
				for (i = (2 + numChannels) * region ;
					i < (signed)(2 + numChannels) * (region + 1) ;
					++i) {
						centers[i] /= mass ;
				}
			}
		}
	}
	//cout<<iter<<endl;
	free(masses) ;
	free(centers) ;

	/* elimiate small regions */
	{
		//CalcTime t("Post");
		int* cleaned = (int*)calloc(numPixels, sizeof(int)) ;
		unsigned int * segment = (unsigned int*)malloc(sizeof(unsigned int) * numPixels) ;
		unsigned int segmentSize ;
		unsigned int label ;
		unsigned int cleanedLabel ;
		unsigned int numExpanded ;
		int const dx [] = {+1, -1,  0,  0} ;
		int const dy [] = { 0,  0, +1, -1} ;
		int direction ;
		int pixel ;

		for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel) {
			if (cleaned[pixel]) continue ;
			label = segmentation[pixel] ;
			numExpanded = 0 ;
			segmentSize = 0 ;
			segment[segmentSize++] = pixel ;

			/*
			find cleanedLabel as the label of an already cleaned
			region neihbour of this pixel
			*/
			cleanedLabel = label + 1 ;
			cleaned[pixel] = label + 1 ;
			x = pixel % width ;
			y = pixel / width ;
			for (direction = 0 ; direction < 4 ; ++direction) {
				int xp = x + dx[direction] ;
				int yp = y + dy[direction] ;
				int neighbor = xp + yp * width ;
				if (0 <= xp && xp < (signed)width &&
					0 <= yp && yp < (signed)height &&
					cleaned[neighbor]) {
						cleanedLabel = cleaned[neighbor] ;
				}
			}

			/* expand the segment */
			while (numExpanded < segmentSize) {
				int open = segment[numExpanded++] ;
				x = open % width ;
				y = open / width ;
				for (direction = 0 ; direction < 4 ; ++direction) {
					int xp = x + dx[direction] ;
					int yp = y + dy[direction] ;
					int neighbor = xp + yp * width ;
					if (0 <= xp && xp < (signed)width &&
						0 <= yp && yp < (signed)height &&
						cleaned[neighbor] == 0 &&
						segmentation[neighbor] == label) {
							cleaned[neighbor] = label + 1 ;
							segment[segmentSize++] = neighbor ;
					}
				}
			}

			/* change label to cleanedLabel if the semgent is too small */
			if (segmentSize < minRegionSize) {
				while (segmentSize > 0) {
					cleaned[segment[--segmentSize]] = cleanedLabel ;
				}
			}
		}
		/* restore base 0 indexing of the regions */
		for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel) cleaned[pixel] -- ;

		memcpy(segmentation, cleaned, numPixels * sizeof(int)) ;
		free(cleaned) ;
		free(segment) ;
	}
}

class SLIC_segmentInvorker : public cv::ParallelLoopBody
{
private:
	int width;
	int height;
	int numChannels;
	float iregionSize;
	float factor;
	int numRegionsX;
	int numRegionsY;

	float* energy;
	const float* image;
	float* centers;
	int* segmentation;

public:

	SLIC_segmentInvorker(float* energy_, const float* image_, float* centers_, int* segmentation_, int width_, int height_, int numChannels_, float iregionSize_, float factor_, int numRegionsX_, int numRegionsY_)
		:energy(energy_), image(image_), centers(centers_), segmentation(segmentation_), width(width_), height(height_), numChannels(numChannels_), iregionSize(iregionSize_), factor(factor_), numRegionsX(numRegionsX_), numRegionsY(numRegionsY_)
	{
		;
	}
	virtual void operator()( const cv::Range &r ) const 
	{

		float CV_DECL_ALIGNED(16) buf[4];
		if(numChannels==3)
		{
			const int cindex= 6;
			const int imstep = width*height;
			const int imstep2 = imstep*2;
			/* assign pixels to centers */
			for (int y = r.start ; y < r.end ; ++y)
			{
				float* im0 = (float*)(image + (y)*width);
				float* im1 = (float*)(image + (y)*width+imstep);
				float* im2 = (float*)(image + (y)*width+imstep2);
				float* eng = energy + (y)*width;
				int* seg = &segmentation[y * width];

				const int v = cvFloor((float)y * iregionSize - 0.5f) ;
				for (int x = 0 ; x < width ; ++x)
				{
					const int u = cvFloor((float)x * iregionSize - 0.5f) ;

					int up, vp ;
					float minDistance = FLT_MAX;
#ifdef CV_SSE3
					const __m128 s1 = _mm_set_ps(0.f,*im2++,*im1++,*im0++);
#else
					float z[3];
					z[0]=*im0++;
					z[1]=*im1++;
					z[2]=*im2++;
#endif
					const int vpend = MIN((signed)numRegionsY-1, v+1);
					const int upend = MIN((signed)numRegionsX-1, u+1);

					//if (v<0)printf("(x,y)=(%d,%d), (u,v)=(%d,%d), (ue,ve)=(%d,%d) numRegionsX-1 %d\n",x,y,u,v,upend,vpend,numRegionsX-1);
					for (vp = MAX(0, v) ; vp <= vpend ; ++vp)
					{
						for (up = MAX(0, u) ; up <= upend ; ++up)
						{
							const int region = up  + vp * numRegionsX ;

							float* c = &centers[cindex * region];
							const float centerx = (float)x-(c[0]);
							const float centery = (float)y-(c[1]);
							//ds in Eq(1) 
							const float spatial = centerx*centerx + centery*centery;
#ifdef CV_SSE3
							//dc in Eq(1) 
							__m128 s2 = _mm_loadu_ps((c+2));
							s2 = _mm_sub_ps(s1,s2);
							s2 = _mm_mul_ps(s2,s2);
							s2 = _mm_hadd_ps(s2,s2);
							s2 = _mm_hadd_ps(s2,s2);
							_mm_store_ps(buf,s2);
							float appearance = buf[0];
#else
							float appearance = (z[0] - c[2]) * (z[0] - c[2]) ;
							appearance += (z[1] - c[3]) * (z[1] - c[3]) ;
							appearance += (z[2] - c[4]) * (z[2] - c[4]) ;
#endif
							
							float cc = appearance;
							appearance += factor * spatial ;
							
							

							if (minDistance > appearance)
							{
//								std::cout<<cc<<","<<spatial<<","<<factor<<std::endl;
								minDistance = appearance;
								*seg = (int)region ;
							}
						}
					}
					seg++;
					*eng = minDistance ;
					eng++;
				}
			}
		}
		else if(numChannels==1)
		{
			const int cindex= 3;
			for (int y = r.start ; y < r.end ; ++y)
			{
				float* eng = energy + (y)*width;
				for (int x = 0 ; x < width ; ++x)
				{
					int u = cvFloor((float)x * iregionSize - 0.5f) ;
					int v = cvFloor((float)y * iregionSize - 0.5f) ;
					/*int u = floor((float)x /regionSize - 0.5f) ;
					int v = floor((float)y /regionSize - 0.5f) ;*/
					int up, vp ;
					float minDistance = FLT_MAX;

					int* seg = &segmentation[x + y * width];

					float g = atimage(x,y,0);
					const int vpend = MIN((signed)numRegionsY-1, v+1);
					const int upend = MIN((signed)numRegionsX-1, u+1);
					for (vp = MAX(0, v) ; vp <= vpend ; ++vp)
					{
						for (up = MAX(0, u) ; up <= upend ; ++up)
						{
							const int region = up  + vp * numRegionsX ;
							const float centerx = (float)x-centers[cindex * region + 0]  ;
							const float centery = (float)y-centers[cindex * region + 1] ;
							const float spatial = (centerx) * (centerx) + (centery) * (centery) ;
							float appearance = (g-centers[cindex * region  + 2])*(g-centers[cindex * region  + 2]) ;
							const float distance = appearance + factor * spatial ;
							if (minDistance > distance)
							{
								minDistance = distance ;
								*seg = (int)region ;
							}
						}
					}
					*eng = minDistance ;
					eng++;
				}
			}
		}
	}
};



float sum_hadd_32f(Mat& src)
{
	float ret=0.f;
	const int ssesize = src.size().area()/4;
	const int rems = src.size().area()-ssesize*4;

	float* s = src.ptr<float>(0);
	float CV_DECL_ALIGNED(16) buf[4];
	for(int i=0;i<ssesize;i++)
	{
		__m128 ms = _mm_load_ps(s);

		ms = _mm_hadd_ps(ms,ms);
		ms = _mm_hadd_ps(ms,ms);
		_mm_store_ps(buf,ms);
		ret+= buf[0];

		s+=4;
	}
	for(int i=0;i<rems;i++)
	{
		ret+= *s++;
	}
	return ret;
}

float sum_32f(Mat& src)
{
	float ret=0.f;
	const int ssesize = src.size().area()/4;
	const int rems = src.size().area()-ssesize*4;

	float* s = src.ptr<float>(0);
	__m128 total = _mm_setzero_ps();
	for(int i=0;i<ssesize;i++)
	{
		__m128 ms = _mm_load_ps(s);
		_mm_add_ps(total,ms);
		s+=4;
	}
	float CV_DECL_ALIGNED(16) buf[4];
	_mm_store_ps(buf,total);
	ret= buf[0]+buf[1]+buf[2]+buf[3];
	for(int i=0;i<rems;i++)
	{
		ret+= *s++;
	}
	return ret;
}

int getMax_int32(int* src, int size)
{
	int* s = src;
	int ssestep = 64;
	int ssesize = size/ssestep;
	int rem = size - ssestep*ssesize;
	__m128i maxval = _mm_set1_epi32(0);
	for(int i=ssesize;i--;)
	{
		//__m128i v = 
		//	_mm_max_epi32(
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_load_si128((const __m128i*)s  ), _mm_load_si128((const __m128i*)s+4)),
		//	_mm_max_epi32(_mm_load_si128((const __m128i*)s+8), _mm_load_si128((const __m128i*)s+12))
		//	),
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_load_si128((const __m128i*)s+16), _mm_load_si128((const __m128i*)s+20)),
		//	_mm_max_epi32(_mm_load_si128((const __m128i*)s+24), _mm_load_si128((const __m128i*)s+28))
		//	)
		//	)
		//	;

		__m128i v = 
			_mm_max_epi32(
			_mm_max_epi32(
			_mm_max_epi32(_mm_load_si128((const __m128i*)s  ), _mm_stream_load_si128((__m128i*)s+4)),
			_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+8), _mm_stream_load_si128((__m128i*)s+12))
			),
			_mm_max_epi32(
			_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+16), _mm_stream_load_si128((__m128i*)s+20)),
			_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+24), _mm_stream_load_si128((__m128i*)s+28))
			)
			)
			;

		//__m128i v = 
		//	_mm_max_epi32(
		//	_mm_max_epi32(
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_load_si128((const __m128i*)s  ), _mm_stream_load_si128((__m128i*)s+4)),
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+8), _mm_stream_load_si128((__m128i*)s+12))
		//	),
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+16), _mm_stream_load_si128((__m128i*)s+20)),
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+24), _mm_stream_load_si128((__m128i*)s+28))
		//	)
		//	),
		//	_mm_max_epi32(
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+32), _mm_stream_load_si128((__m128i*)s+36)),
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+40), _mm_stream_load_si128((__m128i*)s+44))
		//	),
		//	_mm_max_epi32(
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+48), _mm_stream_load_si128((__m128i*)s+52)),
		//	_mm_max_epi32(_mm_stream_load_si128((__m128i*)s+56), _mm_stream_load_si128((__m128i*)s+60))
		//	)
		//	)
		//	)
		;

		maxval = _mm_max_epi32(maxval,   v);


		s+=ssestep;
	}

	int CV_DECL_ALIGNED(16) buf[4];
	_mm_store_si128((__m128i*)buf, maxval);

	int ret = max(max(buf[0],buf[1]) , max(buf[1], buf[2]));

	for(int i=rem;i--;)
	{
		ret = max(ret,*s);
		s++;
	}
	return ret;
}

int getMax_int32(Mat& src)
{
	return getMax_int32(src.ptr<int>(0), src.size().area());
}

void drawSLIC(const Mat& src, Mat& segment, Mat& dest, bool isLine, Scalar line_color)
{
	if(dest.empty() || dest.size()!=src.size()) dest=Mat::zeros(src.size(),src.type());
	else dest.setTo(0);

	int maxseg = getMax_int32(segment)+1;	

	Mat numsegmentM = Mat::zeros(Size(maxseg,1),CV_32S);
	Mat valsegmentM = Mat::zeros(Size(maxseg,1),CV_32SC3);
	Mat ucharsegmentM = Mat::zeros(Size(maxseg,1),CV_8UC3);

	int* numsegment = numsegmentM.ptr<int>(0);
	int* valsegment = valsegmentM.ptr<int>(0);
	uchar* ucharsegment = ucharsegmentM.ptr<uchar>(0);

	int width = src.cols;
	int height = src.rows;

	if(src.channels()==3)
	{
		for (int y = 0 ; y < height ; ++y)
		{

			uchar* im = (uchar*)src.ptr<uchar>(y);
			uchar* d = dest.ptr<uchar>(y);
			int * seg = segment.ptr<int>(y);
			for (int x = 0 ; x < width ; ++x)
			{
				const int idx = *seg++;

				numsegment[idx]++;
				valsegment[3*idx  ]+=im[0];
				valsegment[3*idx+1]+=im[1];
				valsegment[3*idx+2]+=im[2];

				im+=3;
			}
		}
		for(int i=maxseg;i--;)
		{
			if(numsegment[i]!=0)
			{
				float div = 1.f/(float)numsegment[i];
				ucharsegment[3*i  ]= cvRound(valsegment[3*i  ]*div);
				ucharsegment[3*i+1]= cvRound(valsegment[3*i+1]*div);
				ucharsegment[3*i+2]= cvRound(valsegment[3*i+2]*div);
			}	
		}
		for (int y = 0 ; y < height ; ++y)
		{

			uchar* im = dest.ptr<uchar>(y);
			int * seg = segment.ptr<int>(y);
			for (int x = 0 ; x < width ; ++x)
			{
				const int idx = *seg++;

				numsegment[idx]++;
				im[0]=ucharsegment[3*idx  ];
				im[1]=ucharsegment[3*idx+1];
				im[2]=ucharsegment[3*idx+2];

				im+=3;
			}
		}
	}
	else if(src.channels()==1)
	{
		for (int y = 0 ; y < height ; ++y)
		{
			uchar* im = (uchar*)src.ptr<uchar>(y);
			int * seg = segment.ptr<int>(y);
			for (int x = 0 ; x < width ; ++x)
			{
				const int idx = *seg++;
				numsegment[idx]++;
				valsegment[idx]+=im[0];

				im++;
			}
		}
		for(int i=maxseg;i--;)
		{
			if(numsegment[i]!=0)
			{
				float div = 1.f/(float)numsegment[i];
				ucharsegment[i]= cvRound(valsegment[i]*div);
			}			
		}
		for (int y = 0 ; y < height ; ++y)
		{

			uchar* im = dest.ptr<uchar>(y);
			int * seg = segment.ptr<int>(y);
			for (int x = 0 ; x < width ; ++x)
			{
				const int idx = *seg++;

				numsegment[idx]++;
				im[0]=ucharsegment[idx];
				im++;
			}
		}
	}
	if(isLine)
	{
		int* t=segment.ptr<int>(0);
		uchar* d = dest.ptr<uchar>(0);
		uchar b =(uchar)line_color.val[0];
		uchar g =(uchar)line_color.val[1];
		uchar r =(uchar)line_color.val[2];
		if(src.channels()==1)
		{
			for(int i=0;i<src.size().area()-src.cols;i++)
			{
				if(t[i]!=t[i+1])d[i]=b;
				if(t[i]!=t[i+src.cols])d[i]=b;
			}
			for(int i=src.size().area()-src.cols;i<src.size().area()-1;i++)
			{
				if(t[i]!=t[i+1])d[i]=b;
			}
		}
		else
		{
			for(int i=0;i<src.size().area()-src.cols;i++)
			{
				if(t[i]!=t[i+1]){ d[3*i]=b; d[3*i+1]=g; d[3*i+2]=r;}
				if(t[i]!=t[i+src.cols]) { d[3*i]=b; d[3*i+1]=g; d[3*i+2]=r;}
			}
			for(int i=src.size().area()-src.cols;i<src.size().area()-1;i++)
			{
				if(t[i]!=t[i+1]){ d[3*i]=b; d[3*i+1]=g; d[3*i+2]=r;}
			}
		}
	}
}


/*
base for class SLIC_computeCentersInvorker

void computeCenters(Mat& massesm, Mat& centerm, const float* image, int* segmentation, int numChannels, int width, int height, int numRegions)
{
	int cstep = (numChannels==3) ? 6 :3;

	int const numPixels = width * height ;
	massesm.setTo(0);
	centerm.setTo(0);

	unsigned int * masses = (unsigned int*)massesm.ptr<int>(0);
	float* centers = centerm.ptr<float>(0) ;

	if(numChannels==3)
	{
		const int step = width;

		float* im0 = (float*)(image + (0)*width+ (0)*numPixels);
		float* im1 = (float*)(image + (0)*width+ (1)*numPixels);
		float* im2 = (float*)(image + (0)*width+ (2)*numPixels);
		for (int y = 0 ; y < height ; ++y)
		{
			int * seg = &segmentation[(y)*width]; 
			for (int x = 0 ; x < width ; ++x)
			{
				int region = *seg++;
				masses[region]++ ;
				float* c = &centers[region * cstep];
				c[0] += x ;
				c[1] += y ;
				c[2] += im0[x] ;
				c[3] += im1[x];
				c[4] += im2[x];
			}
			im0+=step;
			im1+=step;
			im2+=step;
		}

		for (int region = 0 ; region <numRegions ; ++region)
		{
			float mass = 1.f / MAX((float)masses[region], 1e-8) ;

			float* c = &centers[cstep * region];
			_mm_storeu_ps(c, _mm_mul_ps(_mm_loadu_ps(c), _mm_set1_ps(mass)));

			//c[0]*= mass;
			//c[1]*= mass;
			//c[2]*= mass;
			//c[3]*= mass;
			c[4]*= mass;
			//c[5]*= mass;
		}
	}
	else if(numChannels==1)
	{
		for (int y = 0 ; y < height ; ++y)
		{
			float* im0 = (float*)(image + (y)*width);
			for (int x = 0 ; x < width ; ++x)
			{
				int pixel = x + y * width ;
				int region = segmentation[pixel] ;
				masses[region] ++ ;
				centers[region * cstep + 0] += x ;
				centers[region * cstep + 1] += y ;
				centers[region * cstep + 2] += *im0++ ;
			}
		}

		for (int region = 0 ; region <numRegions ; ++region)
		{
			float mass = 1.f / MAX((float)masses[region], 1e-8) ;

			float* c = &centers[cstep * region];
			c[0]*= mass;
			c[1]*= mass;
			c[2]*= mass;
		}
	}
}
*/

class SLIC_computeCentersInvorker : public cv::ParallelLoopBody
{
private:
	const float* image;
	const int* segmentation;

	int width;
	int height;
	int numChannels;
	int segment_max;
	int nstrips;

	vector<Mat> cmat;
	vector<Mat> mmat;
	float** centerP;
	int** massesP;

	Mat* centerMat;
public:

	SLIC_computeCentersInvorker(const float* image_, const int* segmentation_, Mat& centers_, int width_, int height_, int numChannels_, int segment_max_, int nstrips_)
		:image(image_), centerMat(&centers_), segmentation(segmentation_), width(width_), height(height_), numChannels(numChannels_), segment_max(segment_max_), nstrips(nstrips_)
	{
		int cstep = (numChannels==3) ? 6 :3;
		cmat.resize(nstrips);
		mmat.resize(nstrips);
		centerP = new float*[nstrips];
		massesP = new int*[nstrips];

		for(int i=0;i<nstrips;i++)
		{
			cmat[i] = Mat::zeros(Size(segment_max,1),CV_MAKETYPE(CV_32F,cstep));
			centerP[i] = cmat[i].ptr<float>(0);

			mmat[i] = Mat::zeros(Size(segment_max,1),CV_32S);
			massesP[i]= mmat[i].ptr<int>(0);
		/*Mat* a = (Mat*)(&cmat[idx]);
		*a = Mat::zeros(Size(segment_max,1),CV_MAKETYPE(CV_32F,cstep));*/
		}
	}
	~SLIC_computeCentersInvorker()
	{
		int cstep = (numChannels==3) ? 6 :3;

		cmat[0].copyTo(*centerMat);
		for(int i=1;i<nstrips;i++)
		{
			add(*centerMat, cmat[i], *centerMat);
			add(mmat[0],mmat[i],mmat[0]);
			/*float* d = centerMat->ptr<float>(0);
			float* s = cmat[i].ptr<float>(0);
			for(int j=0;j<segment_max;j++)
			{
				d[6*j  ]+=s[6*j  ];
				d[6*j+1]+=s[6*j+1];
				d[6*j+2]+=s[6*j+2];
				d[6*j+3]+=s[6*j+3];
				d[6*j+4]+=s[6*j+4];
				d[6*j+5]+=s[6*j+5];
			}*/
		}

		int* masses = massesP[0];
		float* center = centerMat->ptr<float>(0);
		
		if(numChannels==3)
		{
			for (int region = 0 ; region < segment_max ; region++)
			{
				float mass = 1.f / (float)MAX(masses[region], 1e-8) ;

				float* c = center + cstep * region;
				_mm_storeu_ps(c, _mm_mul_ps(_mm_loadu_ps(c), _mm_set1_ps(mass)));
				//c[0]*= mass;//sse intrinsics set this value
				//c[1]*= mass;//sse intrinsics set this value
				//c[2]*= mass;//sse intrinsics set this value
				//c[3]*= mass;//sse intrinsics set this value

				c[4]*= mass;
				//c[5]*= mass;no use
			}
		}
		else
		{
			for (int region = 0 ; region <segment_max ; ++region)
			{
				float mass = 1.f / (float)MAX(masses[region], 1e-8) ;

				float* c = &center[cstep * region];
				c[0]*= mass;
				c[1]*= mass;
				c[2]*= mass;
			}
		}
		delete[] centerP;
		delete[] massesP;
	}


	virtual void operator()( const cv::Range &r ) const 
	{
		int cstep = (numChannels==3) ? 6 :3;
		int const numPixels = width * height ;
		const int thread_id = cvRound((double)r.start/double(height) * nstrips);
		
		float* center=centerP[thread_id]; 
		int* masses = massesP[thread_id];
		
		if(numChannels==3)
		{
			const int step = width;

			float* im0 = (float*)(image + (r.start)*width+ (0)*numPixels);
			float* im1 = (float*)(image + (r.start)*width+ (1)*numPixels);
			float* im2 = (float*)(image + (r.start)*width+ (2)*numPixels);

			for (int y = r.start ; y < r.end ; y++)
			{
				int * seg = (int*)(segmentation + (y)*width); 
				for (int x = 0 ; x < width ; ++x)
				{
					int region = *seg++;
					masses[region]++;
					float* c = &center[region * cstep];
					c[0] += x ;
					c[1] += y ;
					c[2] += im0[x];
					c[3] += im1[x];
					c[4] += im2[x];
				}
				im0+=step;
				im1+=step;
				im2+=step;
			}
		}
		else if(numChannels==1)
		{
			float* im0 = (float*)(image + (r.start)*width+ (0)*numPixels);
			for (int y = r.start ; y < r.end ; ++y)
			{
				float* im0 = (float*)(image + (y)*width);
				for (int x = 0 ; x < width ; ++x)
				{
					int pixel = x + y * width ;
					int region = segmentation[pixel] ;
					masses[region] ++ ;
					center[region * cstep + 0] += x ;
					center[region * cstep + 1] += y ;
					center[region * cstep + 2] += *im0++ ;
				}
			}
		}
	}
};

void slic_segment (int* segmentation,
				   float const * image,
				   int width,
				   int height,
				   int numChannels,
				   unsigned int regionSize,
				   float regularization,
				   unsigned int minRegionSize,
				   int const maxNumIterations
				   )
{
	const int threadnum = getNumThreads();
	int i, x, y, u, v, k;
	int iter ;
	int const numRegionsX = (unsigned int) ceil((double) width / regionSize) ;
	int const numRegionsY = (unsigned int) ceil((double) height / regionSize) ;
	const int numRegions = numRegionsX * numRegionsY ;
	int const numPixels = width * height ;
	Mat en(Size(width,height),CV_32F);
	Mat centerm;
	int cstep; 
	if(numChannels==3)
	{
		centerm = Mat::zeros(Size(numRegions,1), CV_MAKETYPE(CV_32F,6));
		cstep = 6;
	}
	else
	{
		centerm = Mat::zeros(Size(numRegions,1),CV_MAKETYPE(CV_32F,3));
		cstep = 3;
	}
	float* centers = centerm.ptr<float>(0) ;

	Mat massesm(Size(numPixels,1),CV_32S);
	unsigned int * masses = (unsigned int*)massesm.ptr<int>(0);

	Mat eMap = Mat::zeros(Size(numPixels,1),CV_32F);
	float * edgeMap = eMap.ptr<float>(0);
	float previousEnergy = FLT_MAX;//VL_INFINITY_F ;
	float startingEnergy ;


	assert(segmentation) ;
	assert(image) ;
	assert(width >= 1) ;
	assert(height >= 1) ;
	assert(numChannels >= 1) ;
	assert(regionSize >= 1) ;
	assert(regularization >= 0) ;

	{
		//the edge computation can be parallerized, but the part is not bottle neck.
		//CalcTime t("edge");
		//compute edge map (gradient strength)
		if(numChannels==3)
		{
			for (y = 1 ; y < (signed)height-1 ; ++y)
			{
				float* im0 = (float*)(image + (y)*width+ (0)*width*height);
				float* im1 = (float*)(image + (y)*width+ (1)*width*height);
				float* im2 = (float*)(image + (y)*width+ (2)*width*height);
				float* emap = (float*)(edgeMap+(y)*width);
				for (x = 1 ; x < (signed)width-1 ; ++x)
				{
					float a = im0[x-1] - im0[x+1];
					float v=a*a;
					a = im1[x-1] - im1[x+1];
					v+=a*a;
					a = im2[x-1] - im2[x+1];
					v+=a*a;
					a = im0[x-width] - im0[x+width];
					v+=a*a;
					a = im1[x-width] - im1[x+width];
					v+=a*a;
					a = im2[x-width] - im2[x+width];
					v+=a*a;
					emap[x]=v;
				}
			}
		}
		else if(numChannels==1)
		{
			for (y = 1 ; y < (signed)height-1 ; ++y)
			{
				float* im0 = (float*)(image + (y)*width);
				float* emap = (float*)(edgeMap+(y)*width);
				for (x = 1 ; x < (signed)width-1 ; ++x)
				{
					float a = im0[x-1] - im0[x+1];
					float v=a*a;
					a = im0[x-width] - im0[x+width];
					v+=a*a;
					emap[x]=v;
				}
			}
		}
	}

	{
		//CalcTime t("kmean init");
		/* initialize K-means centers */
		i = 0 ;
		float* c = &centers[0];
		for (v = 0 ; v < numRegionsY ; ++v)
		{
			for (u = 0 ; u < numRegionsX ; ++u)
			{
				int xp ;
				int yp ;
				int centerx ;
				int centery ;
				float minEdgeValue = FLT_MAX;//VL_INFINITY_F ;

				x = cvRound(regionSize * (u + 0.5)) ;
				y = cvRound(regionSize * (v + 0.5)) ;

				x = MAX(MIN(x, (signed)width-1),0) ;
				y = MAX(MIN(y, (signed)height-1),0) ;

				/* search in a 3x3 neighbourhood the smallest edge response */
				for (yp = MAX(0, y-1) ; yp <= MIN(height-1, y+1) ; ++ yp)
				{
					for (xp = MAX(0, x-1) ; xp <= MIN(width-1, x+1) ; ++ xp)
					{
						float thisEdgeValue = atEdgeMap(xp,yp) ;
						if (thisEdgeValue < minEdgeValue) {
							minEdgeValue = thisEdgeValue ;
							centerx = xp ;
							centery = yp ;
						}
					}
				}

				/* initialize the new center at this location */
				*c++ = (float) centerx ;
				*c++ = (float) centery ;
				if(numChannels==3)
				{
					for (k  = 0 ; k < (signed)numChannels ; ++k)
					{
						*c++= atimage(centerx,centery,k) ;
					}
					*c++;
				}
				else if(numChannels==1)
				{
					*c++= atimage(centerx,centery,0) ;
				}
			}
		}
	}

	{
		/* run k-means iterations */
		//CalcTime t("iter kmn");
		const float iregionSize = 1.f/(float)regionSize;

		float factor = regularization;
		for (iter = 0 ; iter < maxNumIterations ; ++iter)
		{	
			//CalcTime t("loop");
			SLIC_segmentInvorker body(en.ptr<float>(0), (float*)image, centers, segmentation, width, height, numChannels, iregionSize,factor,numRegionsX,numRegionsY);
			cv::parallel_for_(Range(0, height), body);

			float energy=energy =sum_32f(en);
			//Scalar v = sum(en);
			//energy = v[0];
			//energy =sum_hadd_32f(en);

			/* check energy termination conditions */
			if (iter == 0) startingEnergy = energy ; 
			else 
			{
				//if ((previousEnergy - energy) < 1e-5 * (startingEnergy - energy))
				if ((previousEnergy - energy) < 1e-3 * (startingEnergy - energy))
				{
					break ;
				}
			}
			previousEnergy = energy ;

			/* recompute centers */
			//need scope for destructor
			{
				//computeCenters(massesm, centerm, image,segmentation,numChannels,width,height,numRegions);

				SLIC_computeCentersInvorker body2(image, segmentation, centerm, width, height, numChannels, numRegions, threadnum);
				cv::parallel_for_(Range(0, height), body2, threadnum);
			}
			
		}
	}
	/* elimiate small regions */
	{
		//CalcTime t("Post");
		massesm.setTo(0);
		int* cleaned = massesm.ptr<int>(0);

		unsigned int * segment = (unsigned int*)fastMalloc(sizeof(unsigned int) * numPixels) ;
		unsigned int segmentSize ;
		unsigned int label ;
		unsigned int cleanedLabel ;
		unsigned int numExpanded ;
		int const dx [] = {+1, -1,  0,  0} ;
		int const dy [] = { 0,  0, +1, -1} ;
		int direction ;
		int pixel ;

		for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel)
		{
			if (cleaned[pixel]) continue ;
			label = segmentation[pixel] ;
			numExpanded = 0 ;
			segmentSize = 0 ;
			segment[segmentSize++] = pixel ;

			/*
			find cleanedLabel as the label of an already cleaned
			region neihbour of this pixel
			*/
			cleanedLabel = label + 1 ;
			cleaned[pixel] = label + 1 ;
			x = pixel % width ;
			y = pixel / width ;
			for (direction = 0 ; direction < 4 ; ++direction)
			{
				int xp = x + dx[direction] ;
				int yp = y + dy[direction] ;
				int neighbor = xp + yp * width ;
				if (0 <= xp && xp < (signed)width &&
					0 <= yp && yp < (signed)height &&
					cleaned[neighbor])
				{
					cleanedLabel = cleaned[neighbor] ;
				}
			}

			/* expand the segment */
			while (numExpanded < segmentSize)
			{
				int open = segment[numExpanded++] ;
				x = open % width ;
				y = open / width ;
				for (direction = 0 ; direction < 4 ; ++direction)
				{
					int xp = x + dx[direction] ;
					int yp = y + dy[direction] ;
					int neighbor = xp + yp * width ;
					if (0 <= xp && xp < (signed)width &&
						0 <= yp && yp < (signed)height &&
						cleaned[neighbor] == 0 &&
						segmentation[neighbor] == label)
					{
						cleaned[neighbor] = label + 1 ;
						segment[segmentSize++] = neighbor ;
					}
				}
			}

			/* change label to cleanedLabel if the semgent is too small */
			if (segmentSize < minRegionSize)
			{
				while (segmentSize > 0)
				{
					cleaned[segment[--segmentSize]] = cleanedLabel ;
				}
			}
		}

		/* restore base 0 indexing of the regions */
		//subtract(massesm,1,massesm);// cleaned = messes.ptr<int>(0);
		//memcpy(segmentation, cleaned, numPixels * sizeof(int)) ;

		int ssesize = numPixels/4;
		int rem = numPixels - ssesize*4;
		int* d = segmentation;
		int* s = cleaned;
		const __m128i ones = _mm_set1_epi32(1);
		for(int i=0;i<ssesize;i++)
		{
			__m128i ms = _mm_loadu_si128((const __m128i*)s);
			_mm_storeu_si128((__m128i*)d, _mm_sub_epi32(ms,ones));
			d+=4;
			s+=4;
		}
		for(int i=0;i<rem;i++)
		{
			*d = *s-1;
			*s++;
			*d++;
		}
		fastFree(segment) ;
	}
}

void SLIC(const Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration)
{
	//regionSize = S in the paper

	regionSize = max(4,regionSize);
	Mat input,input_;
	if(src.channels()==3)
		cvtColorBGR2PLANE(src,input_);
	else
		input_ = src;

	input_.convertTo(input,CV_32F);
	segment = Mat::zeros(src.size(),CV_32S);
	int maxiter = max_iteration;

	int minRegionSize = (int)(minRegionRatio*(regionSize*regionSize));
	float reg = (regularization*regularization)/(float)(regionSize*regionSize);
	
	slic_segment((int*)segment.data, (float*)input.data, src.cols, src.rows, src.channels(), regionSize, reg, minRegionSize, maxiter);
}

void SLICBase(Mat& src, Mat& segment,int regionSize, float regularization, float minRegionRatio, int max_iteration)
{
	regionSize = max(4,regionSize);
	Mat input,input_;
	if(src.channels()==3)
		cvtColorBGR2PLANE(src,input_);
	else
		input_ = src;

	input_.convertTo(input,CV_32F,1.0/255.0);
	segment = Mat::zeros(src.size(),CV_32S);
	int maxiter = max_iteration;
	int minRegionSize = (int)(minRegionRatio*(regionSize*regionSize));
	float reg = regularization* regionSize;
	slic_segment_base((int*)segment.data, (float*)input.data, src.cols, src.rows, src.channels(),regionSize,reg,minRegionSize, maxiter);
}
}