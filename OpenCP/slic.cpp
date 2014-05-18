
#include <opencv2/core/internal.hpp>

#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <limits>
#include <math.h>

using namespace std;
using namespace cv;
#include "opencp.hpp"

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
		CalcTime t("edge");
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
		CalcTime t("kmean init");
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
		CalcTime t("iter kmn");
		const float iregionSize = 1.0/(float)regionSize;
		for (iter = 0 ; iter < maxNumIterations ; ++iter)
		{	
			float factor = regularization / (regionSize * regionSize) ;
			float energy = 0 ;

			/* assign pixels to centers */
			for (y = 0 ; y < (signed)height ; ++y)
			{
				for (x = 0 ; x < (signed)width ; ++x)
				{
					int u = floor((float)x * iregionSize - 0.5f) ;
					int v = floor((float)y * iregionSize - 0.5f) ;
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
				float mass = MAX(masses[region], 1e-8) ;
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
		CalcTime t("Post");
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

class slic_segmentInvorker : public cv::ParallelLoopBody
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
	float* image;
	float* centers;
	int* segmentation;

public:

	slic_segmentInvorker(float* energy_, float* image_, float* centers_, int* segmentation_, int width_, int height_, int numChannels_, float iregionSize_, float factor_, int numRegionsX_, int numRegionsY_)
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
			/* assign pixels to centers */
			for (int y = r.start ; y < r.end ; ++y)
			{
				float* im0 = (float*)(image + (y)*width+ (0)*width*height);
				float* im1 = (float*)(image + (y)*width+ (1)*width*height);
				float* im2 = (float*)(image + (y)*width+ (2)*width*height);
				float* eng = energy + (y)*width;
				int* seg = &segmentation[y * width];

				const int v = floor((float)y * iregionSize - 0.5f) ;
				for (int x = 0 ; x < width ; ++x)
				{
					const int u = floor((float)x * iregionSize - 0.5f) ;

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
							const float spatial = centerx*centerx + centery*centery;
#ifdef CV_SSE3
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
							appearance += factor * spatial ;
							if (minDistance > appearance)
							{
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
					int u = floor((float)x * iregionSize - 0.5f) ;
					int v = floor((float)y * iregionSize - 0.5f) ;
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

void slic_segment (int* segmentation,
	float const * image,
	int width,
	int height,
	int numChannels,
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
	const unsigned int numRegions = numRegionsX * numRegionsY ;
	unsigned int const numPixels = width * height ;
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
	Mat eMap = Mat::zeros(Size(numPixels,1),CV_32F);
	float * edgeMap = eMap.ptr<float>(0);
	float previousEnergy = FLT_MAX;//VL_INFINITY_F ;
	float startingEnergy ;
	Mat massesm(Size(numPixels,1),CV_32S);
	unsigned int * masses = (unsigned int*)massesm.ptr<int>(0);

	assert(segmentation) ;
	assert(image) ;
	assert(width >= 1) ;
	assert(height >= 1) ;
	assert(numChannels >= 1) ;
	assert(regionSize >= 1) ;
	assert(regularization >= 0) ;

	{
		//CalcTime t("edge");
		/* compute edge map (gradient strength) */

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
		for (v = 0 ; v < (signed)numRegionsY ; ++v)
		{
			for (u = 0 ; u < (signed)numRegionsX ; ++u)
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
				for (yp = MAX(0, y-1) ; yp <= MIN((signed)height-1, y+1) ; ++ yp)
				{
					for (xp = MAX(0, x-1) ; xp <= MIN((signed)width-1, x+1) ; ++ xp)
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
		const float iregionSize = 1.0/(float)regionSize;

		float factor = regularization / (regionSize * regionSize) ;
		for (iter = 0 ; iter < maxNumIterations ; ++iter)
		{	
			float energy=0.f;
			slic_segmentInvorker body(en.ptr<float>(0),(float*)image,centers,segmentation,width,height,numChannels,iregionSize,factor,numRegionsX,numRegionsY);
			cv::parallel_for_(Range(0, height), body);

			energy =sum_32f(en);
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
			massesm.setTo(0);
			centerm.setTo(0);
			if(numChannels==3)
			{
				for (y = 0 ; y < height ; ++y)
				{
					float* im0 = (float*)(image + (y)*width+ (0)*numPixels);
					float* im1 = (float*)(image + (y)*width+ (1)*numPixels);
					float* im2 = (float*)(image + (y)*width+ (2)*numPixels);
					int * seg = &segmentation[(y)*width]; 
					for (x = 0 ; x < width ; ++x)
					{
						int region = *seg++;
						masses[region]++ ;
						float* c = &centers[region * cstep];
						c[0] += x ;
						c[1] += y ;
						c[2] += *im0++ ;
						c[3] += *im1++;
						c[4] += *im2++;
					}
				}
			}
			else if(numChannels==1)
			{
				for (y = 0 ; y < height ; ++y)
				{
					float* im0 = (float*)(image + (y)*width);
					for (x = 0 ; x < width ; ++x)
					{
						int pixel = x + y * width ;
						int region = segmentation[pixel] ;
						masses[region] ++ ;
						centers[region * cstep + 0] += x ;
						centers[region * cstep + 1] += y ;
						centers[region * cstep + 2] += *im0++ ;
					}
				}
			}

			for (region = 0 ; region < (signed)numRegions ; ++region)
			{
				float mass = 1.f / MAX(masses[region], 1e-8) ;
				for (i = (cstep) * region ;
					i < (signed)(cstep) * (region + 1) ;
					++i) 
				{
					centers[i] *= mass ;
				}
			}

		}
	}
	/* elimiate small regions */
	{
		//CalcTime t("Post");
		massesm.setTo(0);
		int* cleaned = massesm.ptr<int>(0);
		unsigned int * segment = (unsigned int*)malloc(sizeof(unsigned int) * numPixels) ;
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
			if (segmentSize < minRegionSize) {
				while (segmentSize > 0) {
					cleaned[segment[--segmentSize]] = cleanedLabel ;
				}
			}
		}
		/* restore base 0 indexing of the regions */
		//for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel) cleaned[pixel] -- ;
		subtract(massesm,1,massesm);

		memcpy(segmentation, cleaned, numPixels * sizeof(int)) ;
		free(segment) ;
	}
}


void SLIC(Mat& src, Mat& dest, unsigned int regionSize, float regularization, int minRegionSize, int max_iteration)
{
	if(dest.empty()||dest.size()!=src.size())dest=Mat::zeros(src.size(),CV_8U);
	else dest.setTo(0);
	Mat input,input_;
	if(src.channels()==3)
		cvtColorBGR2PLANE(src,input_);
	else
		input_ = src;

	input_.convertTo(input,CV_32F);
	Mat temp = Mat::zeros(src.size(),CV_32S);
	int maxiter = max_iteration;

	//slic_segment_base((int*)temp.data, (float*)input.data,src.cols,src.rows,src.channels(),regionSize,regularization,minRegionSize,maxiter);
	slic_segment((int*)temp.data, (float*)input.data,src.cols,src.rows,src.channels(),regionSize,regularization,(unsigned int)minRegionSize,maxiter);

	int* t=temp.ptr<int>(0);
	uchar* d = dest.ptr<uchar>(0);
	for(int i=0;i<temp.size().area()-temp.cols;i++)
	{
		if(t[i]!=t[i+1])d[i]=255;
		if(t[i]!=t[i+temp.cols])d[i]=255;
	}
}

void SLICBase(Mat& src, Mat& dest, unsigned int regionSize, float regularization, unsigned int minRegionSize, int max_iteration)
{
	if(dest.empty()||dest.size()!=src.size())dest=Mat::zeros(src.size(),CV_8U);
	else dest.setTo(0);
	Mat input,input_;
	if(src.channels()==3)
		cvtColorBGR2PLANE(src,input_);
	else
		input_ = src;

	input_.convertTo(input,CV_32F);
	Mat temp = Mat::zeros(src.size(),CV_32S);
	int maxiter = max_iteration;
	slic_segment_base((int*)temp.data, (float*)input.data, src.cols, src.rows, src.channels(),regionSize,regularization,minRegionSize, maxiter);

	int* t=temp.ptr<int>(0);
	uchar* d = dest.ptr<uchar>(0);
	for(int i=0;i<temp.size().area()-temp.cols;i++)
	{
		if(t[i]!=t[i+1])d[i]=255;
		if(t[i]!=t[i+temp.cols])d[i]=255;
	}
}