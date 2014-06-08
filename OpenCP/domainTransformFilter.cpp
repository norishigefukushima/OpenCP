#include "opencp.hpp"
#include <opencv2/core/internal.hpp>


//#define SSE_FUNC

//pow function

//#define USE_FAST_POW
// Fast pow function, referred from
// http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
double fastPow(double a, double b)
{
	union {
		double d;
		int x[2];
	} u = { a };
	u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
	u.x[0] = 0;
	return u.d;
}

// domain transform filter

//RF implimentations

class DomainTransformRFVertical_32F_Invoker : public cv::ParallelLoopBody
{
	int dim;

	Mat* out;
	Mat* dct;

public:
	DomainTransformRFVertical_32F_Invoker(Mat& out_, Mat& dct_, int dim_) :
		out(&out_), dct(& dct_), dim(dim_)
	{}

	virtual void operator() (const Range& range) const
	{
		int width = out->cols;
		int height = out->rows/dim;

		int stepo = out->cols;
		int stepd = dct->cols;

#ifdef SSE_FUNC
		__m128 ones = _mm_set1_ps(1.0);
#endif

		if(dim==1)
		{
			for(int x = range.start; x != range.end; x++)
			{
#ifdef SSE_FUNC
				float* o = out->ptr<float>(0);
				float* pp = dct->ptr<float>(0);
				o+=4*x;
				__m128 prev = _mm_load_ps(o);
				o+=stepo;
				pp+=4*x;

				for(int y=1; y<height-1; y++)
				{
					__m128 mp = _mm_load_ps(pp);
					__m128 mo = _mm_load_ps(o);
					//prev = *o = (1.0 - *pp) * *o + *pp * prev;
					prev = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(prev, mp));
					_mm_store_ps(o,prev);

					o+=stepo;
					pp+=stepd;
				}
				o-=stepo;
				prev = _mm_load_ps(o);
				for(int y=height-2; y>=0; y--)
				{
					__m128 mp = _mm_load_ps(pp);
					__m128 mo = _mm_load_ps(o);

					prev = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(prev, mp));
					_mm_store_ps(o,prev);

					o-=stepo;
					pp-=stepd;
				}
#else

				for(int x = range.start*4; x != range.end*4; x++)
				{
					float* v = (float*)out->data; v+=x;

					for(int y=1; y<height; y++)
					{
						float p = dct->at<float>(y-1, x);
						v[width*y] = (1.f - p) * v[width*y] + p * v[width*(y-1)];
					}

					for(int y=height-2; y>=0; y--)
					{
						float p = dct->at<float>(y, x);

						v[width*y] = (1.f - p) * v[width*y] + p * v[width*(y+1)];
					}
				}
#endif
			}
		}
		else if(dim==3)
		{
			const int istep = height*width;

#ifdef SSE_FUNC
			for(int x = range.start; x != range.end; x++)
			{
				float* b = (float*)out->data; b+=4*x;
				float* g = b+istep;
				float* r = g+istep;
				float* pp = dct->ptr<float>(0); pp+=4*x;
				__m128 preb = _mm_load_ps(b);
				__m128 preg = _mm_load_ps(g);
				__m128 prer = _mm_load_ps(r);

				b+=stepo;
				g+=stepo;
				r+=stepo;

				for(int y=height-2; y--;)
				{
					__m128 mp = _mm_load_ps(pp);

					__m128 mo = _mm_load_ps(b);
					preb = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(preb, mp));
					_mm_store_ps(b,preb);

					mo = _mm_load_ps(g);
					preg = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(preg, mp));
					_mm_store_ps(g,preg);

					mo = _mm_load_ps(r);
					prer = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(prer, mp));
					_mm_store_ps(r,prer);

					b+=stepo;
					g+=stepo;
					r+=stepo;
					pp+=stepd;
				}

				b-=stepo;g-=stepo;r-=stepo;

				for(int y=height-2; y--;)
				{
					__m128 mp = _mm_load_ps(pp);

					__m128 mo = _mm_load_ps(b);
					preb = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(preb, mp));
					_mm_store_ps(b,preb);

					mo = _mm_load_ps(g);
					preg = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(preg, mp));
					_mm_store_ps(g,preg);

					mo = _mm_load_ps(r);
					prer = _mm_add_ps(_mm_mul_ps(mo, _mm_sub_ps(ones,mp)) ,_mm_mul_ps(prer, mp));
					_mm_store_ps(r,prer);

					b-=stepo;
					g-=stepo;
					r-=stepo;
					pp-=stepd;
				}
			}
#else
			for(int x = range.start*4; x != range.end*4; x++)
			{
				float* b = (float*)out->data; b+=x;
				float* g = b+istep;
				float* r = g+istep;

				for(int y=1; y<height; y++)
				{
					float p = dct->at<float>(y-1, x);
					b[width*y] = (1.f - p) * b[width*y] + p * b[width*(y-1)];
					g[width*y] = (1.f - p) * g[width*y] + p * g[width*(y-1)];
					r[width*y] = (1.f - p) * r[width*y] + p * r[width*(y-1)];
				}

				for(int y=height-2; y>=0; y--)
				{
					float p = dct->at<float>(y, x);

					b[width*y] = (1.f - p) * b[width*y] + p * b[width*(y+1)];
					g[width*y] = (1.f - p) * g[width*y] + p * g[width*(y+1)];
					r[width*y] = (1.f - p) * r[width*y] + p * r[width*(y+1)];
				}
			}
#endif
		}
	}
};

// this function can be parallerized by using 4x4 blockwise transpose, but is this fast ?
class DomainTransformRFHorizontal_32F_Invoker : public cv::ParallelLoopBody
{
	int dim;

	Mat* out;
	Mat* dct;

public:
	DomainTransformRFHorizontal_32F_Invoker(Mat& out_, Mat& dct_, int dim_) :
		out(&out_), dct(& dct_), dim(dim_)
	{}

	virtual void operator() (const Range& range) const
	{
		int width = out->cols;
		int height = out->rows/dim;

		if(dim==1)
		{
			for(int y=range.start; y != range.end; y++)
			{
				float* o = out->ptr<float>(y);o++;	
				float* p = dct->ptr<float>(y);
				for(int x=1; x<width; x++)
				{
					*o = (1.f - *p) * *o + *p * *(o-1);
					p++;o++;
				}

				o-=2;p--;
				for(int x=width-2; x>=0; x--)
				{
					*o = (1.f - *p) * *o + *p * *(o+1);
					p--;o--;
				}
			}
		}
		else if(dim==3)
		{
			const int istep = height*width;

			for(int y=range.start; y != range.end; y++)
			{
				float* b = out->ptr<float>(y);b++;	
				float* g = b+istep;
				float* r = g+istep;

				float* p = dct->ptr<float>(y);

				for(int x=width-1;x--; )
				{	
					//*b = (1.f - *p) * *b + *p * *(b-1);
					//*g = (1.f - *p) * *g + *p * *(g-1);
					//*r = (1.f - *p) * *r + *p * *(r-1);
					*b += *p * (*(b-1) -*b);
					*g += *p * (*(g-1) -*g);
					*r += *p * (*(r-1) -*r);
					p++;
					b++;r++;g++;
				}
				p--;
				b-=2;r-=2;g-=2;
				for(int x=width-1;x--; )
				{
					//*b = (1.f - *p) * *b + *p * *(b+1);
					//*g = (1.f - *p) * *g + *p * *(g+1);
					//*r = (1.f - *p) * *r + *p * *(r+1);
					*b +=*p *(*(b+1)-*b);
					*g +=*p *(*(g+1)-*g);
					*r +=*p *(*(r+1)-*r);
					p--;
					b--;r--;g--;
				}
			}
		}
	}
};

class DomainTransformRFInit_32F_Invoker : public cv::ParallelLoopBody
{
	float a;
	float ratio;
	int dim;
	Mat* img;
	Mat* dctx;
	Mat* dcty;
public:
	DomainTransformRFInit_32F_Invoker(Mat& img_, Mat& dctx_, Mat& dcty_, float a_, float ratio_, int dim_) :
		img(&img_), dctx(& dctx_), dcty(& dcty_), a(a_), ratio(ratio_), dim(dim_)
	{
	}

	~DomainTransformRFInit_32F_Invoker()
	{
		int width = img->cols;
		int height = img->rows/dim;
		int ssewidth = ((width)/4);

#ifdef SSE_FUNC
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const __m128 mratio = _mm_set1_ps(ratio);
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
#endif

		// for last line
		if(dim==1)
		{

#ifdef SSE_FUNC
			float* s = img->ptr<float>(height-1);
			float* dx = dctx->ptr<float>(height-1);
			for(int x=ssewidth; x--;)
			{
				__m128 ms = _mm_load_ps(s);
				__m128 msp = _mm_loadu_ps(s+1);

				__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
				__m128 d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));
				_mm_stream_ps(dx,d);

				s+=4;
				dx+=4;
			}
#else
			float* v = img->ptr<float>(height-1);

			float* dx = dctx->ptr<float>(height-1);
			for(int x=0; x<width-1; x++)
			{
				float accumx = 0.0f;
				accumx += abs(v[x]-v[x+1]);

#ifdef USE_FAST_POW
				*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
#else
				*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
#endif
				dx++;
			}
#endif
		}
		else if(dim==3)
		{
			const int istep = height*width;
			float* b = img->ptr<float>(height-1);
			float* g = b+istep;
			float* r = g+istep;

			float* dx = dctx->ptr<float>(height-1);

#ifdef SSE_FUNC
			for(int x=ssewidth; x--;)
			{
				__m128 ms = _mm_load_ps(b);
				__m128 msp = _mm_loadu_ps(b+1);//h diff
				__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);

				ms = _mm_load_ps(g);
				msp = _mm_loadu_ps(g+1);//h diff
				w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

				ms = _mm_load_ps(r);
				msp = _mm_loadu_ps(r+1);//h diff
				w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

				_mm_stream_ps(dx, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w))));

				b+=4;
				g+=4;
				r+=4;
				dx+=4;
			}
#else
			for(int x=0; x<width-1; x++)
			{
				float accumx = 0.0f;
				accumx += abs(b[x]-b[x+1]);
				accumx += abs(g[x]-g[x+1]);
				accumx += abs(r[x]-r[x+1]);

#ifdef USE_FAST_POW
				*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
#else
				*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
#endif
				dx++;
			}
#endif
		}
	}

	virtual void operator() (const Range& range) const
	{
		int width = img->cols;
		int height = img->rows/dim;
		const int step = width;
		int ssewidth = ((width)/4);

#ifdef SSE_FUNC
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const __m128 mratio = _mm_set1_ps(ratio);
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
#endif

		if(dim==1)
		{

			for(int y = range.start; y != range.end; y++)
			{
#ifdef SSE_FUNC
				float* s = img->ptr<float>(y);

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<ssewidth; x++)
				{
					const __m128 ms = _mm_load_ps(s);
					__m128 msp = _mm_loadu_ps(s+1);//h diff

					__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
					__m128 d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));
					_mm_stream_ps(dx,d);

					msp = _mm_load_ps(s+step);//v diff
					w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
					d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));

					_mm_stream_ps(dy,d);

					s+=4;
					dx+=4;
					dy+=4;
				}
#else
				float* v = img->ptr<float>(y);

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);
				for(int x=0; x<width-1; x++)
				{
					float accumx = 0.0f;
					float accumy = 0.0f;
					accumx += abs(v[x]-v[x+1]);
					accumy += abs(v[x]-v[x+width]);

#ifdef USE_FAST_POW
					*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
					*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
					*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
					*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
					dx++;
					dy++;
				}
				float accumy = 0.0f;
				accumy += abs(v[width-1]-v[width-1+width]);
#ifdef USE_FAST_POW
				*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
				*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
#endif
			}

		}
		else if(dim==3)
		{
			const int istep = height*width;

			for(int y = range.start; y != range.end; y++)
			{
#ifdef SSE_FUNC
				float* b = img->ptr<float>(y);
				float* g = b+istep;
				float* r = g+istep;

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<ssewidth; x++)
				{

					/*for(int n=0;n<4;n++)
					{
					dx[n]=pow(a, 1.f + ratio*(abs(b[n]-b[n+1])+abs(g[n]-g[n+1])+abs(r[n]-r[n+1])));
					dy[n]=pow(a, 1.f + ratio*(abs(b[n]-b[n+step])+abs(g[n]-g[n+step])+abs(r[n]-r[n+step])));
					}*/

					__m128 ms = _mm_load_ps(b);
					__m128 msp = _mm_loadu_ps(b+1);//h diff
					__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);

					__m128 msv = _mm_load_ps(b+step);//v diff
					__m128 h =  _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask);

					ms = _mm_load_ps(g);
					msp = _mm_loadu_ps(g+1);//h diff
					w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

					msv = _mm_load_ps(g+step);//v diff
					h =  _mm_add_ps(h, _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask));

					ms = _mm_load_ps(r);
					msp = _mm_loadu_ps(r+1);//h diff
					w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

					msv = _mm_load_ps(r+step);//v diff
					h =  _mm_add_ps(h, _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask));

					_mm_stream_ps(dx, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w))));
					_mm_stream_ps(dy, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,h))));

					b+=4;
					g+=4;
					r+=4;
					dx+=4;
					dy+=4;
				}
#else
				float* b = img->ptr<float>(y);
				float* g = b+istep;
				float* r = g+istep;

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<width-1; x++)
				{
					float accumx = 0.0f;
					float accumy = 0.0f;
					accumx += abs(b[x]-b[x+1]);
					accumy += abs(b[x]-b[x+step]);
					accumx += abs(g[x]-g[x+1]);
					accumy += abs(g[x]-g[x+step]);
					accumx += abs(r[x]-r[x+1]);
					accumy += abs(r[x]-r[x+step]);

#ifdef USE_FAST_POW
					*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
					*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
					*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
					*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
					dx++;
					dy++;
				}
				float accumy = 0.0f;
				accumy += abs(b[width-1]-b[width-1+width]);
				accumy += abs(g[width-1]-g[width-1+width]);
				accumy += abs(r[width-1]-r[width-1+width]);
#ifdef USE_FAST_POW
				*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
				*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
#endif
			}
		}
	}
};


class DomainTransformCT_32F_Invoker : public cv::ParallelLoopBody
{
	float a;
	float ratio;
	int dim;
	Mat* img;
	Mat* dctx;
	Mat* dcty;
public:
	DomainTransformCT_32F_Invoker(Mat& img_, Mat& dctx_, Mat& dcty_, float a_, float ratio_, int dim_) :
		img(&img_), dctx(& dctx_), dcty(& dcty_), a(a_), ratio(ratio_), dim(dim_)
	{
	}

	~DomainTransformCT_32F_Invoker()
	{
		/*
		int width = img->cols;
		int height = img->rows/dim;
		int ssewidth = ((width)/4);

		#ifdef SSE_FUNC
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const __m128 mratio = _mm_set1_ps(ratio);
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
		#endif

		// for last line
		if(dim==1)
		{

		#ifdef SSE_FUNC
		float* s = img->ptr<float>(height-1);
		float* dx = dctx->ptr<float>(height-1);
		for(int x=ssewidth; x--;)
		{
		__m128 ms = _mm_load_ps(s);
		__m128 msp = _mm_loadu_ps(s+1);

		__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
		__m128 d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));
		_mm_stream_ps(dx,d);

		s+=4;
		dx+=4;
		}
		#else
		float* v = img->ptr<float>(height-1);

		float* dx = dctx->ptr<float>(height-1);
		for(int x=0; x<width-1; x++)
		{
		float accumx = 0.0f;
		accumx += abs(v[x]-v[x+1]);

		#ifdef USE_FAST_POW
		*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
		#else
		*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
		#endif
		dx++;
		}
		#endif
		}
		else if(dim==3)
		{
		const int istep = height*width;
		float* b = img->ptr<float>(height-1);
		float* g = b+istep;
		float* r = g+istep;

		float* dx = dctx->ptr<float>(height-1);

		#ifdef SSE_FUNC
		for(int x=ssewidth; x--;)
		{
		__m128 ms = _mm_load_ps(b);
		__m128 msp = _mm_loadu_ps(b+1);//h diff
		__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);

		ms = _mm_load_ps(g);
		msp = _mm_loadu_ps(g+1);//h diff
		w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

		ms = _mm_load_ps(r);
		msp = _mm_loadu_ps(r+1);//h diff
		w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

		_mm_stream_ps(dx, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w))));

		b+=4;
		g+=4;
		r+=4;
		dx+=4;
		}
		#else
		for(int x=0; x<width-1; x++)
		{
		float accumx = 0.0f;
		accumx += abs(b[x]-b[x+1]);
		accumx += abs(g[x]-g[x+1]);
		accumx += abs(r[x]-r[x+1]);

		#ifdef USE_FAST_POW
		*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
		#else
		*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
		#endif
		dx++;
		}
		#endif
		}
		*/
	}

	virtual void operator() (const Range& range) const
	{
		int width = img->cols;
		int height = img->rows/dim;
		const int step = width;
		int ssewidth = ((width)/4);

#ifdef SSE_FUNC
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const __m128 mratio = _mm_set1_ps(ratio);
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
#endif

		if(dim==1)
		{

			for(int y = range.start; y != range.end; y++)
			{
#ifdef SSE_FUNC
				float* s = img->ptr<float>(y);

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<ssewidth; x++)
				{
					const __m128 ms = _mm_load_ps(s);
					__m128 msp = _mm_loadu_ps(s+1);//h diff

					__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
					__m128 d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));
					_mm_stream_ps(dx,d);

					msp = _mm_load_ps(s+step);//v diff
					w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);
					d = _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w)));

					_mm_stream_ps(dy,d);

					s+=4;
					dx+=4;
					dy+=4;
				}
#else
				float* v = img->ptr<float>(y);

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);
				for(int x=0; x<width-1; x++)
				{
					float accumx = 0.0f;
					float accumy = 0.0f;
					accumx += abs(v[x]-v[x+1]);
					accumy += abs(v[x]-v[x+width]);

#ifdef USE_FAST_POW
					*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
					*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
					*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
					*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
					dx++;
					dy++;
				}
				float accumy = 0.0f;
				accumy += abs(v[width-1]-v[width-1+width]);
#ifdef USE_FAST_POW
				*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
				*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
#endif
			}

		}
		else if(dim==3)
		{
			const int istep = height*width;

			for(int y = range.start; y != range.end; y++)
			{
#ifdef SSE_FUNC
				float* b = img->ptr<float>(y);
				float* g = b+istep;
				float* r = g+istep;

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<ssewidth; x++)
				{

					/*for(int n=0;n<4;n++)
					{
					dx[n]=pow(a, 1.f + ratio*(abs(b[n]-b[n+1])+abs(g[n]-g[n+1])+abs(r[n]-r[n+1])));
					dy[n]=pow(a, 1.f + ratio*(abs(b[n]-b[n+step])+abs(g[n]-g[n+step])+abs(r[n]-r[n+step])));
					}*/

					__m128 ms = _mm_load_ps(b);
					__m128 msp = _mm_loadu_ps(b+1);//h diff
					__m128 w =  _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask);

					__m128 msv = _mm_load_ps(b+step);//v diff
					__m128 h =  _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask);

					ms = _mm_load_ps(g);
					msp = _mm_loadu_ps(g+1);//h diff
					w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

					msv = _mm_load_ps(g+step);//v diff
					h =  _mm_add_ps(h, _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask));

					ms = _mm_load_ps(r);
					msp = _mm_loadu_ps(r+1);//h diff
					w =  _mm_add_ps(w, _mm_and_ps(_mm_sub_ps(ms,msp), *(const __m128*)v32f_absmask));

					msv = _mm_load_ps(r+step);//v diff
					h =  _mm_add_ps(h, _mm_and_ps(_mm_sub_ps(ms,msv), *(const __m128*)v32f_absmask));

					_mm_stream_ps(dx, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,w))));
					_mm_stream_ps(dy, _mm_pow_ps(ma,_mm_add_ps(ones, _mm_mul_ps(mratio,h))));

					b+=4;
					g+=4;
					r+=4;
					dx+=4;
					dy+=4;
				}
#else
				float* b = img->ptr<float>(y);
				float* g = b+istep;
				float* r = g+istep;

				float* dx = dctx->ptr<float>(y);
				float* dy = dcty->ptr<float>(y);

				for(int x=0; x<width-1; x++)
				{
					float accumx = 0.0f;
					float accumy = 0.0f;
					accumx += abs(b[x]-b[x+1]);
					accumy += abs(b[x]-b[x+step]);
					accumx += abs(g[x]-g[x+1]);
					accumy += abs(g[x]-g[x+step]);
					accumx += abs(r[x]-r[x+1]);
					accumy += abs(r[x]-r[x+step]);

#ifdef USE_FAST_POW
					*dx = (float)fastPow(a, 1.0f + ratio * accumx); 
					*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
					*dx = (float)cv::pow(a, 1.0f + ratio * accumx); 
					*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
					dx++;
					dy++;
				}
				float accumy = 0.0f;
				accumy += abs(b[width-1]-b[width-1+width]);
				accumy += abs(g[width-1]-g[width-1+width]);
				accumy += abs(r[width-1]-r[width-1+width]);
#ifdef USE_FAST_POW
				*dy = (float)fastPow(a, 1.0f + ratio * accumy); 
#else
				*dy = (float)cv::pow(a, 1.0f + ratio * accumy); 
#endif
#endif
			}
		}
	}
};
void domainTransformFilter_RF_(cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, int maxiter)
{
	double sigma_r = max(sigma_range,0.0001);
	double sigma_s = max(sigma_space,0.0001);

	int dim = src.channels();

	int rem = (4 -(src.cols+1)%4)%4;

	Mat img;
	Mat temp;

	if(dim==1)
	{
		copyMakeBorder(src, temp, 0,0,0,rem+1,cv::BORDER_REPLICATE);

		if(src.depth()==CV_32F) img = temp;
		else temp.convertTo(img, CV_MAKETYPE(CV_32F,  dim));
	}
	else if(dim==3)
	{
		Mat temp2;
		copyMakeBorder(src, temp2, 0,0,0,rem+1,cv::BORDER_REPLICATE);

		cvtColorBGR2PLANE(temp2,temp);// 3channel image is conveted into long 1channel image

		if(src.depth()==CV_32F) img = temp;
		else temp.convertTo(img, CV_MAKETYPE(CV_32F,  dim));
	}

	int width = img.cols;
	int height = img.rows/dim;

	// compute derivatives of transformed domain "dct"
	// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
	cv::Mat dctx = cv::Mat::zeros(Size(width,height), CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(Size(width,height), CV_32FC1);
	float ratio = (float)(sigma_s / sigma_r);
	float a = (float)exp(-sqrt(2.0) / sigma_s);

	/*DomainTransformRFInit_32F_Invoker body(img, dctx, dcty, a, ratio, dim);
	parallel_for_(Range(0, height-1), body);

	for(int i=0;i<maxiter;i++)
	{
	DomainTransformRFHorizontal_32F_Invoker H(img, dctx, dim);
	parallel_for_(Range(0, height), H);

	DomainTransformRFVertical_32F_Invoker V(img, dcty, dim);
	parallel_for_(Range(0, width/4), V);			
	}*/



	for(int i=0;i<maxiter;i++)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);
		DomainTransformRFInit_32F_Invoker body(img, dctx, dcty, a, ratio, dim);
		parallel_for_(Range(0, height-1), body);

		DomainTransformRFHorizontal_32F_Invoker H(img, dctx, dim);
		parallel_for_(Range(0, height), H);

		DomainTransformRFVertical_32F_Invoker V(img, dcty, dim);
		parallel_for_(Range(0, width/4), V);			
	}
	if(dim==1)
	{
		if(src.depth()==CV_32F) 
			img(Rect(0,0,src.cols,src.rows)).copyTo(dst);
		else 
			img(Rect(0,0,src.cols,src.rows)).convertTo(dst, CV_MAKE_TYPE(CV_8U,dim));
	}
	else if(dim==3)
	{
		if(src.depth()==CV_32F) 
		{
			Mat temp;
			cvtColorPLANE2BGR(img,temp);
			temp(Rect(0,0,src.cols,src.rows)).copyTo(dst);
		}
		else if(src.depth()==CV_8U) 
		{
			Mat temp,temp2;
			img.convertTo(temp,CV_8U);
			cvtColorPLANE2BGR(temp,temp2);
			temp2(Rect(0,0,src.cols,src.rows)).copyTo(dst);
		}
		else
		{
			Mat temp;
			cvtColorPLANE2BGR(img,temp);
			temp(Rect(0,0,src.cols,src.rows)).convertTo(dst, CV_MAKE_TYPE(CV_8U,dim));
		}
	}
}


void domainTransformFilter(cv::Mat& src, cv::Mat& dst, double sigma_s, double sigma_r, int maxiter, int method)
{   
	//setNumThreads(4);

	if(method==DTF_RF)
	{
		domainTransformFilter_RF_(src, dst,sigma_s,sigma_r,maxiter);
	}
	else
	{
		printf("Now, Only Recursive Filtering is Supported.\n");
	}
}



/////////////////////////////////////////////////////////////////////////////////////////
//for base implimentation of recursive implimentation


// Recursive filter for vertical direction
void recursiveFilterVerticalB(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();

	for(int x=0; x<width; x++)
	{
		for(int y=1; y<height; y++)
		{
			float p = dct.at<float>(y-1, x);
			for(int c=0; c<dim; c++)
			{
				out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y-1, x*dim+c);
			}
		}

		for(int y=height-2; y>=0; y--)
		{
			float p = dct.at<float>(y, x);
			for(int c=0; c<dim; c++)
			{
				out.at<float>(y, x*dim+c) = p * out.at<float>(y+1, x*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}

// Recursive filter for horizontal direction
void recursiveFilterHorizontalB(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();

	for(int y=0; y<height; y++)
	{
		for(int x=1; x<width; x++)
		{
			float p = dct.at<float>(y, x-1);
			for(int c=0; c<dim; c++)
			{
				out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y, (x-1)*dim+c);
			}
		}

		for(int x=width-2; x>=0; x--)
		{
			float p = dct.at<float>(y, x);
			for(int c=0; c<dim; c++)
			{
				out.at<float>(y, x*dim+c) = p * out.at<float>(y, (x+1)*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}

void powMat(const float a , Mat& src, Mat & dest)
{
	if(dest.empty())dest.create(src.size(),CV_32F);

	int width = src.cols;
	int height = src.rows;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)  
		{
			dest.at<float>(y, x) = cv::pow(a, src.at<float>(y,x)); 
		}
	}
}

// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilterBase(const Mat& src, Mat& dest, double sigma_s, double sigma_r, int maxiter, int method)
{
	Mat img,out;
	src.convertTo(img, CV_MAKETYPE(CV_32F, src.channels()));

	int width = img.cols;
	int height = img.rows;
	int dim = img.channels();

	// compute derivatives of transformed domain "dct"
	// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (float)(sigma_s / sigma_r);

	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width-1; x++)
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				accum += abs(img.at<float>(y, (x+1)*dim+c) - img.at<float>(y, x*dim+c)); 
			}
			dctx.at<float>(y, x) = 1.0f + ratio * accum; 
		}
	}

	for(int y=0; y<height-1; y++)
	{
		for(int x=0; x<width; x++)  
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				accum += abs(img.at<float>(y+1, x*dim+c) - img.at<float>(y, x*dim+c)); 
			}

			dcty.at<float>(y, x) = 1.0f + ratio * accum; 
		}
	}

	// Apply recursive folter maxiter times
	img.convertTo(out, CV_MAKETYPE(CV_32F, dim));

	int i=maxiter;
	Mat amat;
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);

		//pow_fmath(a,dctx, amat);
		powMat(a,dctx, amat);
		recursiveFilterHorizontalB(out, amat);

		powMat(a,dcty, amat);
		//pow_fmath(a,dcty, amat);
		recursiveFilterVerticalB(out, amat);
	}
	out.convertTo(dest,src.type());
}

// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilterBase(const Mat& src, const Mat& guide, Mat& dest, double sigma_s, double sigma_r, int maxiter, int method)
{
	Mat img,out,joint;
	src.convertTo(img, CV_MAKETYPE(CV_32F, src.channels()));
	guide.convertTo(joint, CV_MAKETYPE(CV_32F, src.channels()));
	int width = img.cols;
	int height = img.rows;
	int dim = img.channels();

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (float)(sigma_s / sigma_r);

	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width-1; x++)
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				accum += abs(joint.at<float>(y, (x+1)*dim+c) - joint.at<float>(y, x*dim+c)); 
			}
			dctx.at<float>(y, x) = 1.0f + ratio * accum; 
		}
	}

	for(int y=0; y<height-1; y++)
	{
		for(int x=0; x<width; x++)  
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				accum += abs(joint.at<float>(y+1, x*dim+c) - joint.at<float>(y, x*dim+c)); 
			}

			dcty.at<float>(y, x) = 1.0f + ratio * accum; 
		}
	}


	img.convertTo(out, CV_MAKETYPE(CV_32F, dim));

	// Apply recursive folter maxiter times
	int i=maxiter;
	Mat amat;
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);

		//pow_fmath(a,dctx, amat);
		// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
		powMat(a,dctx, amat);
		recursiveFilterHorizontalB(out, amat);

		powMat(a,dcty, amat);
		//pow_fmath(a,dcty, amat);
		recursiveFilterVerticalB(out, amat);
	}
	out.convertTo(dest,src.type());
}