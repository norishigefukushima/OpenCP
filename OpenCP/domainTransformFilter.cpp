#include "opencp.hpp"
#include <opencv2/core/internal.hpp>


inline int pow2(int x)
{
	return x*x;
}

inline float pow2(float x)
{
	return x*x;
}

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
void domainTransformFilter_RF_(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, int maxiter)
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

	DomainTransformRFInit_32F_Invoker body(img, dctx, dcty, a, ratio, dim);
	parallel_for_(Range(0, height-1), body);

	for(int i=0;i<maxiter;i++)
	{
	DomainTransformRFHorizontal_32F_Invoker H(img, dctx, dim);
	parallel_for_(Range(0, height), H);

	DomainTransformRFVertical_32F_Invoker V(img, dcty, dim);
	parallel_for_(Range(0, width/4), V);			
	}



	/*for(int i=0;i<maxiter;i++)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);
		DomainTransformRFInit_32F_Invoker body(img, dctx, dcty, a, ratio, dim);
		parallel_for_(Range(0, height-1), body);

		DomainTransformRFHorizontal_32F_Invoker H(img, dctx, dim);
		parallel_for_(Range(0, height), H);

		DomainTransformRFVertical_32F_Invoker V(img, dcty, dim);
		parallel_for_(Range(0, width/4), V);			
	}*/
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

#include "fmath.hpp"
using namespace fmath;
inline __m128 _mm_pow_ps(__m128 a, __m128 b)
{
	return exp_ps(_mm_mul_ps(b,log_ps(a)));
}

inline float pow_fmath(float a, float b)
{
	return fmath::exp(b*fmath::log(a));
}

void pow_fmath(const float a , InputArray srcImg, Mat & dest)
{
	Mat src = srcImg.getMat();
	if(dest.empty())dest.create(src.size(),CV_32F);

	int width = src.cols;
	int height = src.rows;

	int size  =src.size().area();
	int i=0;

	const float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	const __m128 ma = _mm_set1_ps(a);
	for(i=0;i<=size-4;i+=4)
	{
		_mm_store_ps(d+i, _mm_pow_ps(ma, _mm_load_ps(s+i)));
	}
	for(;i<size;i++)
	{
		d[i] = cv::pow(a, s[i]); 
	}
}
class DomainTransformPowVerticalBGRA_SSE_Invoker : public cv::ParallelLoopBody
{
	float a;
	Mat* out;
	Mat* dct;
public:
	DomainTransformPowVerticalBGRA_SSE_Invoker(Mat& img_, Mat& dct_, float a_) :
		out(&img_), dct(& dct_), a(a_)
	{
		;//for(int x = range.start; x != range.end; x++)
	}
	virtual void operator() (const Range& range) const
	{
		int width = out->cols;
		int height = out->rows;
		int dim3 = 3;
		int dim = out->channels();
		const int step = 4*out->cols;
		const int dtstep = dct->cols;
		//printf("%d\n",dtstep);

		Mat dtbuff = Mat::zeros(Size(height,1),CV_32F);
		for(int x = range.start; x != range.end; x++)
		{
			int y=1;
			const __m128 ones = _mm_set1_ps(1.f);
			const __m128 ma = _mm_set1_ps(a);
			float* ptr = out->ptr<float>(0)+4*x;
			__m128 mpreo = _mm_loadu_ps(ptr);
			float* d = ptr +step;
			float* dt = dct->ptr<float>(0) + x;
			float* dtb = dtbuff.ptr<float>(0);
			int n=0;
			for(; n<=height-1-4; n+=4)
			{
				__m128 mp = _mm_set_ps(dt[(n+3)*dtstep], dt[(n+2)*dtstep],dt[(n+1)*dtstep], dt[(n)*dtstep]);
				_mm_store_ps(dtb+n, _mm_pow_ps(ma,mp));
				//pow(a, *dt);
			}
			for(; n<height-1; n++)
			{
				dtb[n]=pow_fmath(a, dt[n*dtstep]);
			}

			for(; y<height; y++)
			{
				const float p = *dtb;

				__m128 mp = _mm_set1_ps(p);
				__m128 imp = _mm_sub_ps(ones,mp);
				__m128 mo = _mm_loadu_ps(d);
				mpreo = _mm_add_ps( _mm_mul_ps(imp, mo), _mm_mul_ps(mp, mpreo));
				//mpreo = _mm_add_ps( mo, _mm_mul_ps(mp, _mm_sub_ps(mpreo,mo)));

				_mm_store_ps(d,mpreo);

				d+=step;
				dt+=dtstep;
				dtb++;
			}
			/*for( ;y<height; y++)
			{
			float p = dct.at<float>(y-1, x);
			for(int c=0; c<dim3; c++)
			{
			out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y-1, x*dim+c);
			}
			}*/

			y=height-2;
			d = ptr + (y)*step;
			dt = dct->ptr<float>(y) + x;
			dtb = dtbuff.ptr<float>(0)+y;
			for(; y>=0; y--)
			{
				const float p = *dtb;

				__m128 mp = _mm_set1_ps(p);
				__m128 imp = _mm_sub_ps(ones,mp);
				__m128 mo = _mm_loadu_ps(d);

				mpreo = _mm_add_ps( _mm_mul_ps(mp, mpreo), _mm_mul_ps(imp, mo));
				//mpreo = _mm_add_ps( mo, _mm_mul_ps(mp, _mm_sub_ps(mpreo,mo)));
				_mm_store_ps(d,mpreo);

				d-=step;
				dt-=dtstep;
				dtb--;
			}
			/*for(; y>=0; y--)
			{
			float p = dct.at<float>(y, x);
			for(int c=0; c<dim3; c++)
			{
			out.at<float>(y, x*dim+c) = p * out.at<float>(y+1, x*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
			}
			}*/
		}
	}
};

// Recursive filter for vertical direction
void recursiveFilterPowVerticalBGRA_SSE(cv::Mat& out, cv::Mat& dct, const float a) 
{
	int width = out.cols;
	int height = out.rows;
	int dim3 = 3;
	int dim = out.channels();

	const int step = 4*out.cols;
	const int dtstep = dct.cols;
	Mat dtbuff = Mat::zeros(Size(height,1),CV_32F);

	for(int x=0; x<width; x++)
	{
		int y=1;
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
		float* ptr = out.ptr<float>(0)+4*x;
		__m128 mpreo = _mm_loadu_ps(ptr);

		float* d = ptr +step;
		float* dt = dct.ptr<float>(0) + x;
		float* dtb = dtbuff.ptr<float>(0);
		int n=0;
		for(; n<=height-1-4; n+=4)
		{
			__m128 mp = _mm_set_ps(dt[(n+3)*dtstep], dt[(n+2)*dtstep],dt[(n+1)*dtstep], dt[(n)*dtstep]);
			_mm_store_ps(dtb+n, _mm_pow_ps(ma,mp));
		}
		for(; n<height-1; n++)
		{
			dtb[n]=pow_fmath(a, dt[n*dtstep]);
		}

		for(; y<height; y++)
		{
			const float p = *dtb;

			__m128 mp = _mm_set1_ps(p);
			__m128 imp = _mm_sub_ps(ones,mp);
			__m128 mo = _mm_loadu_ps(d);
			mpreo = _mm_add_ps( _mm_mul_ps(imp, mo), _mm_mul_ps(mp, mpreo));
			_mm_store_ps(d,mpreo);

			d+=step;
			dt+=dtstep;
			dtb++;
		}
		/*for( ;y<height; y++)
		{
		//float p = dct.at<float>(y-1, x);
		float p = dtb[y-1];
		for(int c=0; c<dim3; c++)
		{
		out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y-1, x*dim+c);
		}
		}*/

		y=height-2;
		d = ptr + (y)*step;
		dt = dct.ptr<float>(y) + x;
		dtb = dtbuff.ptr<float>(0)+y;
		for(; y>=0; y--)
		{
			const float p = *dtb;

			__m128 mp = _mm_set1_ps(p);
			__m128 imp = _mm_sub_ps(ones,mp);
			__m128 mo = _mm_loadu_ps(d);
			mpreo = _mm_add_ps( _mm_mul_ps(mp, mpreo), _mm_mul_ps(imp, mo));
			_mm_store_ps(d,mpreo);

			d-=step;
			dt-=dtstep;
			dtb--;
		}
		/*for(; y>=0; y--)
		{
		//float p = dct.at<float>(y, x);
		float p = dtb[y];
		for(int c=0; c<dim3; c++)
		{
		out.at<float>(y, x*dim+c) = p * out.at<float>(y+1, x*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
		}
		}*/
	}
}


class DomainTransformBuildDXDY_Invoker : public cv::ParallelLoopBody
{
	float ratio;
	const Mat* src;
	Mat* dx;
	Mat* dy;

public:
	~DomainTransformBuildDXDY_Invoker()
	{
		Mat joint = *src;
		int width = src->cols;
		int height = src->rows;
		int dim = src->channels();

		int y = height-1;
		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			for(int c=0; c<dim; c++)
			{
				accumx += abs(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c)); 
			}
			dx->at<float>(y, x) = 1.0f + ratio * accumx; 
		}
	}
	DomainTransformBuildDXDY_Invoker(const Mat& img_, Mat& dctx_, Mat& dcty_, float ratio_) :
		src(&img_), dx(& dctx_), dy(& dcty_),ratio(ratio_)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		int width = src->cols;
		int height = src->rows;
		int dim = src->channels();

		Mat joint = *src;

		for(int y = range.start; y != range.end; y++)
			//for(int y=0; y<height-1; y++)
		{
			uchar* jc = joint.ptr<uchar>(y);
			uchar* jp = joint.ptr<uchar>(y+1);
			float* dxp = dx->ptr<float>(y);
			float* dyp = dy->ptr<float>(y);

			for(int x=0; x<width-1; x++)
			{
				int accumx = 0;
				int accumy = 0;
				for(int c=0; c<dim; c++)
				{
					accumx += abs(jc[(x+1)*dim+c] - jc[x*dim+c]); 
					accumy += abs(jp[x*dim+c]     - jc[x*dim+c]); 
				}
				dxp[x]= 1.0f + ratio * accumx; 
				dyp[x]= 1.0f + ratio * accumy; 
			}
			int accumy = 0;
			int x = width -1;
			for(int c=0; c<dim; c++)
			{
				accumy += abs(jp[x*dim+c] - jc[x*dim+c]); 
			}
			dyp[x]= 1.0f + ratio * accumy; 	
		}
	}
};



class DomainTransformPowHorizontalBGRA_SSE_Invoker : public cv::ParallelLoopBody
{
	float a;
	Mat* out;
	Mat* dct;
public:
	DomainTransformPowHorizontalBGRA_SSE_Invoker(Mat& img_, Mat& dct_, float a_) :
		out(&img_), dct(& dct_), a(a_)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		int width = out->cols;
		int height = out->rows;
		int dim3 = 3;
		int dim = out->channels();
		Mat dtbuff = Mat::zeros(Size(width,1),CV_32F);

		//for(int y=0; y<height; y++)
		for(int y = range.start; y != range.end; y++)
		{
			float* dtb = dtbuff.ptr<float>(0); 
			float* d = out->ptr<float>(y);
			float* dt = dct->ptr<float>(y);

			int x;

			const __m128 ones = _mm_set1_ps(1.f);
			const __m128 ma = _mm_set1_ps(a);
			__m128 mpreo = _mm_loadu_ps(d);
			x=1;
			for(; x<=width-4; x+=4)
			{
				__m128 mps = _mm_loadu_ps(dt+x-1);
				mps = _mm_pow_ps(ma,mps);
				_mm_storeu_ps(dtb+x-1,mps);

				__m128 mo = _mm_loadu_ps(d+4*x);
				__m128 mp = _mm_shuffle_ps(mps, mps, 0x00); 
				__m128 imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
				_mm_storeu_ps(d+4*x,mpreo);

				mo = _mm_loadu_ps(d+4*(x+1));
				mp = _mm_shuffle_ps(mps, mps, 0x55); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
				_mm_storeu_ps(d+4*(x+1),mpreo);

				mo = _mm_loadu_ps(d+4*(x+2));
				mp = _mm_shuffle_ps(mps, mps, 0xAA); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
				_mm_storeu_ps(d+4*(x+2),mpreo);

				mo = _mm_loadu_ps(d+4*(x+3));
				mp = _mm_shuffle_ps(mps, mps, 0xFF); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
				_mm_storeu_ps(d+4*(x+3),mpreo);
			}
			for(; x<width; x++)
			{
				float p = pow_fmath(a, dct->at<float>(y, x-1));
				dtbuff.at<float>(x-1) = p;

				for(int c=0; c<dim3; c++)
				{
					out->at<float>(y, x*dim+c) = (1.f - p) * out->at<float>(y, x*dim+c) + p * out->at<float>(y, (x-1)*dim+c);
				}
			}

			mpreo = _mm_loadu_ps(d+4*(width-1));
			x=width-2;
			for(; x>4; x-=4)
			{
				__m128 mps = _mm_loadu_ps(dtb+x-3);
				//__m128 mps = _mm_loadu_ps(dt+x-3);
				//mps = _mm_pow_ps(ma,mps);

				__m128 mo = _mm_loadu_ps(d+4*x);
				__m128 mp = _mm_shuffle_ps(mps, mps, 0xFF); 
				__m128 imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
				_mm_storeu_ps(d+4*x,mpreo);

				mo = _mm_loadu_ps(d+4*(x-1));
				mp = _mm_shuffle_ps(mps, mps, 0xAA); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
				_mm_storeu_ps(d+4*(x-1),mpreo);

				mo = _mm_loadu_ps(d+4*(x-2));
				mp = _mm_shuffle_ps(mps, mps, 0x55); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
				_mm_storeu_ps(d+4*(x-2),mpreo);

				mo = _mm_loadu_ps(d+4*(x-3));
				mp = _mm_shuffle_ps(mps, mps, 0x00); 
				imp = _mm_sub_ps(ones,mp); 
				mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
				_mm_storeu_ps(d+4*(x-3),mpreo);
			}
			for(; x>=0; x--)
			{
				float p = dtbuff.at<float>(x);
				for(int c=0; c<dim3; c++)
				{
					out->at<float>(y, x*dim+c) = p * out->at<float>(y, (x+1)*dim+c) + (1.f - p) * out->at<float>(y, x*dim+c);
				}
			}
		}
	}
};

// Recursive filter for horizontal direction
void recursiveFilterPowHorizontalBGRA_SSE(cv::Mat& out, cv::Mat& dct, const float a) 
{
	int width = out.cols;
	int height = out.rows;
	int dim3 = 3;
	int dim = out.channels();
	Mat dtbuff = Mat::zeros(Size(width,1),CV_32F);

	for(int y=0; y<height; y++)
	{
		float* dtb = dtbuff.ptr<float>(0); 
		float* d = out.ptr<float>(y);
		float* dt = dct.ptr<float>(y);

		int x;

		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 ma = _mm_set1_ps(a);
		__m128 mpreo = _mm_loadu_ps(d);
		x=1;
		for(; x<=width-4; x+=4)
		{
			__m128 mps = _mm_loadu_ps(dt+x-1);
			mps = _mm_pow_ps(ma,mps);
			_mm_storeu_ps(dtb+x-1,mps);

			__m128 mo = _mm_loadu_ps(d+4*x);
			__m128 mp = _mm_shuffle_ps(mps, mps, 0x00); 
			__m128 imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*x,mpreo);

			mo = _mm_loadu_ps(d+4*(x+1));
			mp = _mm_shuffle_ps(mps, mps, 0x55); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+1),mpreo);

			mo = _mm_loadu_ps(d+4*(x+2));
			mp = _mm_shuffle_ps(mps, mps, 0xAA); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+2),mpreo);

			mo = _mm_loadu_ps(d+4*(x+3));
			mp = _mm_shuffle_ps(mps, mps, 0xFF); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+3),mpreo);
		}
		for(; x<width; x++)
		{
			float p = pow_fmath(a, dct.at<float>(y, x-1));
			dtbuff.at<float>(x-1) = p;

			for(int c=0; c<dim3; c++)
			{
				out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y, (x-1)*dim+c);
			}
		}

		mpreo = _mm_loadu_ps(d+4*(width-1));
		x=width-2;
		for(; x>4; x-=4)
		{
			__m128 mps = _mm_loadu_ps(dtb+x-3);
			//__m128 mps = _mm_loadu_ps(dt+x-3);
			//mps = _mm_pow_ps(ma,mps);

			__m128 mo = _mm_loadu_ps(d+4*x);
			__m128 mp = _mm_shuffle_ps(mps, mps, 0xFF); 
			__m128 imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*x,mpreo);

			mo = _mm_loadu_ps(d+4*(x-1));
			mp = _mm_shuffle_ps(mps, mps, 0xAA); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-1),mpreo);

			mo = _mm_loadu_ps(d+4*(x-2));
			mp = _mm_shuffle_ps(mps, mps, 0x55); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-2),mpreo);

			mo = _mm_loadu_ps(d+4*(x-3));
			mp = _mm_shuffle_ps(mps, mps, 0x00); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-3),mpreo);
		}
		for(; x>=0; x--)
		{
			float p = dtbuff.at<float>(x);
			for(int c=0; c<dim3; c++)
			{
				out.at<float>(y, x*dim+c) = p * out.at<float>(y, (x+1)*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}


// Recursive filter for vertical direction
void recursiveFilterVerticalBGRA_SSE(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	int dim3 = 3;
	int dim = out.channels();
	const int step = 4*out.cols;
	const int dtstep = dct.cols;
	for(int x=0; x<width; x++)
	{
		int y=1;
		const __m128 ones = _mm_set1_ps(1.f);
		float* ptr = out.ptr<float>(0)+4*x;
		__m128 mpreo = _mm_loadu_ps(ptr);

		float* d = ptr +step;
		float* dt = dct.ptr<float>(0) + x;
		for(; y<height; y++)
		{
			float p = *dt;

			__m128 mp = _mm_set1_ps(p);
			__m128 imp = _mm_sub_ps(ones,mp);
			__m128 mo = _mm_loadu_ps(d);

			//(1-p) *a + p*b
			//a + p*(b-a)

			mpreo = _mm_add_ps( _mm_mul_ps(imp, mo), _mm_mul_ps(mp, mpreo));
			//mpreo = _mm_add_ps( mo, _mm_mul_ps(mp, _mm_sub_ps(mpreo,mo)));
			_mm_store_ps(d,mpreo);

			d+=step;
			dt+=dtstep;
		}
		/*for( ;y<height; y++)
		{
		float p = dct.at<float>(y-1, x);
		for(int c=0; c<dim3; c++)
		{
		out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y-1, x*dim+c);
		}
		}*/

		y=height-2;
		d = ptr + (y)*step;
		dt = dct.ptr<float>(y) + x;
		for(; y>=0; y--)
		{
			float p = *dt;

			__m128 mp = _mm_set1_ps(p);
			__m128 imp = _mm_sub_ps(ones,mp);
			__m128 mo = _mm_loadu_ps(d);
			mpreo = _mm_add_ps( _mm_mul_ps(mp, mpreo), _mm_mul_ps(imp, mo));
			//mpreo = _mm_add_ps( mo, _mm_mul_ps(mp, _mm_sub_ps(mpreo,mo)));

			_mm_store_ps(d,mpreo);

			d-=step;
			dt-=dtstep;
		}
		/*for(; y>=0; y--)
		{
		float p = dct.at<float>(y, x);
		for(int c=0; c<dim3; c++)
		{
		out.at<float>(y, x*dim+c) = p * out.at<float>(y+1, x*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
		}
		}*/
	}
}

// Recursive filter for horizontal direction
void recursiveFilterHorizontalBGRA_SSE(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	int dim3 = 3;
	int dim = out.channels();

	for(int y=0; y<height; y++)
	{
		float* d = out.ptr<float>(y);
		float* dt = dct.ptr<float>(y);
		int x;
		const __m128 ones = _mm_set1_ps(1.f);
		__m128 mpreo = _mm_loadu_ps(d);

		x=1;
		for(; x<=width-4; x+=4)
		{
			__m128 mps = _mm_loadu_ps(dt+x-1);

			__m128 mo = _mm_loadu_ps(d+4*x);
			__m128 mp = _mm_shuffle_ps(mps, mps, 0x00); 
			__m128 imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*x,mpreo);

			mo = _mm_loadu_ps(d+4*(x+1));
			mp = _mm_shuffle_ps(mps, mps, 0x55); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+1),mpreo);

			mo = _mm_loadu_ps(d+4*(x+2));
			mp = _mm_shuffle_ps(mps, mps, 0xAA); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+2),mpreo);

			mo = _mm_loadu_ps(d+4*(x+3));
			mp = _mm_shuffle_ps(mps, mps, 0xFF); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(imp,mo), _mm_mul_ps(mp,mpreo));
			_mm_storeu_ps(d+4*(x+3),mpreo);
		}
		for(; x<width; x++)
		{
			float p = dct.at<float>(y, x-1);
			for(int c=0; c<dim3; c++)
			{
				out.at<float>(y, x*dim+c) = (1.f - p) * out.at<float>(y, x*dim+c) + p * out.at<float>(y, (x-1)*dim+c);
			}
		}

		mpreo = _mm_loadu_ps(d+4*(width-1));
		x=width-2;
		for(; x>4; x-=4)
		{
			__m128 mps = _mm_loadu_ps(dt+x-3);

			__m128 mo = _mm_loadu_ps(d+4*x);
			__m128 mp = _mm_shuffle_ps(mps, mps, 0xFF); 
			__m128 imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*x,mpreo);

			mo = _mm_loadu_ps(d+4*(x-1));
			mp = _mm_shuffle_ps(mps, mps, 0xAA); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-1),mpreo);

			mo = _mm_loadu_ps(d+4*(x-2));
			mp = _mm_shuffle_ps(mps, mps, 0x55); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-2),mpreo);

			mo = _mm_loadu_ps(d+4*(x-3));
			mp = _mm_shuffle_ps(mps, mps, 0x00); 
			imp = _mm_sub_ps(ones,mp); 
			mpreo = _mm_add_ps( _mm_mul_ps(mp,mpreo), _mm_mul_ps(imp,mo));
			_mm_storeu_ps(d+4*(x-3),mpreo);
		}
		for(; x>=0; x--)
		{
			float p = dct.at<float>(y, x);
			for(int c=0; c<dim3; c++)
			{
				out.at<float>(y, x*dim+c) = p * out.at<float>(y, (x+1)*dim+c) + (1.f - p) * out.at<float>(y, x*dim+c);
			}
		}
	}
}

/*
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
(0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0,0,0,0,0,0,0,0,0,0);

const __m128i bmask2 = _mm_setr_epi8
(0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0,0,0,0,0);

const __m128i bmask3 = _mm_setr_epi8
(0xFF,0xFF,0xFF,0xFF,0xFF,0,0,0,0,0,0,0,0,0,0,0);

const __m128i bmask4 = _mm_setr_epi8
(0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0,0,0,0,0,0);	

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
*/


/*
void buid_ct_L_8u(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio, int th)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;
	
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	//#pragma omp parallel for
	for(int y=0; y<height-1; y++)
	{
		uchar* jc = joint.ptr<uchar>(y);
		uchar* jp = joint.ptr<uchar>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);
		

		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			int accumy = 0;
			for(int c=0; c<dim; c++)
			{
				//accumx += (abs(jc[(x+1)*dim+c] - jc[x*dim+c])<th) ? 0 :abs(jc[(x+1)*dim+c] - jc[x*dim+c]); 
				//accumy += (abs(jp[x*dim+c]     - jc[x*dim+c])<th) ? 0 : abs(jp[x*dim+c]     - jc[x*dim+c]); 
				accumx += max(abs(jc[(x+1)*dim+c] - jc[x*dim+c])-th,0); 
				accumy += max(abs(jp[x*dim+c]     - jc[x*dim+c])-th,0);
			}
			ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
			cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
		}
		int accumy = 0;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += max(abs(jp[x*dim+c] - jc[x*dim+c])-th,0); 
		}
		cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		int accumx = 0;
		for(int c=0; c<dim; c++)
		{
			accumx += max(abs(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c))-th,0); 
		}
		ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
	}
}
*/
void buid_ct_L1_8u(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;
	
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	//#pragma omp parallel for
	for(int y=0; y<height-1; y++)
	{
		uchar* jc = joint.ptr<uchar>(y);
		uchar* jp = joint.ptr<uchar>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);
		
		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			int accumy = 0;
			for(int c=0; c<dim; c++)
			{
				accumx += abs(jc[(x+1)*dim+c] - jc[x*dim+c]); 
				accumy += abs(jp[x*dim+c]     - jc[x*dim+c]); 
			}
			ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
			cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
		}
		int accumy = 0;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += abs(jp[x*dim+c] - jc[x*dim+c]); 
		}
		cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		int accumx = 0;
		for(int c=0; c<dim; c++)
		{
			accumx += abs(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c)); 
		}
		ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
	}
}

void buid_ct_L1_32f(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;
	
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	//#pragma omp parallel for
	for(int y=0; y<height-1; y++)
	{
		float* jc = joint.ptr<float>(y);
		float* jp = joint.ptr<float>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);
		

		for(int x=0; x<width-1; x++)
		{
			float accumx = 0.f;
			float accumy = 0.f;
			for(int c=0; c<dim; c++)
			{
				accumx += abs(jc[(x+1)*dim+c] - jc[x*dim+c]); 
				accumy += abs(jp[x*dim+c]     - jc[x*dim+c]); 
			}
			ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
			cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
		}
		float accumy = 0.f;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += abs(jp[x*dim+c] - jc[x*dim+c]); 
		}
		cty[x]= ctyp[x]+ 1.0f + ratio * accumy; 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		float accumx = 0.f;
		for(int c=0; c<dim; c++)
		{
			accumx += abs(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c)); 
		}
		ctx[x+1] = ctx[x] + 1.0f + ratio * accumx; 
	}
}

void buid_ct_absL2_8u(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();
	const float ratio2 = ratio*ratio;

	Mat joint = src;
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	for(int y=0; y<height-1; y++)
	{
		uchar* jc = joint.ptr<uchar>(y);
		uchar* jp = joint.ptr<uchar>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);

		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			int accumy = 0;
			for(int c=0; c<dim; c++)
			{
				accumx += abs((int)(jc[(x+1)*dim+c] - jc[x*dim+c])); 
				accumy += abs((int)(jp[x*dim+c]     - jc[x*dim+c])); 
			}
			accumx = pow2(accumx); 
			accumy = pow2(accumy); 
			//ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
			//cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
			ctx[x+1] = ctx[x] + sqrt(1.f + ratio2 * accumx); 
			cty[x]= ctyp[x]+ sqrt(1.f + ratio2 * accumy); 
		}
		int accumy = 0;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += abs((int)(jp[x*dim+c]     - jc[x*dim+c])); 
		}
		accumy = pow2(accumy); 
		//cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
		cty[x]= ctyp[x]+ sqrt(1.f + ratio2 * accumy); 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		int accumx = 0;
		for(int c=0; c<dim; c++)
		{
			accumx += abs((int)(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c))); 
		}
		accumx = pow2(accumx); 
		//ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
		ctx[x+1] = ctx[x] + sqrt(1.f + ratio2 * accumx); 
	}
}

void buid_ct_L2_8u(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();
	const float ratio2 = ratio*ratio;

	Mat joint = src;
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	for(int y=0; y<height-1; y++)
	{
		uchar* jc = joint.ptr<uchar>(y);
		uchar* jp = joint.ptr<uchar>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);

		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			int accumy = 0;
			for(int c=0; c<dim; c++)
			{
				accumx += pow2((int)(jc[(x+1)*dim+c] - jc[x*dim+c])); 
				accumy += pow2((int)(jp[x*dim+c]     - jc[x*dim+c])); 
			}
			//ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
			//cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
			ctx[x+1] = ctx[x] + sqrt(1.f + ratio2 * accumx); 
			cty[x]= ctyp[x]+ sqrt(1.f + ratio2 * accumy); 
		}
		int accumy = 0;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += pow2((int)(jp[x*dim+c] - jc[x*dim+c])); 
		}
		//cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
		cty[x]= ctyp[x]+ sqrt(1.f + ratio2 * accumy); 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		int accumx = 0;
		for(int c=0; c<dim; c++)
		{
			accumx += pow2((int)(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c))); 
		}
		//ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
		ctx[x+1] = ctx[x] + sqrt(1.f + ratio2 * accumx); 
	}
}

void buid_ct_L2_32f(const Mat& src, Mat& ct_x, Mat& ct_y, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();
	const float ratio2 = ratio*ratio;

	Mat joint = src;
	memset(ct_y.ptr<float>(0), 0, sizeof(float)*width);

	for(int y=0; y<height-1; y++)
	{
		float* jc = joint.ptr<float>(y);
		float* jp = joint.ptr<float>(y+1);
		float* ctx = ct_x.ptr<float>(y);
		ctx[0]=0.f;
		float* ctyp = ct_y.ptr<float>(y);
		float* cty = ct_y.ptr<float>(y+1);

		for(int x=0; x<width-1; x++)
		{
			float accumx = 0.f;
			float accumy = 0.f;
			for(int c=0; c<dim; c++)
			{
				accumx += pow2(jc[(x+1)*dim+c] - jc[x*dim+c]); 
				accumy += pow2(jp[x*dim+c]     - jc[x*dim+c]); 
			}
			ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
			cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
		}
		float accumy = 0.f;
		int x = width -1;
		for(int c=0; c<dim; c++)
		{
			accumy += pow2(jp[x*dim+c] - jc[x*dim+c]); 
		}
		cty[x]= ctyp[x]+ sqrt(ratio2 + ratio2 * accumy); 
	}

	int y = height-1;
	float* ctx = ct_x.ptr<float>(y);
	ctx[0]=0.f;
	for(int x=0; x<width-1; x++)
	{
		float accumx = 0.f;
		for(int c=0; c<dim; c++)
		{
			accumx += pow2(joint.at<float>(y, (x+1)*dim+c) - joint.at<float>(y, x*dim+c)); 
		}
		ctx[x+1] = ctx[x] + sqrt(ratio2 + ratio2 * accumx); 
	}
}

void cunsum_32f(const Mat& src, Mat& dest)
{
	for(int j=0;j<src.rows;j++)
	{
		float* s = (float*)src.ptr<float>(j);
		float* d = dest.ptr<float>(j);

		d[0]=s[0];
		for(int i=1;i<src.cols;i++)
		{
			d[i]=s[i]+d[i-1];
		}
	}
}

void cunsum_32f(Mat& inplace)
{
	for(int j=0;j<inplace.rows;j++)
	{
		float* s = (float*)inplace.ptr<float>(j);
		for(int i=1;i<inplace.cols;i++)
		{
			s[i]+=s[i-1];
		}
	}
}

void buid_dxdyL1_8u(const Mat& src, Mat& dx, Mat& dy, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;

	if(dim==1)
	{
		for(int y=0; y<height-1; y++)
		{
			uchar* jc = joint.ptr<uchar>(y);
			uchar* jp = joint.ptr<uchar>(y+1);
			float* dxp = dx.ptr<float>(y);
			float* dyp = dy.ptr<float>(y);

			for(int x=0; x<width-1; x++)
			{
				dxp[x]= 1.0f + ratio * abs(jc[x+1] - jc[x]); 
				dyp[x]= 1.0f + ratio * abs(jp[x]   - jc[x]); 
			}
			int accumy = 0;
			int x = width -1;

			dyp[x]= 1.0f + ratio * abs(jp[x] - jc[x]); 	
		}

		int y = height-1;
		for(int x=0; x<width-1; x++)
		{
			dx.at<float>(y, x) = 1.0f + ratio * abs(joint.at<uchar>(y, (x+1)) - joint.at<uchar>(y, x)); 
		}
	}
	else
	{
		for(int y=0; y<height-1; y++)
		{
			uchar* jc = joint.ptr<uchar>(y);
			uchar* jp = joint.ptr<uchar>(y+1);
			float* dxp = dx.ptr<float>(y);
			float* dyp = dy.ptr<float>(y);

			for(int x=0; x<width-1; x++)
			{
				int accumx = 0;
				int accumy = 0;
				for(int c=0; c<dim; c++)
				{
					accumx += abs(jc[(x+1)*dim+c] - jc[x*dim+c]); 
					accumy += abs(jp[x*dim+c]     - jc[x*dim+c]); 
				}
				dxp[x]= 1.0f + ratio * accumx; 
				dyp[x]= 1.0f + ratio * accumy; 
			}
			int accumy = 0;
			int x = width -1;
			for(int c=0; c<dim; c++)
			{
				accumy += abs(jp[x*dim+c] - jc[x*dim+c]); 
			}
			dyp[x]= 1.0f + ratio * accumy; 	
		}


		int y = height-1;
		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			for(int c=0; c<dim; c++)
			{
				accumx += abs(joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c)); 
			}
			dx.at<float>(y, x) = 1.0f + ratio * accumx; 
		}
	}
}

void buid_dxdyL2_8u(const Mat& src, Mat& dx, Mat& dy, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;
	const float ratio2 = ratio*ratio;

	/*if(dim==1)
	{
		for(int y=0; y<height-1; y++)
		{
			uchar* jc = joint.ptr<uchar>(y);
			uchar* jp = joint.ptr<uchar>(y+1);
			float* dxp = dx.ptr<float>(y);
			float* dyp = dy.ptr<float>(y);

			for(int x=0; x<width-1; x++)
			{
				dxp[x]= sqrt(ratio2 + ratio2 * (jc[x+1] - jc[x])*(jc[x+1] - jc[x])); 
				dyp[x]= sqrt(ratio2 + ratio2 * (jp[x]   - jc[x])*(jp[x]   - jc[x])); 
			}
			int accumy = 0;
			int x = width -1;

			dyp[x]= sqrt(ratio2 + ratio2 * (jp[x] - jc[x])*(jp[x] - jc[x])); 	
		}

		int y = height-1;
		for(int x=0; x<width-1; x++)
		{
			dx.at<float>(y, x) = sqrt(ratio2 + ratio2 * (joint.at<uchar>(y, (x+1)) - joint.at<uchar>(y, x))*(joint.at<uchar>(y, (x+1)) - joint.at<uchar>(y, x))); 
		}
	}
	else*/
	{
		for(int y=0; y<height-1; y++)
		{
			uchar* jc = joint.ptr<uchar>(y);
			uchar* jp = joint.ptr<uchar>(y+1);
			float* dxp = dx.ptr<float>(y);
			float* dyp = dy.ptr<float>(y);

			for(int x=0; x<width-1; x++)
			{
				int accumx = 0;
				int accumy = 0;
				for(int c=0; c<dim; c++)
				{
					int v = (jc[(x+1)*dim+c] - jc[x*dim+c]);
					accumx += v*v; 
					v = (jp[x*dim+c]     - jc[x*dim+c]);
					accumy += v*v; 
				}
				dxp[x]= sqrt(ratio2 + ratio2 * accumx); 
				dyp[x]= sqrt(ratio2 + ratio2 * accumy); 
			}
			int accumy = 0;
			int x = width -1;
			for(int c=0; c<dim; c++)
			{
				int v = (jp[x*dim+c] - jc[x*dim+c]);
				accumy +=v*v; 
			}
			dyp[x]= sqrt(ratio2 + ratio2 * accumy); 
		}

		int y = height-1;
		for(int x=0; x<width-1; x++)
		{
			int accumx = 0;
			for(int c=0; c<dim; c++)
			{
				int v = (joint.at<uchar>(y, (x+1)*dim+c) - joint.at<uchar>(y, x*dim+c));
				accumx += v*v; 
			}
			dx.at<float>(y, x) = sqrt(ratio2 + ratio2 * accumx); 
		}
	}
}

void buid_dxdyL1_32f(const Mat& src, Mat& dx, Mat& dy, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;

	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width-1; x++)
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				accum += abs(joint.at<float>(y, (x+1)*dim+c) - joint.at<float>(y, x*dim+c)); 
			}
			dx.at<float>(y, x) = 1.0f + ratio * accum; 
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

			dy.at<float>(y, x) = 1.0f + ratio * accum; 
		}
	}
}

void buid_dxdyL2_32f(const Mat& src, Mat& dx, Mat& dy, const float ratio)
{
	int width = src.cols;
	int height = src.rows;
	int dim = src.channels();

	Mat joint = src;
	float ratio2 = ratio*ratio;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width-1; x++)
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				float v = joint.at<float>(y, (x+1)*dim+c) - joint.at<float>(y, x*dim+c);
				accum += v*v; 
			}
			dx.at<float>(y, x) = sqrt(ratio2 + ratio2 * accum); 
		}
	}
	for(int y=0; y<height-1; y++)
	{
		for(int x=0; x<width; x++)  
		{
			float accum = 0.0f;
			for(int c=0; c<dim; c++)
			{
				float v = joint.at<float>(y+1, x*dim+c) - joint.at<float>(y, x*dim+c);
				accum += v*v; 
			}

			dy.at<float>(y, x) = sqrt(ratio2 + ratio2 * accum); 
		}
	}
}


// Domain transform filtering: fast implimentation for optimization BGR2BGR2 ->SSE optimization
void domainTransformFilter_RF_BGRA_SSE_PARALLEL(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	Mat img;
	cvtColorBGR8u2BGRA32f(src,img);

	int width = img.cols;
	int height = img.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) 
		{
			DomainTransformBuildDXDY_Invoker B(guide, dctx, dcty, ratio);
			parallel_for_(Range(0, height-1), B);
		}
		else if(norm == DTF_L2) buid_dxdyL2_8u(guide, dctx,dcty,ratio);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_dxdyL1_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_32f(guidef, dctx,dcty,ratio);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;
	Mat out = img.clone();
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"

		float a = (float)exp(-sqrt(2.0) / sigma_h);
		{
			DomainTransformPowHorizontalBGRA_SSE_Invoker H(out, dctx, a);
			parallel_for_(Range(0, height), H);			
		}
		{
			DomainTransformPowVerticalBGRA_SSE_Invoker V(out, dcty, a);
			parallel_for_(Range(0, width), V);			
		}
		
	}
	cvtColorBGRA32f2BGR8u(out,dest);
}

//single rgba
// Domain transform filtering: fast implimentation for optimization BGR2BGR2 ->SSE optimization
void domainTransformFilter_RF_BGRA_SSE_SINGLE(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	Mat img;
	cvtColorBGR8u2BGRA32f(src,img);
	
	

	int width = img.cols;
	int height = img.rows;
	
	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_dxdyL1_8u(guide, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_8u(guide, dctx,dcty,ratio);	
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);

		if(norm == DTF_L1) buid_dxdyL1_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_32f(guidef, dctx,dcty,ratio);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
		float a = (float)exp(-sqrt(2.0) / sigma_h);
		recursiveFilterPowHorizontalBGRA_SSE(img, dctx, a);
		
		recursiveFilterPowVerticalBGRA_SSE(img, dcty, a);
		
	}

	cvtColorBGRA32f2BGR8u(img,dest);
}

void domainTransformFilter_RF_BGRA_SSE_SINGLE(const Mat& src, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	domainTransformFilter_RF_BGRA_SSE_SINGLE(src, src, dest, sigma_r, sigma_s, maxiter, norm);
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

/////////////////////////////////////////////////////////////////////////////////////////
//for base implimentation of recursive implimentation
// Recursive filter for vertical direction

void recursiveFilterPowVerticalGraySSE(cv::Mat& out, cv::Mat& dct, float a) 
{

	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();
	int x=0;

		const __m128 ma = _mm_set1_ps(a);
	for(x=0; x<=width-4; x+=4)
	{
		const __m128 ones = _mm_set1_ps(1.f);
		for(int y=1; y<height; y++)
		{
			float* pdct = dct.ptr<float>(y-1);

			float* puout = out.ptr<float>(y-1);
			float* pout = out.ptr<float>(y);

			//float p = dct.at<float>(y-1, x);
			__m128 mp = _mm_loadu_ps(pdct+x);
			mp = _mm_pow_ps(ma, mp);
			__m128 mo = _mm_loadu_ps(pout+x);
			__m128 muo = _mm_loadu_ps(puout+x);

			//_mm_mul_ps
			_mm_storeu_ps(pout+x,_mm_add_ps(_mm_mul_ps (_mm_sub_ps(ones,mp), mo), _mm_mul_ps (mp, muo)));
		}

		for(int y=height-2; y>=0; y--)
		{
			float* pdct = dct.ptr<float>(y);

			float* puout = out.ptr<float>(y+1);
			float* pout = out.ptr<float>(y);

			//float p = dct.at<float>(y-1, x);
			__m128 mp = _mm_loadu_ps(pdct+x);
			__m128 mo = _mm_loadu_ps(pout+x);
			__m128 muo = _mm_loadu_ps(puout+x);

			//_mm_mul_ps
			_mm_storeu_ps(pout+x,_mm_add_ps(_mm_mul_ps (_mm_sub_ps(ones,mp), mo), _mm_mul_ps (mp, muo)));

		}
	}
	for(; x<width; x++)
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

void recursiveFilterVerticalGraySSE(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	int dim = out.channels();
	int x=0;
	for(x=0; x<=width-4; x+=4)
	{
		const __m128 ones = _mm_set1_ps(1.f);
		for(int y=1; y<height; y++)
		{
			float* pdct = dct.ptr<float>(y-1);

			float* puout = out.ptr<float>(y-1);
			float* pout = out.ptr<float>(y);

			//float p = dct.at<float>(y-1, x);
			__m128 mp = _mm_loadu_ps(pdct+x);
			__m128 mo = _mm_loadu_ps(pout+x);
			__m128 muo = _mm_loadu_ps(puout+x);

			//_mm_mul_ps
			_mm_storeu_ps(pout+x,_mm_add_ps(_mm_mul_ps (_mm_sub_ps(ones,mp), mo), _mm_mul_ps (mp, muo)));
		}

		for(int y=height-2; y>=0; y--)
		{
			float* pdct = dct.ptr<float>(y);

			float* puout = out.ptr<float>(y+1);
			float* pout = out.ptr<float>(y);

			//float p = dct.at<float>(y-1, x);
			__m128 mp = _mm_loadu_ps(pdct+x);
			__m128 mo = _mm_loadu_ps(pout+x);
			__m128 muo = _mm_loadu_ps(puout+x);

			//_mm_mul_ps
			_mm_storeu_ps(pout+x,_mm_add_ps(_mm_mul_ps (_mm_sub_ps(ones,mp), mo), _mm_mul_ps (mp, muo)));

		}
	}
	for(; x<width; x++)
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
void recursiveFilterVerticalBGR(cv::Mat& out, cv::Mat& dct) 
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
/*
// Recursive filter for horizontal direction
void recursiveFilterHorizontalBGR(cv::Mat& out, cv::Mat& dct) 
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
*/

void recursiveFilterHorizontalGray(cv::Mat& out, cv::Mat& dct) 
{
	int width = out.cols;
	int height = out.rows;
	
	for(int y=0; y<height; y++)
	{
		float* pdct = dct.ptr<float>(y); 
		float* o = out.ptr<float>(y);o++;
		for(int x=1; x<width; x++)
		{
			//float p = dct.at<float>(y, x-1);
			float p = *pdct++;
			//out.at<float>(y, x) = (1.f - p) * out.at<float>(y, x) + p * out.at<float>(y, (x-1));

			*o = (1.f - p) * *o + p * *(o-1);
			o++;
		}

		o-=2;
		pdct--;
		for(int x=width-2; x>=0; x--)
		{
			float p = *(pdct--);
			*o = p * *(o+1) + (1.f-p) * *o;
			o--;
		}
	}
}

void recursiveFilterHorizontalBGR(cv::Mat& out, cv::Mat& dct) 
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

void domainTransformFilter_RF_GRAY_SSE_SINGLE(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
		Mat img;
	src.convertTo(img, CV_32F);

	int width = src.cols;
	int height = src.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_dxdyL1_8u(guide, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_8u(guide, dctx,dcty,ratio);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_dxdyL1_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_32f(guidef, dctx,dcty,ratio);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;
	Mat amat,amat2;

//	Mat dctxt;transpose(dctx,dctxt);
	Mat temp;
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);

		// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
		pow_fmath(a,dctx, amat);
		recursiveFilterHorizontalGray(img, amat);

		pow_fmath(a,dcty, amat2);
		recursiveFilterVerticalGraySSE(img, amat2);
		// transpose approach
		//transpose(img,temp);
		//pow_fmath(a,dctxt, amat);
		//recursiveFilterVerticalGraySSE(temp, amat);
		//transpose(temp,img);

	}
	//out.convertTo(dest,src.type(),1.0,0.5);
	img.convertTo(dest,src.type(), 1.0, 0.5);
}
// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilter_RF_Base(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	Mat img;
	src.convertTo(img, CV_32F);

	int width = src.cols;
	int height = src.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width-1, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height-1, width, CV_32FC1);
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_dxdyL1_8u(guide, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_8u(guide, dctx,dcty,ratio);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_dxdyL1_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_dxdyL2_32f(guidef, dctx,dcty,ratio);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;
	Mat amat,amat2;

	Mat dctxt;transpose(dctx,dctxt);
	Mat temp;
	while(i--)
	{
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		float a = (float)exp(-sqrt(2.0) / sigma_h);

		// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
		//powMat(a,dctx, amat);
		pow_fmath(a,dctx, amat);
		recursiveFilterHorizontalBGR(img, amat);

		//powMat(a,dcty, amat2);
		pow_fmath(a,dcty, amat2);
		recursiveFilterVerticalBGR(img, amat2);


	}
	//out.convertTo(dest,src.type(),1.0,0.5);
	img.convertTo(dest,src.type(), 1.0, 0.5);
}

// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilter_RF_Base(const Mat& src, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	domainTransformFilter_RF_Base(src,src,dest,sigma_r,sigma_s,maxiter, norm);
}

void domainTransformFilterRF(const Mat& src, const Mat& guide, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{   
	//setNumThreads(4);

	if(implementation==DTF_SLOWEST)
	{
		domainTransformFilter_RF_Base(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
	else if(implementation==DTF_BGRA_SSE)
	{
		if(src.channels()==1) domainTransformFilter_RF_GRAY_SSE_SINGLE(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
		else	domainTransformFilter_RF_BGRA_SSE_SINGLE(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
	else if(implementation==DTF_BGRA_SSE_PARALLEL)
	{
		if(src.channels()==1) domainTransformFilter_RF_GRAY_SSE_SINGLE(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
		else domainTransformFilter_RF_BGRA_SSE_PARALLEL(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
}

void domainTransformFilterRF(const Mat& src, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{
	domainTransformFilterRF(src,src,dst,sigma_r,sigma_s,maxiter,norm,implementation);
}


///////////////////////////////////////////////////////////////////////////////
//IC
///////////////////////////////////////////////////////////////////////////////

void interpolatedConvolutionX_32f(const Mat& src_,Mat& dest,const float radius, const Mat& ctx_)
{
	if(dest.empty()) dest.create(src_.size(),src_.type());

	Mat src;
	copyMakeBorder(src_, src,0,0,1,1,BORDER_REPLICATE);
	Mat ctx;
	copyMakeBorder(ctx_, ctx,0,0,1,1,BORDER_CONSTANT,Scalar::all(1));

	if(src.channels()==1)
	{
		for(int j=0; j<src.rows; j++)
		{
			const float* s = src.ptr<float>(j)+1;
			const float* dx = ctx.ptr<float>(j)+1;
			float* d = dest.ptr<float>(j);
			
			int kk=0;

			int left=kk;
			int right=kk;

			float sum=s[0];
			int sumN=1;

			int i;
			for(kk=0;kk<src.cols; kk++)
			{
				float L=0.f,R=0.f;
				float Lr=0.f,Rr=0.f;

				int TMPright=right;
				for(i=TMPright+1;i<src.cols; i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						sum += s[i];
						sumN++;
						right++;
					}
					else
					{
						break;
					}    
				}
				//right fractional area
				float a = (dx[kk]+radius -dx[i-1]) /(dx[i]-dx[i-1]);
				float yi = s[i-1] + a*(s[i]-s[i-1]);
				R = 0.5f*(yi+s[i-1])*a*(dx[i]-dx[i-1]);
				Rr = dx[i-1];
				//std::cout<<a<<","<<(dx[i]-dx[i-1])<<","<<yi<<","<<R<<std::endl;

				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						sum -= s[i];
						sumN--;
						left++;
					}
					else
					{
						break;
					}    
				}
				//left fractional area
				a = (dx[kk]-radius -dx[i-1]) /(dx[i]-dx[i-1]);
				yi = s[i-1] + a*(s[i]-s[i-1]);
				L= 0.5f*(yi+s[i])*(1.f-a)*(dx[i]-dx[i-1]);
				Lr=dx[i];
				
				d[kk] = ((sum/(float)sumN)*(Rr-Lr) + L+R)/(float)(2.f*radius);
			}
		}
	}
	else if(src.channels()==3)
	{
		for(int j=0;j<src.rows;j++)
		{
			const float* s = src.ptr<float>(j)+3;
			const float* dx = ctx.ptr<float>(j)+1;
			float* d = dest.ptr<float>(j);
			
			int kk=0;

			int left=kk;
			int right=kk;

			Vec3f bgr = Vec3f (s[0],s[1],s[2]);
			int sumN=1;

			int i;
			for(kk=0;kk<src_.cols;kk++)
			{
				Vec3f L = Vec3f(0.f,0.f,0.f);
				Vec3f R = Vec3f(0.f,0.f,0.f);
				float Lr=0.f,Rr=0.f;

				int TMPright=right;
				for(i=TMPright+1;i<src_.cols;i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						bgr.val[0] += s[3*i+0];
						bgr.val[1] += s[3*i+1];
						bgr.val[2] += s[3*i+2];
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				//right fractional area
				float a = (dx[kk]+radius -dx[i-1]) /(dx[i]-dx[i-1]);
				Vec3f yi = Vec3f(
						s[3*(i-1)+0] + a*(s[3*i+0]-s[3*(i-1)+0]),
						s[3*(i-1)+1] + a*(s[3*i+1]-s[3*(i-1)+1]),
						s[3*(i-1)+2] + a*(s[3*i+2]-s[3*(i-1)+2])
						);

				R = Vec3f(
						0.5f*(yi.val[0]+s[3*(i-1)+0])*a*(dx[i]-dx[i-1]),
						0.5f*(yi.val[1]+s[3*(i-1)+1])*a*(dx[i]-dx[i-1]),
						0.5f*(yi.val[2]+s[3*(i-1)+2])*a*(dx[i]-dx[i-1])
						);
				Rr = dx[i-1];


				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						bgr.val[0] -= s[3*i+0];
						bgr.val[1] -= s[3*i+1];
						bgr.val[2] -= s[3*i+2];
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				//left fractional area
				a = (dx[kk]-radius -dx[i-1]) /(dx[i]-dx[i-1]);
				yi = Vec3f(
						s[3*(i-1)+0] + a*(s[3*i+0]-s[3*(i-1)+0]),
						s[3*(i-1)+1] + a*(s[3*i+1]-s[3*(i-1)+1]),
						s[3*(i-1)+2] + a*(s[3*i+2]-s[3*(i-1)+2])
						);

				L = Vec3f(
						0.5f*(yi.val[0]+s[3*(i)+0])*(1.f-a)*(dx[i]-dx[i-1]),
						0.5f*(yi.val[1]+s[3*(i)+1])*(1.f-a)*(dx[i]-dx[i-1]),
						0.5f*(yi.val[2]+s[3*(i)+2])*(1.f-a)*(dx[i]-dx[i-1])
						);
				Lr = dx[i];


				Vec3f v= ((bgr/(float)sumN)*(Rr-Lr) + L+R)/(2.f*radius);
				d[3*kk+0] = v.val[0];
				d[3*kk+1] = v.val[1];
				d[3*kk+2] = v.val[2];
			}
		}
	}
}

void normalizedConvolutionX_32f(const Mat& src,Mat& dest,const float radius, const Mat& ctx)
{
	if(dest.empty()) dest.create(src.size(),src.type());

	if(src.channels()==1)
	{
		for(int j=0; j<src.rows; j++)
		{
			const float* s = src.ptr<float>(j);
			const float* dx = ctx.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			
			int kk=0;

			int left=kk;
			int right=kk;

			float sum=s[0];
			int sumN=1;

			int i;
			for(kk=0;kk<src.cols; kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.cols; i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						sum += s[i];
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
//					int Dsum=sum;
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						sum -= s[i];
						sumN--;
						left++;
					}else{
						break;
					}    
				}
				d[kk] = sum/(float)sumN;
			}
		}
	}
	else if(src.channels()==3)
	{
		for(int j=0;j<src.rows;j++)
		{
			const float* s = src.ptr<float>(j);
			const float* dx = ctx.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			
			int kk=0;

			int left=kk;
			int right=kk;

			float CV_DECL_ALIGNED(16) bgr[3];
			bgr[0]=s[0];
			bgr[1]=s[1];
			bgr[2]=s[2];
			int sumN=1;

			int i;
			for(kk=0;kk<src.cols;kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.cols;i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						bgr[0] += s[3*i+0];
						bgr[1] += s[3*i+1];
						bgr[2] += s[3*i+2];
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						bgr[0] -= s[3*i+0];
						bgr[1] -= s[3*i+1];
						bgr[2] -= s[3*i+2];
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				float div = 1.f/(float)sumN;
				d[3*kk+0] = bgr[0]*div;
				d[3*kk+1] = bgr[1]*div;
				d[3*kk+2] = bgr[2]*div;
			}
		}
	}
}

void normalizedConvolutionX_RGBA_32f_SSE(const Mat& src,Mat& dest,const float radius, const Mat& ctx)
{
	if(dest.empty()) dest.create(src.size(),src.type());

	bool sse = true;
	if(sse)
		for(int j=0;j<src.rows;j++)
		{
			const float* s = src.ptr<float>(j);
			const float* dx = ctx.ptr<float>(j);
			float* d = dest.ptr<float>(j);

			int kk=0;

			int left=kk;
			int right=kk;

			__m128 mbgr = _mm_loadu_ps(s);
			int sumN=1;

			int i;
			for(kk=0;kk<src.cols;kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.cols;i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						mbgr = _mm_add_ps(mbgr,_mm_loadu_ps(s+4*i));
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						mbgr = _mm_sub_ps(mbgr,_mm_loadu_ps(s+4*i));
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				__m128 mdiv = _mm_set1_ps((1.f/(float)sumN));
				_mm_storeu_ps(d+4*kk,_mm_mul_ps(mbgr,mdiv));
			}
		}
	else
	{
		for(int j=0;j<src.rows;j++)
		{
			const float* s = src.ptr<float>(j);
			const float* dx = ctx.ptr<float>(j);
			float* d = dest.ptr<float>(j);

			int kk=0;

			int left=kk;
			int right=kk;

			float CV_DECL_ALIGNED(16) bgr[4];
			bgr[0]=s[0];
			bgr[1]=s[1];
			bgr[2]=s[2];
			int sumN=1;

			int i;
			for(kk=0;kk<src.cols;kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.cols;i++)
				{
					float dis = dx[i]-dx[kk];
					if(abs(dis)<=radius)
					{
						bgr[0] += s[4*i+0];
						bgr[1] += s[4*i+1];
						bgr[2] += s[4*i+2];
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = dx[i]-dx[kk];
					if(fabs(dis)>radius)
					{
						bgr[0] -= s[4*i+0];
						bgr[1] -= s[4*i+1];
						bgr[2] -= s[4*i+2];
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				float div = 1.f/(float)sumN;
				d[4*kk+0] = bgr[0]*div;
				d[4*kk+1] = bgr[1]*div;
				d[4*kk+2] = bgr[2]*div;
			}
		}
	}
}


void normalizedConvolutionY_RGBA_32f_SSE(const Mat& src,Mat& dest,const float radius,Mat& cty)
{
	if(dest.empty()) dest.create(src.size(),src.type());

	bool sse = true;

	int vstepcty = cty.cols; 
	int vstep = src.cols*4;
	if(sse)
	{
		for(int j=0;j<src.cols;j++)
		{
			int kk=0;

			int left=kk;
			int right=kk;

			float* s = (float*)src.ptr<float>(0);s+=4*j;
			__m128 mbgr = _mm_loadu_ps(s);

			int sumN=1;

			int i;
			for(kk=0;kk<src.rows;kk++)
			{
				int TMPright=right;
				const float ctyk = cty.at<float>(kk, j);
				float* pcty = cty.ptr<float>(TMPright+1); pcty+=j;
				for(i=TMPright+1;i<src.rows;i++)
				{
					float dis = *(pcty) - ctyk;
					if(abs(dis)<=radius)
					{
						mbgr = _mm_add_ps(mbgr, _mm_loadu_ps(s+vstep*i));
						
						sumN++;
						right++;
					}else
					{
						break;
					}   
					pcty+=vstepcty;
				}
				int TMPleft=left;
				pcty = cty.ptr<float>(TMPleft); pcty+=j;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = *(pcty) - ctyk;
					if(fabs(dis)>radius)
					{
						mbgr = _mm_sub_ps(mbgr, _mm_loadu_ps(s+vstep*i));
						sumN--;
						left++;
					}else
					{
						break;
					}  
					pcty+=vstepcty;
				}
				float* d = dest.ptr<float>(kk);
				__m128 mdiv = _mm_set1_ps((1.f/(float)sumN));
				_mm_storeu_ps(d+4*j,_mm_mul_ps(mbgr,mdiv));
			}
		}
	}
	else
	{
		for(int j=0;j<src.cols;j++)
		{
			int kk=0;

			int left=kk;
			int right=kk;

			float b=src.at<float>(0,4*j+0);
			float g=src.at<float>(0,4*j+1);
			float r=src.at<float>(0,4*j+2);
			int sumN=1;

			int i;
			for(kk=0;kk<src.rows;kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.rows;i++)
				{
					float dis = cty.at<float>(i, j) - cty.at<float>(kk, j);
					if(abs(dis)<=radius)
					{
						b += src.at<float>(i,4*j+0);
						g += src.at<float>(i,4*j+1);
						r += src.at<float>(i,4*j+2);
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = cty.at<float>(i, j) - cty.at<float>(kk, j);
					if(fabs(dis)>radius)
					{
						b -= src.at<float>(i,4*j+0);
						g -= src.at<float>(i,4*j+1);
						r -= src.at<float>(i,4*j+2);
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				dest.at<float>(kk,4*j+0) = b/(float)sumN;
				dest.at<float>(kk,4*j+1) = g/(float)sumN;
				dest.at<float>(kk,4*j+2) = r/(float)sumN;
			}
		}
	}
}

void normalizedConvolutionY_32f(const Mat& src,Mat& dest,const float radius,Mat& cty)
{
	if(dest.empty()) dest.create(src.size(),src.type());

	if(src.channels()==1)
	{
		;
	}
	else if(src.channels()==3)
	{
		for(int j=0;j<src.cols;j++)
		{
			int kk=0;

			int left=kk;
			int right=kk;
			
			float b=src.at<float>(0,3*j+0);
			float g=src.at<float>(0,3*j+1);
			float r=src.at<float>(0,3*j+2);
			int sumN=1;

			int i;
			for(kk=0;kk<src.rows;kk++)
			{
				int TMPright=right;
				for(i=TMPright+1;i<src.rows;i++)
				{
					float dis = cty.at<float>(i, j) - cty.at<float>(kk, j);
					if(abs(dis)<=radius)
					{
						b += src.at<float>(i,3*j+0);
						g += src.at<float>(i,3*j+1);
						r += src.at<float>(i,3*j+2);
						sumN++;
						right++;
					}else
					{
						break;
					}    
				}
				int TMPleft=left;
				for(i=TMPleft;i<kk;i++)
				{
					//int Dsum=sum;
					float dis = cty.at<float>(i, j) - cty.at<float>(kk, j);
					if(fabs(dis)>radius)
					{
						b -= src.at<float>(i,3*j+0);
						g -= src.at<float>(i,3*j+1);
						r -= src.at<float>(i,3*j+2);
						sumN--;
						left++;
					}else
					{
						break;
					}    
				}
				dest.at<float>(kk,3*j+0) = b/(float)sumN;
				dest.at<float>(kk,3*j+1) = g/(float)sumN;
				dest.at<float>(kk,3*j+2) = r/(float)sumN;
			}
		}
	}
}


// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilter_NC_Base(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	Mat img;
	src.convertTo(img, CV_32F);

	int width = src.cols;
	int height = src.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat ch_v,ch_h;
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_ct_L1_8u(guide, dctx,dcty,ratio);
		//if(norm == DTF_L1) buid_ct_L_8u(guide, dctx,dcty,ratio,5);
		else if(norm == DTF_L2) buid_ct_L2_8u(guide, dctx,dcty,ratio);
		else buid_ct_L1_8u(guide, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		//ch_v=dcty;
		transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else buid_ct_L2_32f(guidef, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;

	Mat out;
	Mat imgt;
	Mat outt;
	while(i--)
	{	
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		const float radius = sigma_h*sqrt(3.f);

		normalizedConvolutionX_32f(img,out, radius, ch_h);
		transpose(out,imgt);
		normalizedConvolutionX_32f(imgt,outt, radius, ch_v);
		//RunNC_Y_32F(out,img, radius, ch_v);
		transpose(outt,img);
	}

	img.convertTo(dest,src.type(), 1.0, 0.5);
}

// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilter_NC_BGRA_SSE_SINGLE(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	printf("sse\n");
	Mat img;
	cvtColorBGR8u2BGRA32f(src,img);

	int width = src.cols;
	int height = src.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat ch_v,ch_h;
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_ct_L1_8u(guide, dctx,dcty,ratio);
		//if(norm == DTF_L1) buid_ct_L_8u(guide, dctx,dcty,ratio,5);
		else if(norm == DTF_L2) buid_ct_L2_8u(guide, dctx,dcty,ratio);
		else buid_ct_L1_8u(guide, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		ch_v=dcty;
		//transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else buid_ct_L2_32f(guidef, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		ch_v=dcty;
		//transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;

	Mat out;
	Mat imgt;
	Mat outt;
	while(i--)
	{	
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		const float radius = sigma_h*sqrt(3.f);

		normalizedConvolutionX_RGBA_32f_SSE(img,out, radius, ch_h);
		normalizedConvolutionY_RGBA_32f_SSE(out,img, radius, ch_v);
		
		/*
		normalizedConvolutionX_RGBA_32f_SSE(img,out, radius, ch_h);
		transpose(out,imgt);
		normalizedConvolutionX_RGBA_32f_SSE(imgt,outt, radius, ch_v);
		transpose(outt,img);
		*/
	}

	cvtColorBGRA32f2BGR8u(img,dest);
	//img.convertTo(dest,src.type(), 1.0, 0.5);
}

void domainTransformFilter_NC_Base(const Mat& src, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm)
{
	domainTransformFilter_NC_Base(src,src,dst,sigma_r,sigma_s,maxiter,norm);
}

void domainTransformFilterNC(const Mat& src, const Mat& guide, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{   
	if(implementation==DTF_SLOWEST)
	{
		domainTransformFilter_NC_Base(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
	else if(implementation==DTF_BGRA_SSE)
	{
		if(src.channels()==1)domainTransformFilter_NC_Base(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
		else domainTransformFilter_NC_BGRA_SSE_SINGLE(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
	else if(implementation==DTF_BGRA_SSE_PARALLEL)
	{
		std::cout<<"no inplimentation"<<std::endl;
		src.copyTo(dst);
	}
}

void domainTransformFilterNC(const Mat& src, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{
	domainTransformFilterNC(src,src,dst,sigma_r,sigma_s,maxiter,norm,implementation);
}

///////////////////////////////////////////////////////////////////////////////
//IC
////////////////////////////////////////////////////////////////////////////////
// Domain transform filtering: baseline implimentation for optimization
void domainTransformFilter_IC_Base(const Mat& src, const Mat& guide, Mat& dest, float sigma_r, float sigma_s, int maxiter, int norm)
{
	Mat img;
	src.convertTo(img, CV_32F);

	int width = src.cols;
	int height = src.rows;

	// compute derivatives of transformed domain "dct"
	cv::Mat dctx = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat dcty = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat ch_v,ch_h;
	float ratio = (sigma_s / sigma_r);

	if(guide.depth()==CV_8U)
	{
		if(norm == DTF_L1) buid_ct_L1_8u(guide, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_ct_L2_8u(guide, dctx,dcty,ratio);
		else buid_ct_L1_8u(guide, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		//ch_v=dcty;
		transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}
	else
	{
		Mat guidef;
		guide.convertTo(guidef, CV_32F);
		if(norm == DTF_L1) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else if(norm == DTF_L2) buid_ct_L2_32f(guidef, dctx,dcty,ratio);
		else buid_ct_L2_32f(guidef, dctx,dcty,ratio);

		//cunsum_32f(dctx);
		ch_h=dctx;
		transpose(dcty,ch_v);
		//cunsum_32f(ch_v);
	}

	// Apply recursive folter maxiter times
	int i=maxiter;

	Mat out;
	Mat imgt;
	Mat outt;
	while(i--)
	{	
		float sigma_h = (float) (sigma_s * sqrt(3.0) * pow(2.0,(maxiter - (i+1))) / sqrt(pow(4.0,maxiter) -1));
		const float radius = sigma_h*sqrt(3.f);

		interpolatedConvolutionX_32f(img,out, radius, ch_h);
		transpose(out,imgt);
		interpolatedConvolutionX_32f(imgt,outt, radius, ch_v);
		//RunNC_Y_32F(out,img, radius, ch_v);
		transpose(outt,img);
	}

	img.convertTo(dest,src.type(), 1.0, 0.5);
}

void domainTransformFilterIC(const Mat& src, const Mat& guide, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{   
	if(implementation==DTF_SLOWEST)
	{
		domainTransformFilter_IC_Base(src, guide, dst, sigma_r,sigma_s,maxiter, norm);
	}
	else if(implementation==DTF_BGRA_SSE)
	{
		std::cout<<"no inplimentation"<<std::endl;
		src.copyTo(dst);
	}
	else if(implementation==DTF_BGRA_SSE_PARALLEL)
	{
		std::cout<<"no inplimentation"<<std::endl;
		src.copyTo(dst);
	}
}

void domainTransformFilterIC(const Mat& src, Mat& dst, float sigma_r, float sigma_s, int maxiter, int norm, int implementation)
{
	domainTransformFilterIC(src,src,dst,sigma_r,sigma_s,maxiter,norm,implementation);
}



void domainTransformFilter(InputArray srcImage, InputArray guideImage, OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm, const int convolutionType, const int implementation)
{
	Mat src = srcImage.getMat();
	if(destImage.empty())destImage.create(src.size(),src.type());
	Mat guide = guideImage.getMat();
	Mat dst = destImage.getMat();

	if(convolutionType==DTF_RF)
		domainTransformFilterRF(src,guide, dst,sigma_r,sigma_s,maxiter,norm,implementation);
	if(convolutionType==DTF_NC)
		domainTransformFilterNC(src,guide, dst,sigma_r,sigma_s,maxiter,norm,implementation);
	if(convolutionType==DTF_IC)
		domainTransformFilterIC(src,guide, dst,sigma_r,sigma_s,maxiter,norm,implementation);

	
}

void domainTransformFilter(InputArray srcImage, OutputArray destImage, const float sigma_r, const float sigma_s, const int maxiter, const int norm, const int convolutionType, const int implementation)
{
	domainTransformFilter(srcImage,srcImage,destImage,sigma_r,sigma_s,maxiter,norm, convolutionType, implementation);
}