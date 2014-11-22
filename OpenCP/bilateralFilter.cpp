#include "opencp.hpp"
#include "filterCore.h"
#include <opencv2/core/internal.hpp>

void bilateralFilterSlowest(const Mat& src, Mat& dest, Size kernel, double sigma_color, double sigma_space)
{
	Mat srcd;src.convertTo(srcd,CV_64F);
	Mat destd = Mat::zeros(src.size(),CV_MAKETYPE(CV_64F,src.channels()));
	const int r = max(kernel.width,kernel.height)/2;
	const int hr = kernel.width/2;
	const int vr = kernel.height/2;
	int channels = src.channels();
	if(channels==1)
	{
		for(int j=0;j<src.rows;j++)
		{
			for(int i=0;i<src.cols;i++)
			{
				double sum = 0.0;
				double coeff = 0.0;
				for(int l=-vr;l<=vr;l++)
				{
					for(int k=-hr;k<=hr;k++)
					{
						if(sqrt(l*l+k*k)<=r && i+k>=0 && i+k<src.cols && j+l>=0 && j+l<src.rows )
						{
							double c = exp(-0.5*((srcd.at<double>(j+l,i+k)-srcd.at<double>(j,i))*(srcd.at<double>(j+l,i+k)-srcd.at<double>(j,i)))/(sigma_color*sigma_color));
							double s = exp(-0.5*(l*l+k*k)/(sigma_space*sigma_space));
							coeff+=c*s;
							sum+=srcd.at<double>(j+l,i+k)*c*s;
						}
					}
				}
				destd.at<double>(j,i)=sum/coeff;
			}
		}
	}
	else if(channels ==3)
	{
		for(int j=0;j<src.rows;j++)
		{
			for(int i=0;i<src.cols;i++)
			{
				double sumb = 0.0;
				double sumg = 0.0;
				double sumr = 0.0;
				double coeff = 0.0;
				for(int l=-vr;l<=vr;l++)
				{
					for(int k=-hr;k<=hr;k++)
					{
						if(sqrt(l*l+k*k)<=r && i+k>=0 && i+k<src.cols && j+l>=0 && j+l<src.rows )
						{
							double c = exp(-0.5*(
								(srcd.at<double>(j+l,3*(i+k)+0)-srcd.at<double>(j,3*i+0))*(srcd.at<double>(j+l,3*(i+k)+0)-srcd.at<double>(j,3*i+0))+
								(srcd.at<double>(j+l,3*(i+k)+1)-srcd.at<double>(j,3*i+1))*(srcd.at<double>(j+l,3*(i+k)+1)-srcd.at<double>(j,3*i+1))+
								(srcd.at<double>(j+l,3*(i+k)+2)-srcd.at<double>(j,3*i+2))*(srcd.at<double>(j+l,3*(i+k)+2)-srcd.at<double>(j,3*i+2))
								)/(sigma_color*sigma_color));
							double s = exp(-0.5*(l*l+k*k)/(sigma_space*sigma_space));
							coeff+=c*s;
							sumb+=srcd.at<double>(j+l,3*(i+k)+0)*c*s;
							sumg+=srcd.at<double>(j+l,3*(i+k)+1)*c*s;
							sumr+=srcd.at<double>(j+l,3*(i+k)+2)*c*s;
						}
					}
				}
				destd.at<double>(j,3*i+0)=sumb/coeff;
				destd.at<double>(j,3*i+1)=sumg/coeff;
				destd.at<double>(j,3*i+2)=sumr/coeff;
			}
		}
	}
	destd.convertTo(dest,src.type());
}


//for non sse function
void bilateralFilterL2_8u( const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space,int borderType)
{
	if(d==0){src.copyTo(dst);return;}
	Size size = src.size();
	if(dst.empty())dst=Mat::zeros(src.size(),src.type());
	//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
	//	src.type() == dst.type() && src.size() == dst.size() &&
	//	src.data != dst.data );

	if( sigma_color <= 0.0 )
		sigma_color = 1.0;
	if( sigma_space <= 0.0 )
		sigma_space = 1.0;
	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	const int cn = src.channels();

	int radius;
	if( d <= 0 )
		radius = cvRound(sigma_space*1.5);
	else
		radius = d/2;
	radius = max(radius, 1);
	d = radius*2 + 1;

	Mat sim;
	copyMakeBorder( src, sim, radius, radius, radius, radius, borderType );

	vector<float> _color_weight(cn*256*256);
	vector<float> _space_weight(d*d);
	vector<int> _space_ofs_src(d*d);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs_src = &_space_ofs_src[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < (256*256)*cn; i++ )
		color_weight[i] = (float)std::exp(i*gauss_color_coeff);

	int maxk=0;
	// initialize space-related bilateral filter coefficients
	for(int i = -radius; i <= radius; i++ )
	{
		for(int j = -radius; j <= radius; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > radius )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs_src[maxk++] = (int)(i*sim.step + j*cn);
		}
	}

#pragma omp parallel for
	for(int i = 0; i < size.height; i++ )
	{
		const uchar* sptr = sim.data + (i+radius)*sim.step + radius*cn;
		uchar* dptr = dst.data + i*dst.step;

		if( cn == 1)
		{
			for(int j = 0; j < size.width; j++ )
			{
				float sum = 0, wsum = 0;
				int val0 = sptr[j];
				for(int k = 0; k < maxk; k++ )
				{
					int val = sptr[j + space_ofs_src[k]];
					float w = space_weight[k]*color_weight[(val - val0)*(val - val0)];
					sum += val*w;
					wsum += w;
				}
				// overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[j] = (uchar)cvRound(sum/wsum);
			}
		}
		else if(cn == 3)
		{
			for(int j = 0; j < size.width*3; j += 3 )
			{
				float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
				for(int k = 0; k < maxk; k++ )
				{
					const uchar* sptr_k = sptr + j + space_ofs_src[k];
					int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

					float w = space_weight[k]*color_weight[(b - b0)*(b - b0) + abs(g - g0)*abs(g - g0) + (r - r0)*(r - r0)];
					sum_b += b*w; sum_g += g*w; sum_r += r*w;
					wsum += w;
				}
				wsum = 1.f/wsum;
				b0 = cvRound(sum_b*wsum);
				g0 = cvRound(sum_g*wsum);
				r0 = cvRound(sum_r*wsum);
				dptr[j] = (uchar)b0; dptr[j+1] = (uchar)g0; dptr[j+2] = (uchar)r0;
			}
		}
	}
}

//for non sse function
void bilateralFilterBase_8u( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType)
{
	if(d==0){src.copyTo(dst);return;}
	Size size = src.size();
	if(dst.empty())dst=Mat::zeros(src.size(),src.type());
	//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
	//	src.type() == dst.type() && src.size() == dst.size() &&
	//	src.data != dst.data );

	if( sigma_color <= 0.0 )
		sigma_color = 1.0;
	if( sigma_space <= 0.0 )
		sigma_space = 1.0;
	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	const int cn = src.channels();

	int radius;
	if( d <= 0 )
		radius = cvRound(sigma_space*1.5);
	else
		radius = d/2;
	radius = max(radius, 1);
	d = radius*2 + 1;

	Mat sim;
	copyMakeBorder( src, sim, radius, radius, radius, radius, borderType );

	vector<float> _color_weight(cn*256);
	vector<float> _space_weight(d*d);
	vector<int> _space_ofs_src(d*d);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs_src = &_space_ofs_src[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < 256*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	int maxk=0;
	// initialize space-related bilateral filter coefficients
	for(int i = -radius; i <= radius; i++ )
	{
		for(int j = -radius; j <= radius; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > radius )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs_src[maxk++] = (int)(i*sim.step + j*cn);
		}
	}

	//#pragma omp parallel for
	for(int i = 0; i < size.height; i++ )
	{
		const uchar* sptr = sim.data + (i+radius)*sim.step + radius*cn;
		uchar* dptr = dst.data + i*dst.step;

		if( cn == 1)
		{
			for(int j = 0; j < size.width; j++ )
			{
				float sum = 0, wsum = 0;
				int val0 = sptr[j];
				for(int k = 0; k < maxk; k++ )
				{
					int val = sptr[j + space_ofs_src[k]];
					float w = space_weight[k]*color_weight[std::abs(val - val0)];
					sum += val*w;
					wsum += w;
				}
				// overflow is not possible here => there is no need to use CV_CAST_8U
				dptr[j] = (uchar)cvRound(sum/wsum);
			}
		}
		else if(cn == 3)
		{
			for(int j = 0; j < size.width*3; j += 3 )
			{
				float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
				for(int k = 0; k < maxk; k++ )
				{
					const uchar* sptr_k = sptr + j + space_ofs_src[k];
					int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
					float w = space_weight[k]*color_weight[std::abs(b - b0) +
						std::abs(g - g0) + std::abs(r - r0)];
					sum_b += b*w; sum_g += g*w; sum_r += r*w;
					wsum += w;
				}
				wsum = 1.f/wsum;
				b0 = cvRound(sum_b*wsum);
				g0 = cvRound(sum_g*wsum);
				r0 = cvRound(sum_r*wsum);
				dptr[j] = (uchar)b0; dptr[j+1] = (uchar)g0; dptr[j+2] = (uchar)r0;
			}
		}
	}
}

void bilateralFilterBase_32f( const Mat& src, Mat& dst, int d,double sigma_color, double sigma_space,int borderType)
{
	if(d==0){src.copyTo(dst);return;}
	Size size = src.size();
	if(dst.empty())dst=Mat::zeros(src.size(),src.type());

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0.0 )
		sigma_color = 1.0;
	if( sigma_space <= 0.0 )
		sigma_space = 1.0;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	const int cn = src.channels();

	int radius= d/2;
	d = radius*2 + 1;

	Mat sim;

	copyMakeBorder( src, sim, radius, radius, radius, radius, borderType );

	double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);

	vector<float> _color_weight(cn*color_range);
	vector<float> _space_weight(d*d);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];

	vector<int> _space_ofs_src(d*d);
	int* space_ofs_src = &_space_ofs_src[0];

	// initialize color-related bilateral filter coefficients


	for(int i = 0; i < color_range*cn; i++ )
	{
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
	}

	int maxk=0;
	// initialize space-related bilateral filter coefficients
	for(int i = -radius; i <= radius; i++ )
	{
		for(int j = -radius; j <= radius; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > radius )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
		}
	}

	for(int i = 0; i < size.height; i++ )
	{
		const float* sptr = sim.ptr<float>(i+radius)+ radius*cn;
		float* dptr = dst.ptr<float>(i);

		if( cn == 1 )
		{
			for(int j = 0; j < size.width; j++ )
			{
				float sum = 0.f, wsum = 0.f;
				float val0 = sptr[j];
				for(int k = 0; k < maxk; k++ )
				{
					float vals = sptr[j + space_ofs_src[k]];
					float w = space_weight[k]*color_weight[cvRound(std::abs(vals - val0))];

					sum += vals*w;
					wsum += w;
				}
				dptr[j] = sum/wsum;
			}
		}
		else if(cn == 3)
		{
			for(int j = 0; j < size.width*3; j += 3 )
			{
				float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f, wsum = 0.f;
				const float b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];

				for(int k = 0; k < maxk; k++ )
				{
					const float* sptr_k = sptr + j + space_ofs_src[k];
					const float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];

					float w = space_weight[k]
					*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
					sum_b += b*w; 
					sum_g += g*w;
					sum_r += r*w;
					wsum += w;
				}
				dptr[j  ] = sum_b/wsum;
				dptr[j+1] = sum_g/wsum;
				dptr[j+2] = sum_r/wsum;
			}
		}
	}
}


void bilateralFilterBase( const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space,int borderType)
{
	if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
	{
		bilateralFilterBase_8u(src,dst,d,sigma_color,sigma_space,borderType);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
	{
		bilateralFilterBase_32f(src,dst,d,sigma_color,sigma_space,borderType);
	}
}

void bilateralWeightMapBase_32f( const Mat& src, Mat& dst, int d,
	double sigma_color,  double sigma_space,int borderType)
{
	if(d==0){src.copyTo(dst);return;}
	Size size = src.size();
	if(dst.empty())dst=Mat::zeros(src.size(),CV_32F);
	//CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
	//	src.type() == dst.type() && src.size() == dst.size() &&
	//	src.data != dst.data );

	if( sigma_color <= 0.0 )
		sigma_color = 1.0;
	if( sigma_space <= 0.0 )
		sigma_space = 1.0;
	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	const int cn = src.channels();

	int radius;
	if( d <= 0 )
		radius = cvRound(sigma_space*1.5);
	else
		radius = d/2;
	radius = MAX(radius, 1);
	d = radius*2 + 1;

	Mat sim;
	copyMakeBorder( src, sim, radius, radius, radius, radius, borderType );

	vector<float> _color_weight(cn*256);
	vector<float> _space_weight(d*d);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];

	vector<int> _space_ofs_src(d*d);
	int* space_ofs_src = &_space_ofs_src[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < 256*cn; i++ )//trilateral(5)
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	int maxk=0;
	// initialize space-related bilateral filter coefficients
	for(int i = -radius; i <= radius; i++ )
	{
		for(int j = -radius; j <= radius; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > radius )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs_src[maxk++] = (int)(i*sim.cols*cn + j*cn);
		}
	}

	for(int i = 0; i < size.height; i++ )
	{
		const float* sptr = sim.ptr<float>(i+radius)+ radius*cn;
		float* dptr = dst.ptr<float>(i);

		if( cn == 1)
		{
			for(int j = 0; j < size.width; j++ )
			{
				float wsum = 0.f;
				const float vals0 = sptr[j];

				for(int k = 0; k < maxk; k++ )
				{
					float vals = sptr[j + space_ofs_src[k]];

					float w = space_weight[k]
					*color_weight[cvRound(std::abs(vals - vals0))];
					wsum += w;
				}
				dptr[j] = wsum;
			}
		}
		else if(cn == 3)
		{
			for(int j = 0,l=0; l < size.width; j += 3,l++ )
			{
				float wsum = 0.f;
				float bs0 = sptr[j], gs0 = sptr[j+1], rs0 = sptr[j+2];

				for(int k = 0; k < maxk; k++ )
				{
					const float* sptr_k = sptr + j + space_ofs_src[k];
					const float bs = sptr_k[0], gs = sptr_k[1], rs = sptr_k[2];

					float w = space_weight[k]
					*color_weight[cvRound(std::abs(bs - bs0) +std::abs(gs - gs0) + std::abs(rs - rs0))];
					wsum += w;
				}
				dptr[l] =wsum;
			}
		}
	}
}

void bilateralWeightMapBase( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType)
{
	if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
	{
		bilateralWeightMapBase_32f(src,dst,d,sigma_color,sigma_space,borderType);
	}
	else
	{
		Mat ss;
		src.convertTo(ss,CV_32F);
		bilateralWeightMapBase_32f(ss,dst,d,sigma_color,sigma_space,borderType);
	}
}

class BilateralBoxFilterORDER2_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralBoxFilterORDER2_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float _color_f) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs) , color_f(_color_f)
	{
	}
	virtual void operator() (const Range& range) const
	{
		int i, j, cn = dest->channels(), k;
		Size size = dest->size();
		const float color2_f = 1.f/(color_f*color_f);
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);
			const int sstep = temp->cols;
			const int dstep = dest->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];
						const uchar* sptrj = sptr+j;
						const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 tval1 = _mm_set1_ps(0.0f);
						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 tval2 = _mm_set1_ps(0.0f);
						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 tval3 = _mm_set1_ps(0.0f);
						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 tval4 = _mm_set1_ps(0.0f);
						const __m128 one = _mm_set1_ps(1.f);
						const __m128 cef = _mm_set1_ps(color2_f);
						const __m128 mif = _mm_set1_ps(0.00001f);
						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							const __m128i sub = _mm_add_epi8(_mm_subs_epu8(sval,sref),_mm_subs_epu8(sref,sval));

							__m128i m1 = _mm_unpacklo_epi8(sref,zero);
							__m128i m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							__m128 fv = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,0)));
							__m128 _w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv))));

							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1,_valF);
							wval1 = _mm_add_ps(wval1,_w);

							fv = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,4)));
							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv))));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2,_valF);
							wval2 = _mm_add_ps(wval2,_w);

							m1 = _mm_unpackhi_epi8(sref,zero);
							m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,8)));
							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv))));
							_valF =_mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3,_w);
							tval3 = _mm_add_ps(tval3,_valF);

							fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,12)));
							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv))));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4,_w);
							tval4 = _mm_add_ps(tval4,_valF);
						}
						tval1 = _mm_div_ps(tval1,wval1);
						tval2 = _mm_div_ps(tval2,wval2);
						tval3 = _mm_div_ps(tval3,wval3);
						tval4 = _mm_div_ps(tval4,wval4);
						_mm_stream_si128((__m128i*)(dptr+j), _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)) , _mm_packs_epi32( _mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						int v = (val - val0);
						float w = max(1.f -( color2_f*v*v),0.000001f);
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			assert( cn == 3 );//color

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);

			uchar* dptr = dest->ptr(range.start);			
			//cout<<"mod: "<<size.width%16 <<","<<size.width<<endl;
			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)
					{
						int* ofs = &space_ofs[0];
						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;
						const __m128 one = _mm_set1_ps(1.f);
						const __m128 cef = _mm_set1_ps(color2_f);
						const __m128 mif = _mm_set1_ps(0.00001f);
						const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 rval2 = _mm_set1_ps(0.0f);
						__m128 gval2 = _mm_set1_ps(0.0f);
						__m128 bval2 = _mm_set1_ps(0.0f);

						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 rval3 = _mm_set1_ps(0.0f);
						__m128 gval3 = _mm_set1_ps(0.0f);
						__m128 bval3 = _mm_set1_ps(0.0f);

						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 rval4 = _mm_set1_ps(0.0f);
						__m128 gval4 = _mm_set1_ps(0.0f);
						__m128 bval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k <= maxk; k ++, ofs++)
						{
							__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));

							__m128i r1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
							__m128i r2 = _mm_unpackhi_epi8(r1,zero);
							r1 = _mm_unpacklo_epi8(r1,zero);

							__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
							__m128i g2 = _mm_unpackhi_epi8(g1,zero);
							g1 = _mm_unpacklo_epi8(g1,zero);

							r1 = _mm_add_epi16(r1,g1);
							r2 = _mm_add_epi16(r2,g2);

							__m128i b1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
							__m128i b2 = _mm_unpackhi_epi8(b1,zero);
							b1 = _mm_unpacklo_epi8(b1,zero);

							r1 = _mm_add_epi16(r1,b1);
							r2 = _mm_add_epi16(r2,b2);

							g1 = _mm_unpacklo_epi8(r1,zero);
							g2 = _mm_unpackhi_epi8(r1,zero);
							__m128 fv1 = _mm_cvtepi32_ps(g1);
							__m128 fv2 = _mm_cvtepi32_ps(g2);
							g1 = _mm_unpacklo_epi8(r2,zero);
							g2 = _mm_unpackhi_epi8(r2,zero);
							__m128 fv3 = _mm_cvtepi32_ps(g1);
							__m128 fv4 = _mm_cvtepi32_ps(g2);


							r1 = _mm_unpacklo_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);
							g1 = _mm_unpacklo_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);
							b1 = _mm_unpacklo_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							__m128 _w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv1,fv1))));

							__m128 _valr = _mm_cvtepi32_ps(r1);
							__m128 _valg = _mm_cvtepi32_ps(g1);
							__m128 _valb = _mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval1 = _mm_add_ps(rval1,_valr);
							gval1 = _mm_add_ps(gval1,_valg);
							bval1 = _mm_add_ps(bval1,_valb);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv2,fv2))));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval2 = _mm_add_ps(rval2,_valr);
							gval2 = _mm_add_ps(gval2,_valg);
							bval2 = _mm_add_ps(bval2,_valb);
							wval2 = _mm_add_ps(wval2,_w);

							r1 = _mm_unpackhi_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);

							g1 = _mm_unpackhi_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);

							b1 = _mm_unpackhi_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);


							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv3,fv3))));

							_valr =_mm_cvtepi32_ps(r1);
							_valg =_mm_cvtepi32_ps(g1);
							_valb =_mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval3 = _mm_add_ps(wval3,_w);
							rval3 = _mm_add_ps(rval3,_valr);
							gval3 = _mm_add_ps(gval3,_valg);
							bval3 = _mm_add_ps(bval3,_valb);

							_w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv4,fv4))));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval4 = _mm_add_ps(wval4,_w);
							rval4 = _mm_add_ps(rval4,_valr);
							gval4 = _mm_add_ps(gval4,_valg);
							bval4 = _mm_add_ps(bval4,_valb);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						rval2 = _mm_div_ps(rval2,wval2);
						rval3 = _mm_div_ps(rval3,wval3);
						rval4 = _mm_div_ps(rval4,wval4);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_div_ps(gval1,wval1);
						gval2 = _mm_div_ps(gval2,wval2);
						gval3 = _mm_div_ps(gval3,wval3);
						gval4 = _mm_div_ps(gval4,wval4);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_div_ps(bval1,wval1);
						bval2 = _mm_div_ps(bval2,wval2);
						bval3 = _mm_div_ps(bval3,wval3);
						bval4 = _mm_div_ps(bval4,wval4);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

						//sse4///


						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8
							(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);

						const __m128i bmask2 = _mm_setr_epi8
							(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

						a = _mm_shuffle_epi8(a,mask1);
						b = _mm_shuffle_epi8(b,mask2);
						c = _mm_shuffle_epi8(c,mask3);
						uchar* dptrc = dptr+3*j;
						_mm_stream_si128((__m128i*)(dptrc),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
						_mm_stream_si128((__m128i*)(dptrc+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
						_mm_stream_si128((__m128i*)(dptrc+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;

					int r0 = sptrrj[0];
					int g0 = sptrgj[0];
					int b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];						
						float w = max(1.f - color2_f*((r-r0)*(r-r0)+(g-g0)*(g-g0)+(b-b0)*(b-b0)),0.000001f);

						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U

					wsum = 1.f/wsum;
					dptr[3*j] = (uchar)cvRound(sum_b*wsum); dptr[3*j+1] = (uchar)cvRound(sum_g*wsum); dptr[3*j+2] = (uchar)cvRound(sum_r*wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float color_f;
};

class BilateralFilterORDER2_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilterORDER2_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float _color_f) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_f(_color_f)
	{
	}
	virtual void operator() (const Range& range) const
	{
		int i, j, cn = dest->channels(), k;
		Size size = dest->size();
		const float color2_f = 1.f/(color_f*color_f);
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);
			const int sstep = temp->cols;
			const int dstep = dest->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const uchar* sptrj = sptr+j;
						const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 tval1 = _mm_set1_ps(0.0f);
						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 tval2 = _mm_set1_ps(0.0f);
						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 tval3 = _mm_set1_ps(0.0f);
						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 tval4 = _mm_set1_ps(0.0f);
						const __m128 one = _mm_set1_ps(1.f);
						const __m128 cef = _mm_set1_ps(color2_f);
						const __m128 mif = _mm_set1_ps(0.00001f);
						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							const __m128i sub = _mm_add_epi8(_mm_subs_epu8(sval,sref),_mm_subs_epu8(sref,sval));

							__m128i m1 = _mm_unpacklo_epi8(sref,zero);
							__m128i m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,0)));
							__m128 _w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv)))));

							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1,_valF);
							wval1 = _mm_add_ps(wval1,_w);

							fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,4)));
							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv)))));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2,_valF);
							wval2 = _mm_add_ps(wval2,_w);

							m1 = _mm_unpackhi_epi8(sref,zero);
							m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,8)));
							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv)))));
							_valF =_mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3,_w);
							tval3 = _mm_add_ps(tval3,_valF);

							fv =  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(sub,12)));
							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv)))));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4,_w);
							tval4 = _mm_add_ps(tval4,_valF);
						}
						tval1 = _mm_div_ps(tval1,wval1);
						tval2 = _mm_div_ps(tval2,wval2);
						tval3 = _mm_div_ps(tval3,wval3);
						tval4 = _mm_div_ps(tval4,wval4);
						_mm_stream_si128((__m128i*)(dptr+j), _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)) , _mm_packs_epi32( _mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						int v = (val - val0);
						float w = space_weight[k]*max(1.f -( color2_f*v*v),0.000001f);
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			assert( cn == 3 );//color

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);

			uchar* dptr = dest->ptr(range.start);			
			//cout<<"mod: "<<size.width%16 <<","<<size.width<<endl;
			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;
						const __m128 one = _mm_set1_ps(1.f);
						const __m128 cef = _mm_set1_ps(color2_f);
						const __m128 mif = _mm_set1_ps(0.00001f);
						const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 rval2 = _mm_set1_ps(0.0f);
						__m128 gval2 = _mm_set1_ps(0.0f);
						__m128 bval2 = _mm_set1_ps(0.0f);

						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 rval3 = _mm_set1_ps(0.0f);
						__m128 gval3 = _mm_set1_ps(0.0f);
						__m128 bval3 = _mm_set1_ps(0.0f);

						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 rval4 = _mm_set1_ps(0.0f);
						__m128 gval4 = _mm_set1_ps(0.0f);
						__m128 bval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k <= maxk; k ++, ofs++,spw++)
						{
							__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));

							__m128i r1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
							__m128i r2 = _mm_unpackhi_epi8(r1,zero);
							r1 = _mm_unpacklo_epi8(r1,zero);

							__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
							__m128i g2 = _mm_unpackhi_epi8(g1,zero);
							g1 = _mm_unpacklo_epi8(g1,zero);

							r1 = _mm_add_epi16(r1,g1);
							r2 = _mm_add_epi16(r2,g2);

							__m128i b1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
							__m128i b2 = _mm_unpackhi_epi8(b1,zero);
							b1 = _mm_unpacklo_epi8(b1,zero);

							r1 = _mm_add_epi16(r1,b1);
							r2 = _mm_add_epi16(r2,b2);

							g1 = _mm_unpacklo_epi8(r1,zero);
							g2 = _mm_unpackhi_epi8(r1,zero);
							__m128 fv1 = _mm_cvtepi32_ps(g1);
							__m128 fv2 = _mm_cvtepi32_ps(g2);
							g1 = _mm_unpacklo_epi8(r2,zero);
							g2 = _mm_unpackhi_epi8(r2,zero);
							__m128 fv3 = _mm_cvtepi32_ps(g1);
							__m128 fv4 = _mm_cvtepi32_ps(g2);


							r1 = _mm_unpacklo_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);
							g1 = _mm_unpacklo_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);
							b1 = _mm_unpacklo_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);
							__m128 _w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv1,fv1)))));

							__m128 _valr = _mm_cvtepi32_ps(r1);
							__m128 _valg = _mm_cvtepi32_ps(g1);
							__m128 _valb = _mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval1 = _mm_add_ps(rval1,_valr);
							gval1 = _mm_add_ps(gval1,_valg);
							bval1 = _mm_add_ps(bval1,_valb);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv2,fv2)))));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval2 = _mm_add_ps(rval2,_valr);
							gval2 = _mm_add_ps(gval2,_valg);
							bval2 = _mm_add_ps(bval2,_valb);
							wval2 = _mm_add_ps(wval2,_w);

							r1 = _mm_unpackhi_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);

							g1 = _mm_unpackhi_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);

							b1 = _mm_unpackhi_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);


							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv3,fv3)))));

							_valr =_mm_cvtepi32_ps(r1);
							_valg =_mm_cvtepi32_ps(g1);
							_valb =_mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval3 = _mm_add_ps(wval3,_w);
							rval3 = _mm_add_ps(rval3,_valr);
							gval3 = _mm_add_ps(gval3,_valg);
							bval3 = _mm_add_ps(bval3,_valb);

							_w = _mm_mul_ps(_sw,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv4,fv4)))));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval4 = _mm_add_ps(wval4,_w);
							rval4 = _mm_add_ps(rval4,_valr);
							gval4 = _mm_add_ps(gval4,_valg);
							bval4 = _mm_add_ps(bval4,_valb);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						rval2 = _mm_div_ps(rval2,wval2);
						rval3 = _mm_div_ps(rval3,wval3);
						rval4 = _mm_div_ps(rval4,wval4);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_div_ps(gval1,wval1);
						gval2 = _mm_div_ps(gval2,wval2);
						gval3 = _mm_div_ps(gval3,wval3);
						gval4 = _mm_div_ps(gval4,wval4);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_div_ps(bval1,wval1);
						bval2 = _mm_div_ps(bval2,wval2);
						bval3 = _mm_div_ps(bval3,wval3);
						bval4 = _mm_div_ps(bval4,wval4);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

						//sse4///


						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8
							(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);

						const __m128i bmask2 = _mm_setr_epi8
							(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

						a = _mm_shuffle_epi8(a,mask1);
						b = _mm_shuffle_epi8(b,mask2);
						c = _mm_shuffle_epi8(c,mask3);
						uchar* dptrc = dptr+3*j;
						_mm_stream_si128((__m128i*)(dptrc),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
						_mm_stream_si128((__m128i*)(dptrc+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
						_mm_stream_si128((__m128i*)(dptrc+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;

					int r0 = sptrrj[0];
					int g0 = sptrgj[0];
					int b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];						
						float w = space_weight[k]*max(1.f - color2_f*((r-r0)*(r-r0)+(g-g0)*(g-g0)+(b-b0)*(b-b0)),0.000001f);

						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U

					wsum = 1.f/wsum;
					dptr[3*j] = (uchar)cvRound(sum_b*wsum); dptr[3*j+1] = (uchar)cvRound(sum_g*wsum); dptr[3*j+2] = (uchar)cvRound(sum_r*wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight;
	float color_f;
};


class BilateralWeightMap_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralWeightMap_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH,int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, cn = (temp->rows-2*radiusV)/dest->rows, k;
		Size size = dest->size();		
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			int CV_DECL_ALIGNED(16) buf[4];
			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);
			const int sstep = temp->cols;
			const int dstep = dest->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const float* sptrj = sptr+j;
						const __m128 sval = _mm_load_ps((sptrj));

						__m128 wval1 = _mm_setzero_ps();


						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128 sref = _mm_loadu_ps((sptrj+*ofs));

							_mm_store_si128((__m128i*)buf,_mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval,sref), *(const __m128*)v32f_absmask)));
							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							wval1 = _mm_add_ps(wval1,_w);

						}
						_mm_stream_ps(dptr+j,wval1);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(val - val0))];
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = wsum;
				}
			}
		}
		else
		{
			assert( cn == 3 );
			int CV_DECL_ALIGNED(16) buf[16];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols;
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);			
			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						__m128 _w;
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;
						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_setzero_ps();

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							const __m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							const __m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							const __m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							_mm_store_si128((__m128i*)buf,
								_mm_cvtps_epi32(
								_mm_add_ps(
								_mm_add_ps(
								_mm_and_ps(_mm_sub_ps(rval,rref), *(const __m128*)v32f_absmask),
								_mm_and_ps(_mm_sub_ps(gval,gref), *(const __m128*)v32f_absmask)),
								_mm_and_ps(_mm_sub_ps(bval,bref), *(const __m128*)v32f_absmask)
								)
								));

							const __m128 _sw = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							wval1 = _mm_add_ps(wval1,_w);
						}
						_mm_stream_ps(dptr+j,wval1);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;
					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};

class BilateralWeightMap_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralWeightMap_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH,int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = (temp->rows-2*radiusV)/dest->rows;
		Size size = dest->size();		
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const uchar* sptrj = sptr+j;

						const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_setzero_ps();
						__m128 wval2 = _mm_setzero_ps();
						__m128 wval3 = _mm_setzero_ps();
						__m128 wval4 = _mm_setzero_ps();

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval,sref),_mm_subs_epu8(sref,sval)));

							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							wval2 = _mm_add_ps(wval2,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							wval3 = _mm_add_ps(wval3,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							wval4 = _mm_add_ps(wval4,_w);
						}
						_mm_stream_ps(dptr+j,wval1);
						_mm_stream_ps(dptr+j+4,wval2);
						_mm_stream_ps(dptr+j+8,wval3);
						_mm_stream_ps(dptr+j+12,wval4);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[j];
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(val - val0)];
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = wsum;
				}
			}
		}
		else
		{
			assert( cn == 3 );
			short CV_DECL_ALIGNED(16) buf[16];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols;
			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);

			float* dptr = dest->ptr<float>(range.start);			

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						__m128i m1,m2,n1,n2;
						__m128 _w;
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;
						const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_setzero_ps();
						__m128 wval2 = _mm_setzero_ps();
						__m128 wval3 = _mm_setzero_ps();
						__m128 wval4 = _mm_setzero_ps();

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							const __m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));

							m1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
							m2 = _mm_unpackhi_epi8(m1,zero);
							m1 = _mm_unpacklo_epi8(m1,zero);

							n1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
							n2 = _mm_unpackhi_epi8(n1,zero);
							n1 = _mm_unpacklo_epi8(n1,zero);

							m1 = _mm_add_epi16(m1,n1);
							m2 = _mm_add_epi16(m2,n2);

							n1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
							n2 = _mm_unpackhi_epi8(n1,zero);
							n1 = _mm_unpacklo_epi8(n1,zero);

							m1 = _mm_add_epi16(m1,n1);
							m2 = _mm_add_epi16(m2,n2);

							_mm_store_si128((__m128i*)(buf+8),m2);
							_mm_store_si128((__m128i*)buf,m1);

							const __m128 _sw = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							wval2 = _mm_add_ps(wval2,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							wval3 = _mm_add_ps(wval3,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							wval4 = _mm_add_ps(wval4,_w);
						}
						_mm_stream_ps(dptr+j,wval1);
						_mm_stream_ps(dptr+j+4,wval2);
						_mm_stream_ps(dptr+j+8,wval3);
						_mm_stream_ps(dptr+j+12,wval4);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;
					const int r0 = sptrrj[0];
					const int g0 = sptrgj[0];
					const int b0 = sptrbj[0];

					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};


class BilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();

#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];

						float* spw = space_weight;

						const uchar* sptrj = sptr+j;

						const __m128i sval0 = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 tval1 = _mm_set1_ps(0.0f);
						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 tval2 = _mm_set1_ps(0.0f);
						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 tval3 = _mm_set1_ps(0.0f);
						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 tval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval0,sref),_mm_subs_epu8(sref,sval0)));

							__m128i m1 = _mm_unpacklo_epi8(sref,zero);
							__m128i m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1,_valF);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2,_valF);
							wval2 = _mm_add_ps(wval2,_w);

							m1 = _mm_unpackhi_epi8(sref,zero);
							m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);


							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							_valF =_mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3,_w);
							tval3 = _mm_add_ps(tval3,_valF);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4,_w);
							tval4 = _mm_add_ps(tval4,_valF);
						}
						tval1 = _mm_div_ps(tval1,wval1);
						tval2 = _mm_div_ps(tval2,wval2);
						tval3 = _mm_div_ps(tval3,wval3);
						tval4 = _mm_div_ps(tval4,wval4);
						_mm_stream_si128((__m128i*)(dptr+j), _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)) , _mm_packs_epi32( _mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(val - val0)];
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			short CV_DECL_ALIGNED(16) buf[16];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;

			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);

			uchar* dptr = dest->ptr(range.start);			

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];

						float* spw = space_weight;

						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;

						const __m128i bval0 = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval0 = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval0 = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 rval2 = _mm_set1_ps(0.0f);
						__m128 gval2 = _mm_set1_ps(0.0f);
						__m128 bval2 = _mm_set1_ps(0.0f);

						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 rval3 = _mm_set1_ps(0.0f);
						__m128 gval3 = _mm_set1_ps(0.0f);
						__m128 bval3 = _mm_set1_ps(0.0f);

						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 rval4 = _mm_set1_ps(0.0f);
						__m128 gval4 = _mm_set1_ps(0.0f);
						__m128 bval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();

						for(k = 0;  k < maxk; k ++, ofs++, spw++)
						{
							__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));

							__m128i r1 = _mm_add_epi8(_mm_subs_epu8(rval0,rref),_mm_subs_epu8(rref,rval0));
							__m128i r2 = _mm_unpackhi_epi8(r1,zero);
							r1 = _mm_unpacklo_epi8(r1,zero);

							__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gval0,gref),_mm_subs_epu8(gref,gval0));
							__m128i g2 = _mm_unpackhi_epi8(g1,zero);
							g1 = _mm_unpacklo_epi8(g1,zero);

							r1 = _mm_add_epi16(r1,g1);
							r2 = _mm_add_epi16(r2,g2);

							__m128i b1 = _mm_add_epi8(_mm_subs_epu8(bval0,bref),_mm_subs_epu8(bref,bval0));
							__m128i b2 = _mm_unpackhi_epi8(b1,zero);
							b1 = _mm_unpacklo_epi8(b1,zero);

							r1 = _mm_add_epi16(r1,b1);
							r2 = _mm_add_epi16(r2,b2);

							_mm_store_si128((__m128i*)(buf+8),r2);
							_mm_store_si128((__m128i*)buf,r1);

							r1 = _mm_unpacklo_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);
							g1 = _mm_unpacklo_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);
							b1 = _mm_unpacklo_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);
							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							__m128 _valr = _mm_cvtepi32_ps(r1);
							__m128 _valg = _mm_cvtepi32_ps(g1);
							__m128 _valb = _mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval1 = _mm_add_ps(rval1,_valr);
							gval1 = _mm_add_ps(gval1,_valg);
							bval1 = _mm_add_ps(bval1,_valb);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval2 = _mm_add_ps(rval2,_valr);
							gval2 = _mm_add_ps(gval2,_valg);
							bval2 = _mm_add_ps(bval2,_valb);
							wval2 = _mm_add_ps(wval2,_w);

							r1 = _mm_unpackhi_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);

							g1 = _mm_unpackhi_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);

							b1 = _mm_unpackhi_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));

							_valr =_mm_cvtepi32_ps(r1);
							_valg =_mm_cvtepi32_ps(g1);
							_valb =_mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval3 = _mm_add_ps(wval3,_w);
							rval3 = _mm_add_ps(rval3,_valr);
							gval3 = _mm_add_ps(gval3,_valg);
							bval3 = _mm_add_ps(bval3,_valb);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval4 = _mm_add_ps(wval4,_w);
							rval4 = _mm_add_ps(rval4,_valr);
							gval4 = _mm_add_ps(gval4,_valg);
							bval4 = _mm_add_ps(bval4,_valb);
						}


						/*
						wval1 = _mm_rcp_ps(wval1);
						wval2 = _mm_rcp_ps(wval2);
						wval3 = _mm_rcp_ps(wval3);
						wval4 = _mm_rcp_ps(wval4);

						rval1 = _mm_mul_ps(rval1,wval1);
						rval2 = _mm_mul_ps(rval2,wval2);
						rval3 = _mm_mul_ps(rval3,wval3);
						rval4 = _mm_mul_ps(rval4,wval4);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_mul_ps(gval1,wval1);
						gval2 = _mm_mul_ps(gval2,wval2);
						gval3 = _mm_mul_ps(gval3,wval3);
						gval4 = _mm_mul_ps(gval4,wval4);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_mul_ps(bval1,wval1);
						bval2 = _mm_mul_ps(bval2,wval2);
						bval3 = _mm_mul_ps(bval3,wval3);
						bval4 = _mm_mul_ps(bval4,wval4);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));
						*/

						rval1 = _mm_div_ps(rval1,wval1);
						rval2 = _mm_div_ps(rval2,wval2);
						rval3 = _mm_div_ps(rval3,wval3);
						rval4 = _mm_div_ps(rval4,wval4);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_div_ps(gval1,wval1);
						gval2 = _mm_div_ps(gval2,wval2);
						gval3 = _mm_div_ps(gval3,wval3);
						gval4 = _mm_div_ps(gval4,wval4);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_div_ps(bval1,wval1);
						bval2 = _mm_div_ps(bval2,wval2);
						bval3 = _mm_div_ps(bval3,wval3);
						bval4 = _mm_div_ps(bval4,wval4);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

						//sse4///
						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8
							(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);

						const __m128i bmask2 = _mm_setr_epi8
							(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

						a = _mm_shuffle_epi8(a,mask1);
						b = _mm_shuffle_epi8(b,mask2);
						c = _mm_shuffle_epi8(c,mask3);
						uchar* dptrc = dptr+3*j;
						_mm_stream_si128((__m128i*)(dptrc),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
						_mm_stream_si128((__m128i*)(dptrc+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
						_mm_stream_si128((__m128i*)(dptrc+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;

					int r0 = sptrrj[0];
					int g0 = sptrgj[0];
					int b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U

					wsum = 1.f/wsum;
					b0 = cvRound(sum_b*wsum);
					g0 = cvRound(sum_g*wsum);
					r0 = cvRound(sum_r*wsum);
					dptr[3*j] = (uchar)r0; dptr[3*j+1] = (uchar)g0; dptr[3*j+2] = (uchar)b0;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};

class BilateralFilterORDER2_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilterORDER2_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float _color_f) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_f(_color_f)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();
		const float color2_f = 1.f/(color_f*color_f);
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					const __m128 one = _mm_set1_ps(1.f);
					const __m128 cef = _mm_set1_ps(color2_f);
					const __m128 mif = _mm_set1_ps(0.00001f);
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrj = sptr+j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_set1_ps(0.f);
						__m128 wval = _mm_set1_ps(0.f);

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128 sref = _mm_sub_ps(_mm_loadu_ps((sptrj+*ofs)),sval0);

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(sref,sref)))));

							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval,sref);
							wval = _mm_add_ps(wval,_w);
						}
						tval = _mm_div_ps(tval,wval);
						_mm_stream_ps((dptr+j),tval);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float v = (val - val0);
						float w = space_weight[k]*max(1.f -( color2_f*v*v),0.000001f);
						sum += val*w;
						wsum += w;
					}
					dptr[j] = sum/wsum;
				}
			}
		}
		else
		{
			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					const __m128 one = _mm_set1_ps(1.f);
					const __m128 cef = _mm_set1_ps(color2_f);
					const __m128 mif = _mm_set1_ps(0.00001f);
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;

						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++, spw++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							__m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							__m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							bref = _mm_sub_ps(bref,bval);
							__m128 fv = _mm_mul_ps(bref,bref);
							gref = _mm_sub_ps(gref,gval);
							fv = _mm_add_ps(_mm_mul_ps(gref,gref),fv);
							rref = _mm_sub_ps(rref,rval);
							fv = _mm_add_ps(_mm_mul_ps(rref,rref),fv);

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv)))));

							rref = _mm_mul_ps(_w, rref);
							gref = _mm_mul_ps(_w, gref);
							bref = _mm_mul_ps(_w, bref);

							rval1 = _mm_add_ps(rval1,rref);
							gval1 = _mm_add_ps(gval1,gref);
							bval1 = _mm_add_ps(bval1,bref);
							wval1 = _mm_add_ps(wval1,_w);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						gval1 = _mm_div_ps(gval1,wval1);
						bval1 = _mm_div_ps(bval1,wval1);

						float* dptrc = dptr+3*j;
						__m128 a = _mm_shuffle_ps(rval1,rval1,_MM_SHUFFLE(3,0,1,2));
						__m128 b = _mm_shuffle_ps(bval1,bval1,_MM_SHUFFLE(1,2,3,0));
						__m128 c = _mm_shuffle_ps(gval1,gval1,_MM_SHUFFLE(2,3,0,1));

						_mm_stream_ps((dptrc),_mm_blend_ps(_mm_blend_ps(b,a,4),c,2));
						_mm_stream_ps((dptrc+4),_mm_blend_ps(_mm_blend_ps(c,b,4),a,2));
						_mm_stream_ps((dptrc+8),_mm_blend_ps(_mm_blend_ps(a,c,4),b,2));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;

					float r0 = sptrrj[0];
					float g0 = sptrgj[0];
					float b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float v = (b - b0)*(b - b0) +(g - g0)*(g - g0) + (r - r0)*(r - r0);
						float w = space_weight[k]*max(1.f -( color2_f*v),0.000001f);
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					wsum = 1.f/wsum;
					dptr[3*j  ] = sum_b*wsum;
					dptr[3*j+1] = sum_g*wsum;
					dptr[3*j+2] = sum_r*wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight;
	float color_f;
};

class BilateralBoxFilterORDER2_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralBoxFilterORDER2_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float _color_f) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), color_f(_color_f)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();
		const float color2_f = 1.f/(color_f*color_f);
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					const __m128 one = _mm_set1_ps(1.f);
					const __m128 cef = _mm_set1_ps(color2_f);
					const __m128 mif = _mm_set1_ps(0.00001f);
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];

						const float* sptrj = sptr+j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_set1_ps(0.f);
						__m128 wval = _mm_set1_ps(0.f);

						for(k = 0;  k < maxk; k ++, ofs++)
						{
							__m128 sref = _mm_sub_ps(_mm_loadu_ps((sptrj+*ofs)),sval0);
							__m128 _w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(sref,sref))));

							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval,sref);
							wval = _mm_add_ps(wval,_w);
						}
						tval = _mm_div_ps(tval,wval);
						_mm_stream_ps((dptr+j),tval);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float v = (val - val0);
						float w = max(1.f -( color2_f*v*v),0.000001f);
						sum += val*w;
						wsum += w;
					}
					dptr[j] = sum/wsum;
				}
			}
		}
		else
		{
			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					const __m128 one = _mm_set1_ps(1.f);
					const __m128 cef = _mm_set1_ps(color2_f);
					const __m128 mif = _mm_set1_ps(0.00001f);
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];

						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;

						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							__m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							__m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							bref = _mm_sub_ps(bref,bval);
							__m128 fv = _mm_mul_ps(bref,bref);
							gref = _mm_sub_ps(gref,gval);
							fv = _mm_add_ps(_mm_mul_ps(gref,gref),fv);
							rref = _mm_sub_ps(rref,rval);
							fv = _mm_add_ps(_mm_mul_ps(rref,rref),fv);

							__m128 _w = _mm_max_ps(mif,_mm_sub_ps(one, _mm_mul_ps(cef,_mm_mul_ps(fv,fv))));

							rref = _mm_mul_ps(_w, rref);
							gref = _mm_mul_ps(_w, gref);
							bref = _mm_mul_ps(_w, bref);

							rval1 = _mm_add_ps(rval1,rref);
							gval1 = _mm_add_ps(gval1,gref);
							bval1 = _mm_add_ps(bval1,bref);
							wval1 = _mm_add_ps(wval1,_w);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						gval1 = _mm_div_ps(gval1,wval1);
						bval1 = _mm_div_ps(bval1,wval1);

						float* dptrc = dptr+3*j;
						__m128 a = _mm_shuffle_ps(rval1,rval1,_MM_SHUFFLE(3,0,1,2));
						__m128 b = _mm_shuffle_ps(bval1,bval1,_MM_SHUFFLE(1,2,3,0));
						__m128 c = _mm_shuffle_ps(gval1,gval1,_MM_SHUFFLE(2,3,0,1));

						_mm_stream_ps((dptrc),_mm_blend_ps(_mm_blend_ps(b,a,4),c,2));
						_mm_stream_ps((dptrc+4),_mm_blend_ps(_mm_blend_ps(c,b,4),a,2));
						_mm_stream_ps((dptrc+8),_mm_blend_ps(_mm_blend_ps(a,c,4),b,2));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;

					float r0 = sptrrj[0];
					float g0 = sptrgj[0];
					float b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float v = (b - b0)*(b - b0) +(g - g0)*(g - g0) + (r - r0)*(r - r0);
						float w = max(1.f -( color2_f*v),0.000001f);
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					wsum = 1.f/wsum;
					dptr[3*j  ] = sum_b*wsum;
					dptr[3*j+1] = sum_g*wsum;
					dptr[3*j+2] = sum_r*wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float color_f;
};

class BilateralFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();

#if CV_SSE4_1
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			int CV_DECL_ALIGNED(16) buf[4];

			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrj = sptr+j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_set1_ps(0.f);
						__m128 wval = _mm_set1_ps(0.f);

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128 sref = _mm_loadu_ps((sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval0,sref), *(const __m128*)v32f_absmask)));

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval,sref);
							wval = _mm_add_ps(wval,_w);
						}
						tval = _mm_div_ps(tval,wval);
						_mm_stream_ps((dptr+j),tval);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(val - val0))];
						sum += val*w;
						wsum += w;
					}
					dptr[j] = sum/wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;

						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++, spw++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							__m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							__m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							_mm_store_si128((__m128i*)buf,
								_mm_cvtps_epi32(
								_mm_add_ps(
								_mm_add_ps(
								_mm_and_ps(_mm_sub_ps(rval,rref), *(const __m128*)v32f_absmask),
								_mm_and_ps(_mm_sub_ps(gval,gref), *(const __m128*)v32f_absmask)),
								_mm_and_ps(_mm_sub_ps(bval,bref), *(const __m128*)v32f_absmask)
								)
								));

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							rref = _mm_mul_ps(_w, rref);
							gref = _mm_mul_ps(_w, gref);
							bref = _mm_mul_ps(_w, bref);

							rval1 = _mm_add_ps(rval1,rref);
							gval1 = _mm_add_ps(gval1,gref);
							bval1 = _mm_add_ps(bval1,bref);
							wval1 = _mm_add_ps(wval1,_w);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						gval1 = _mm_div_ps(gval1,wval1);
						bval1 = _mm_div_ps(bval1,wval1);

						float* dptrc = dptr+3*j;
						__m128 a = _mm_shuffle_ps(rval1,rval1,_MM_SHUFFLE(3,0,1,2));
						__m128 b = _mm_shuffle_ps(bval1,bval1,_MM_SHUFFLE(1,2,3,0));
						__m128 c = _mm_shuffle_ps(gval1,gval1,_MM_SHUFFLE(2,3,0,1));

						_mm_stream_ps((dptrc),_mm_blend_ps(_mm_blend_ps(b,a,4),c,2));
						_mm_stream_ps((dptrc+4),_mm_blend_ps(_mm_blend_ps(c,b,4),a,2));
						_mm_stream_ps((dptrc+8),_mm_blend_ps(_mm_blend_ps(a,c,4),b,2));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;

					float r0 = sptrrj[0];
					float g0 = sptrgj[0];
					float b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					wsum = 1.f/wsum;
					dptr[3*j  ] = sum_b*wsum;
					dptr[3*j+1] = sum_g*wsum;
					dptr[3*j+2] = sum_r*wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};



class BilateralFilter_32f_shuffle_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_32f_shuffle_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();

#if CV_SSE4_1
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			int CV_DECL_ALIGNED(16) buf[4];

			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrj = sptr+j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_set1_ps(0.f);
						__m128 wval = _mm_set1_ps(0.f);

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128 sref = _mm_loadu_ps((sptrj+*ofs));
							sref = _mm_shuffle_ps(sref,sref,1);
							_mm_store_si128((__m128i*)buf,_mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval0,sref), *(const __m128*)v32f_absmask)));

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval,sref);
							wval = _mm_add_ps(wval,_w);
						}
						tval = _mm_div_ps(tval,wval);
						_mm_stream_ps((dptr+j),tval);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(val - val0))];
						sum += val*w;
						wsum += w;
					}
					dptr[j] = sum/wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;

						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++, spw++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							__m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							__m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							_mm_store_si128((__m128i*)buf,
								_mm_cvtps_epi32(
								_mm_add_ps(
								_mm_add_ps(
								_mm_and_ps(_mm_sub_ps(rval,rref), *(const __m128*)v32f_absmask),
								_mm_and_ps(_mm_sub_ps(gval,gref), *(const __m128*)v32f_absmask)),
								_mm_and_ps(_mm_sub_ps(bval,bref), *(const __m128*)v32f_absmask)
								)
								));

							__m128 _w = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_w,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							rref = _mm_mul_ps(_w, rref);
							gref = _mm_mul_ps(_w, gref);
							bref = _mm_mul_ps(_w, bref);

							rval1 = _mm_add_ps(rval1,rref);
							gval1 = _mm_add_ps(gval1,gref);
							bval1 = _mm_add_ps(bval1,bref);
							wval1 = _mm_add_ps(wval1,_w);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						gval1 = _mm_div_ps(gval1,wval1);
						bval1 = _mm_div_ps(bval1,wval1);

						float* dptrc = dptr+3*j;
						__m128 a = _mm_shuffle_ps(rval1,rval1,_MM_SHUFFLE(3,0,1,2));
						__m128 b = _mm_shuffle_ps(bval1,bval1,_MM_SHUFFLE(1,2,3,0));
						__m128 c = _mm_shuffle_ps(gval1,gval1,_MM_SHUFFLE(2,3,0,1));

						_mm_stream_ps((dptrc),_mm_blend_ps(_mm_blend_ps(b,a,4),c,2));
						_mm_stream_ps((dptrc+4),_mm_blend_ps(_mm_blend_ps(c,b,4),a,2));
						_mm_stream_ps((dptrc+8),_mm_blend_ps(_mm_blend_ps(a,c,4),b,2));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;

					float r0 = sptrrj[0];
					float g0 = sptrgj[0];
					float b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					wsum = 1.f/wsum;
					dptr[3*j  ] = sum_b*wsum;
					dptr[3*j+1] = sum_g*wsum;
					dptr[3*j+2] = sum_r*wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};
class WeightedBilateralFilter_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	WeightedBilateralFilter_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _weightMap, int _radiusH, int _radiusV,int _maxk,
		int* _space_ofs, int* _space_w_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), weightMap(&_weightMap),dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_w_ofs(_space_w_ofs),space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, cn = dest->channels(), k;
		Size size = dest->size();
		static int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			int CV_DECL_ALIGNED(16) buf[4];
			float* sptr = (float*)temp->ptr<float>(range.start+radiusV) + 4 * (radiusH/4 + 1);
			float* wptr = (float*)weightMap->ptr<float>(range.start+radiusV)+ 4 * (radiusH/4 + 1);
			float* dptr = dest->ptr<float>(range.start);
			const int sstep = temp->cols;
			const int dstep = dest->cols;
			const int wstep = weightMap->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep,wptr+=wstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;

						const float* sptrj = sptr+j;
						const float* wptrj = wptr+j;
						int* wofs = &space_w_ofs[0];
						const __m128 sval = _mm_load_ps(sptrj);

						__m128 wval = _mm_set1_ps(0.0f);
						__m128 tval = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++,wofs++,spw++)
						{
							__m128 sref = _mm_loadu_ps((sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval,sref), *(const __m128*)v32f_absmask)));
							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							__m128 _wm = _mm_loadu_ps(wptrj+*wofs);_w = _mm_mul_ps(_w,_wm);
							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval,sref);
							wval = _mm_add_ps(wval,_w);
						}
						tval = _mm_div_ps(tval,wval);
						_mm_stream_ps((dptr+j),tval);
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float val = sptr[j + space_ofs[k]];
						float w = wptr[j+space_w_ofs[k]]*space_weight[k]*color_weight[cvRound(std::abs(val - val0))];
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			assert( cn == 3 );
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;
			const int wstep = weightMap->cols;

			float* sptrb = (float*)temp->ptr(3*radiusV+3*range.start  ) + 4 * (radiusH/4 + 1);
			float* sptrg = (float*)temp->ptr(3*radiusV+3*range.start+1) + 4 * (radiusH/4 + 1);
			float* sptrr = (float*)temp->ptr(3*radiusV+3*range.start+2) + 4 * (radiusH/4 + 1);
			float* wptr = (float*)weightMap->ptr<float>(range.start+radiusV)+ 4 * (radiusH/4 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep,wptr+=wstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=4)//4 pixel unit
					{
						int* ofs = &space_ofs[0];
						float* spw = space_weight;
						const float* wptrj = wptr+j;
						int* wofs = &space_w_ofs[0];
						const float* sptrrj = sptrr+j;
						const float* sptrgj = sptrg+j;
						const float* sptrbj = sptrb+j;
						const __m128 bval = _mm_load_ps((sptrbj));
						const __m128 gval = _mm_load_ps((sptrgj));
						const __m128 rval = _mm_load_ps((sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj+*ofs));
							__m128 gref = _mm_loadu_ps((sptrgj+*ofs));
							__m128 rref = _mm_loadu_ps((sptrrj+*ofs));

							_mm_store_si128((__m128i*)buf,
								_mm_cvtps_epi32(
								_mm_add_ps(
								_mm_add_ps(
								_mm_and_ps(_mm_sub_ps(rval,rref), *(const __m128*)v32f_absmask),
								_mm_and_ps(_mm_sub_ps(gval,gref), *(const __m128*)v32f_absmask)),
								_mm_and_ps(_mm_sub_ps(bval,bref), *(const __m128*)v32f_absmask)
								)
								));

							const __m128 _sw = _mm_set1_ps(*spw);
							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							__m128 _wm = _mm_loadu_ps(wptrj+*wofs);_w = _mm_mul_ps(_w,_wm);
							rref = _mm_mul_ps(_w, rref);
							gref = _mm_mul_ps(_w, gref);
							bref = _mm_mul_ps(_w, bref);

							rval1 = _mm_add_ps(rval1,rref);
							gval1 = _mm_add_ps(gval1,gref);
							bval1 = _mm_add_ps(bval1,bref);
							wval1 = _mm_add_ps(wval1,_w);
						}

						rval1 = _mm_div_ps(rval1,wval1);//rrrr
						gval1 = _mm_div_ps(gval1,wval1);//gggg
						bval1 = _mm_div_ps(bval1,wval1);//bbbb


						float* dptrc = dptr+3*j;
						__m128 a = _mm_shuffle_ps(rval1,rval1,_MM_SHUFFLE(3,0,1,2));
						__m128 b = _mm_shuffle_ps(bval1,bval1,_MM_SHUFFLE(1,2,3,0));
						__m128 c = _mm_shuffle_ps(gval1,gval1,_MM_SHUFFLE(2,3,0,1));


						_mm_stream_ps((dptrc),_mm_blend_ps(_mm_blend_ps(b,a,4),c,2));
						_mm_stream_ps((dptrc+4),_mm_blend_ps(_mm_blend_ps(c,b,4),a,2));
						_mm_stream_ps((dptrc+8),_mm_blend_ps(_mm_blend_ps(a,c,4),b,2));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const float* sptrrj = sptrr+j;
					const float* sptrgj = sptrg+j;
					const float* sptrbj = sptrb+j;

					float r0 = sptrrj[0];
					float g0 = sptrgj[0];
					float b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = wptr[j+space_w_ofs[k]]*space_weight[k]*color_weight[cvRound(std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0))];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					wsum = 1.f/wsum;
					dptr[3*j  ] = sum_b*wsum;
					dptr[3*j+1] = sum_g*wsum;
					dptr[3*j+2] = sum_r*wsum;
				}
			}
		}
	}
private:
	const Mat *temp;
	const Mat *weightMap;
	Mat *dest;
	int radiusH, radiusV,maxk, *space_ofs, *space_w_ofs;
	float *space_weight, *color_weight;
};

class WeightedBilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	WeightedBilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const Mat& _weightMap, int _radiusH, int _radiusV,int _maxk,
		int* _space_ofs, int* _space_w_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), weightMap(&_weightMap),dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_w_ofs(_space_w_ofs),space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, cn = dest->channels(), k;
		Size size = dest->size();
#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			float* wptr = (float*)weightMap->ptr<float>(range.start+radiusV)+ 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;
			const int wstep = weightMap->cols;

			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep,wptr+=wstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];
						int* wofs = &space_w_ofs[0];
						float* spw = space_weight;
						const uchar* sptrj = sptr+j;
						const float* wptrj = wptr+j;
						const __m128i sval = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 tval1 = _mm_set1_ps(0.0f);
						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 tval2 = _mm_set1_ps(0.0f);
						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 tval3 = _mm_set1_ps(0.0f);
						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 tval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++,wofs++,spw++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval,sref),_mm_subs_epu8(sref,sval)));

							__m128i m1 = _mm_unpacklo_epi8(sref,zero);
							__m128i m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							__m128 _wm = _mm_loadu_ps(wptrj+*wofs);_w = _mm_mul_ps(_w,_wm);
							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1,_valF);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							_wm = _mm_loadu_ps(wptrj+*wofs+4);_w = _mm_mul_ps(_w,_wm);
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2,_valF);
							wval2 = _mm_add_ps(wval2,_w);

							m1 = _mm_unpackhi_epi8(sref,zero);
							m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							_wm = _mm_loadu_ps(wptrj+*wofs+8);_w = _mm_mul_ps(_w,_wm);
							_valF =_mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3,_w);
							tval3 = _mm_add_ps(tval3,_valF);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							_wm = _mm_loadu_ps(wptrj+*wofs+12);_w = _mm_mul_ps(_w,_wm);
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4,_w);
							tval4 = _mm_add_ps(tval4,_valF);
						}
						tval1 = _mm_div_ps(tval1,wval1);
						tval2 = _mm_div_ps(tval2,wval2);
						tval3 = _mm_div_ps(tval3,wval3);
						tval4 = _mm_div_ps(tval4,wval4);
						_mm_stream_si128((__m128i*)(dptr+j), _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)) , _mm_packs_epi32( _mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[j];
					const float* wptrj = wptr+j;
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						float w = wptrj[space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(val - val0)];
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			assert( cn == 3 );
			short CV_DECL_ALIGNED(16) buf[16];

			const int sstep = 3*temp->cols;
			const int dstep = 3*dest->cols;
			const int wstep = weightMap->cols;
			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);;
			float* wptr = (float*)weightMap->ptr<float>(range.start+radiusV)+ 16 * (radiusH/16 + 1);

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep,wptr+=wstep )
			{	
				j=0;	
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						__m128i r1,r2,g1,g2,b1,b2;
						__m128 _valr,_valg,_valb,_w;

						int* ofs = &space_ofs[0];
						int* wofs = &space_w_ofs[0];

						float* spw = space_weight;
						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;
						const float* wptrj  = wptr +j;
						const __m128i bval = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval = _mm_load_si128((__m128i*)(sptrrj));


						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 rval2 = _mm_set1_ps(0.0f);
						__m128 gval2 = _mm_set1_ps(0.0f);
						__m128 bval2 = _mm_set1_ps(0.0f);

						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 rval3 = _mm_set1_ps(0.0f);
						__m128 gval3 = _mm_set1_ps(0.0f);
						__m128 bval3 = _mm_set1_ps(0.0f);

						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 rval4 = _mm_set1_ps(0.0f);
						__m128 gval4 = _mm_set1_ps(0.0f);
						__m128 bval4 = _mm_set1_ps(0.0f);


						for(k = 0;  k < maxk; k ++, ofs++,wofs++,spw++)
						{
							const __m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));
							const __m128i zero = _mm_setzero_si128();
							r1 = _mm_add_epi8(_mm_subs_epu8(rval,rref),_mm_subs_epu8(rref,rval));
							r2 = _mm_unpackhi_epi8(r1,zero);
							r1 = _mm_unpacklo_epi8(r1,zero);

							g1 = _mm_add_epi8(_mm_subs_epu8(gval,gref),_mm_subs_epu8(gref,gval));
							g2 = _mm_unpackhi_epi8(g1,zero);
							g1 = _mm_unpacklo_epi8(g1,zero);

							r1 = _mm_add_epi16(r1,g1);
							r2 = _mm_add_epi16(r2,g2);

							b1 = _mm_add_epi8(_mm_subs_epu8(bval,bref),_mm_subs_epu8(bref,bval));
							b2 = _mm_unpackhi_epi8(b1,zero);
							b1 = _mm_unpacklo_epi8(b1,zero);

							r1 = _mm_add_epi16(r1,b1);
							r2 = _mm_add_epi16(r2,b2);

							_mm_store_si128((__m128i*)(buf+8),r2);
							_mm_store_si128((__m128i*)buf,r1);

							r1 = _mm_unpacklo_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);
							g1 = _mm_unpacklo_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);
							b1 = _mm_unpacklo_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);
							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							_w = _mm_mul_ps(_w,_mm_loadu_ps(wptrj+wofs[0]));

							_valr = _mm_cvtepi32_ps(r1);
							_valg = _mm_cvtepi32_ps(g1);
							_valb = _mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval1 = _mm_add_ps(rval1,_valr);
							gval1 = _mm_add_ps(gval1,_valg);
							bval1 = _mm_add_ps(bval1,_valb);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							_w = _mm_mul_ps(_w,_mm_loadu_ps(wptrj+wofs[0]+4));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval2 = _mm_add_ps(rval2,_valr);
							gval2 = _mm_add_ps(gval2,_valg);
							bval2 = _mm_add_ps(bval2,_valb);
							wval2 = _mm_add_ps(wval2,_w);

							r1 = _mm_unpackhi_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);

							g1 = _mm_unpackhi_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);

							b1 = _mm_unpackhi_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							_w = _mm_mul_ps(_w,_mm_loadu_ps(wptrj+wofs[0]+8));

							_valr =_mm_cvtepi32_ps(r1);
							_valg =_mm_cvtepi32_ps(g1);
							_valb =_mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval3 = _mm_add_ps(wval3,_w);
							rval3 = _mm_add_ps(rval3,_valr);
							gval3 = _mm_add_ps(gval3,_valg);
							bval3 = _mm_add_ps(bval3,_valb);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							_w = _mm_mul_ps(_w,_mm_loadu_ps(wptr+j+wofs[0]+12));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval4 = _mm_add_ps(wval4,_w);
							rval4 = _mm_add_ps(rval4,_valr);
							gval4 = _mm_add_ps(gval4,_valg);
							bval4 = _mm_add_ps(bval4,_valb);
						}


						rval1 = _mm_div_ps(rval1,wval1);
						rval2 = _mm_div_ps(rval2,wval2);
						rval3 = _mm_div_ps(rval3,wval3);
						rval4 = _mm_div_ps(rval4,wval4);
						r1 = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_div_ps(gval1,wval1);
						gval2 = _mm_div_ps(gval2,wval2);
						gval3 = _mm_div_ps(gval3,wval3);
						gval4 = _mm_div_ps(gval4,wval4);
						g1 = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_div_ps(bval1,wval1);
						bval2 = _mm_div_ps(bval2,wval2);
						bval3 = _mm_div_ps(bval3,wval3);
						bval4 = _mm_div_ps(bval4,wval4);
						b1 = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));


						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8
							(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);

						const __m128i bmask2 = _mm_setr_epi8
							(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

						r1 = _mm_shuffle_epi8(r1,mask1);
						g1 = _mm_shuffle_epi8(g1,mask2);
						b1 = _mm_shuffle_epi8(b1,mask3);
						uchar* dptrc = dptr+3*j;
						_mm_stream_si128((__m128i*)(dptrc),_mm_blendv_epi8(b1,_mm_blendv_epi8(r1,g1,bmask1),bmask2));
						_mm_stream_si128((__m128i*)(dptrc+16),_mm_blendv_epi8(g1,_mm_blendv_epi8(r1,b1,bmask2),bmask1));		
						_mm_stream_si128((__m128i*)(dptrc+32),_mm_blendv_epi8(b1,_mm_blendv_epi8(g1,r1,bmask2),bmask1));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;
					const float* wptrj = wptr+j;

					int r0 = sptrrj[0];
					int g0 = sptrgj[0];
					int b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = wptrj[space_w_ofs[k]]*space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U

					wsum = 1.f/wsum;
					b0 = cvRound(sum_b*wsum);
					g0 = cvRound(sum_g*wsum);
					r0 = cvRound(sum_r*wsum);
					dptr[3*j] = (uchar)r0; dptr[3*j+1] = (uchar)g0; dptr[3*j+2] = (uchar)b0;
				}
			}
		}
	}
private:
	const Mat *temp;
	const Mat *weightMap;
	Mat *dest;
	int radiusH, radiusV,maxk, *space_ofs, *space_w_ofs;
	float *space_weight, *color_weight;
};

void weightedBilateralFilter_32f( const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}

	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		weight.type() ==CV_32FC1 &&
		src.type() == dst.type() && src.size() == dst.size() &&
		src.data != dst.data );

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp,wtemp;
	int dpad = (4- src.cols%4)%4;
	int spad =  dpad + (4-(2*radiusH)%4)%4;
	if(spad<4) spad +=4;
	int lpad = 4*(radiusH/4+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		copyMakeBorder( weight, wtemp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);

		copyMakeBorder( weight, wtemp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}

	double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);

	vector<float> _color_weight(cn*color_range);
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	vector<int> _space_w_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];
	int* space_w_ofs = &_space_w_ofs[0];

	// initialize color-related bilateral filter coefficients

	for( i = 0; i < cn*color_range; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		j = -radiusH;

		for( ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_w_ofs[maxk] = (int)(i*wtemp.cols   + j);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}
	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	WeightedBilateralFilter_32f_InvokerSSE4 body(dest, temp, wtemp, radiusH,radiusV, maxk, space_ofs, space_w_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void weightedBilateralFilter_8u( const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}

	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.type() == dst.type() && src.size() == dst.size() &&
		src.data != dst.data );

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp,wtemp;
	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		copyMakeBorder( weight, wtemp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );

	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);

		copyMakeBorder( weight, wtemp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );

	}

	/*double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);*/
	const int color_range=256;

	vector<float> _color_weight(cn*color_range);
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	vector<int> _space_w_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];
	int* space_w_ofs = &_space_w_ofs[0];

	// initialize color-related bilateral filter coefficients

	for( i = 0; i < color_range*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		j = -radiusH;

		for( ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_w_ofs[maxk] = (int)(i*wtemp.cols   + j);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}
	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	WeightedBilateralFilter_8u_InvokerSSE4 body(dest, temp, wtemp, radiusH,radiusV, maxk, space_ofs, space_w_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralWeightMap_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		src.size() == dst.size() &&
		src.data != dst.data );

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (4- src.cols%4)%4;
	int spad =  dpad + (4-(2*radiusH)%4)%4;
	if(spad<4) spad +=4;
	int lpad = 4*(radiusH/4+1)-radiusH;
	int rpad = spad-lpad;

	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);

	vector<float> _color_weight(cn*color_range);

	//float CV_DECL_ALIGNED(16) _space_weight[255];
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients

	for( i = 0; i < color_range*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		j = -radiusH;

		for( ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),CV_32F);
	BilateralWeightMap_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralWeightMap_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.size() == dst.size() &&
		src.data != dst.data );

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;

	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	/*double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);
	*/
	const int color_range=256;

	vector<float> _color_weight(cn*color_range);

	//float CV_DECL_ALIGNED(16) _space_weight[255];
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients

	for( i = 0; i < color_range*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		j = -radiusH;

		for( ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),CV_32F);
	BilateralWeightMap_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralBoxFilterORDER2_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	vector<int> _space_ofs(kernelSize.area()+1);

	int* space_ofs = &_space_ofs[0];

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		for(j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	BilateralBoxFilterORDER2_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, (float)sigma_color);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralFilterORDER2_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);

	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		for(j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	BilateralFilterORDER2_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, (float)sigma_color);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralFilter_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType, bool isRectangle)
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	Size size = src.size();

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (4- src.cols%4)%4;
	int spad =  dpad + (4-(2*radiusH)%16)%4;
	if(spad<4) spad +=4;
	int lpad = 4*(radiusH/4+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);

	vector<float> _color_weight(cn*color_range);

	//float CV_DECL_ALIGNED(16) _space_weight[255];
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < color_range*cn; i++ )
	{
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
	}

	// initialize space-related bilateral filter coefficients
	int maxk=0;
	setSpaceKernel(space_weight, space_ofs, maxk, radiusH,radiusV, gauss_space_coeff, temp.cols*cn, isRectangle);

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	BilateralFilter_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralBoxFilterORDER2_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	
	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (4- src.cols%4)%4;
	int spad =  dpad + (4-(2*radiusH)%16)%4;
	if(spad<4) spad +=4;
	int lpad = 4*(radiusH/4+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	vector<int> _space_ofs(kernelSize.area()+1);
	int* space_ofs = &_space_ofs[0];

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		for(j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}
	//cout<<"MAXK: "<<maxk<<","<<kernelSize.area()<< endl;

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),src.type());
	BilateralBoxFilterORDER2_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, (float)sigma_color);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralFilterORDER2_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (4- src.cols%4)%4;
	int spad =  dpad + (4-(2*radiusH)%16)%4;
	if(spad<4) spad +=4;
	int lpad = 4*(radiusH/4+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	//float CV_DECL_ALIGNED(16) _space_weight[255];
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];


	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		for(j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}
	//cout<<"MAXK: "<<maxk<<","<<kernelSize.area()<< endl;

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),src.type());
	BilateralFilterORDER2_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight,(float)sigma_color);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralFilter_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType, bool isRectangle)
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);


	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	const int color_range=256;
	vector<float> _color_weight(cn*color_range);
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < color_range*cn; i++ )
	{
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
	}

	// initialize space-related bilateral filter coefficients
	int maxk=0;
	setSpaceKernel( space_weight, space_ofs, maxk, radiusH, radiusV, gauss_color_coeff, temp.cols*cn, isRectangle);
	
	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),src.type());
	BilateralFilter_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void bilateralFilterSP_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width<=1) src.copyTo(dst);

	else bilateralFilter_8u(src, dst, Size(kernelSize.width,1), sigma_color, sigma_space, borderType, true);
	if(kernelSize.width>1) 
		bilateralFilter_8u(dst, dst, Size(1,kernelSize.height), sigma_color, sigma_space, borderType, true);
}

void bilateralFilterSP_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width<=1) src.copyTo(dst);
	else bilateralFilter_32f(src, dst, Size(kernelSize.width,1), sigma_color, sigma_space, borderType, true);

	if(kernelSize.width>1) 
		bilateralFilter_32f(dst, dst, Size(1,kernelSize.height), sigma_color, sigma_space, borderType, true);
}

void bilateralFilterSP_32f( const Mat& src, Mat& dst, int d, double sigma_color, double sigma_space, int borderType )
{
	bilateralFilterSP_32f(src,dst,Size(d,d),sigma_color,sigma_space,borderType);
}

void bilateralFilter_direction_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType, const int direction, bool isRectangle)
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3));

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);


	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	const int color_range=256;
	vector<float> _color_weight(cn*color_range);
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients
	for(int i = 0; i < color_range*cn; i++ )
	{
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);
	}

	// initialize space-related bilateral filter coefficients
	int maxk=0;
	if(direction>0)
	{
		set1DSpaceKernel45( space_weight, space_ofs, maxk, radiusH, radiusV, gauss_color_coeff, temp.cols*cn, isRectangle);
	}
	else
	{
		set1DSpaceKernel135( space_weight, space_ofs, maxk, radiusH, radiusV, gauss_color_coeff, temp.cols*cn, isRectangle);
	}
	
	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),src.type());
	BilateralFilter_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void separableBilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, double alpha, int method, int borderType)
{
	if(method==DUAL_KERNEL_HV)
	{
		bilateralFilter(src, dst, Size(kernelSize.width,1), sigma_color, sigma_space, FILTER_RECTANGLE, borderType);
		jointBilateralFilter(dst, src, dst, Size(1, kernelSize.height), sigma_color*alpha, sigma_space, FILTER_RECTANGLE, borderType);
	}
	else if(method==DUAL_KERNEL_HVVH)
	{
		bilateralFilter(src, dst, Size(kernelSize.width,1), sigma_color, sigma_space, FILTER_RECTANGLE, borderType);
		jointBilateralFilter(dst, src, dst, Size(1, kernelSize.height), sigma_color*alpha, sigma_space, FILTER_RECTANGLE, borderType);
		Mat dst2;
		bilateralFilter(src, dst2, Size(1, kernelSize.height), sigma_color, sigma_space, FILTER_RECTANGLE, borderType);
		jointBilateralFilter(dst2, src, dst2, Size(kernelSize.width,1), sigma_color*alpha, sigma_space, FILTER_RECTANGLE, borderType);

		alphaBlend(dst,dst2,0.5,dst);
	}
	else if (method==DUAL_KERNEL_CROSS)
	{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1, true);
	}
	else if (method==DUAL_KERNEL_CROSSCROSS)
	{
		bilateralFilter_direction_8u(src, dst, kernelSize, sigma_color, sigma_space, borderType, 1, true);
		jointBilateralFilter_direction_8u(dst, src, dst, kernelSize, sigma_color*alpha, sigma_space, borderType, -1,true);
		
		Mat dst2(src.size(),src.type());
		bilateralFilter_direction_8u(src, dst2, kernelSize, sigma_color, sigma_space, borderType, -1, true);
		jointBilateralFilter_direction_8u(dst2, src, dst2, kernelSize, sigma_color*alpha, sigma_space, borderType, 1, true);

		alphaBlend(dst,dst2,0.5,dst);
	}
}

void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType)
{
	if(dst.empty())dst.create(src.size(),src.type());

	if(method== FILTER_CIRCLE || method== FILTER_DEFAULT)
	{
		if(src.depth()==CV_8U)
		{
			bilateralFilter_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType, false);
		}
		else if(src.depth()==CV_32F)
		{
			bilateralFilter_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType, false);
		}
	}
	else if(method== FILTER_RECTANGLE)
	{
		if(src.depth()==CV_8U)
		{
			bilateralFilter_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType, true);
		}
		else if(src.depth()==CV_32F)
		{
			bilateralFilter_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType, true);
		}
	}
	else if(method== FILTER_SEPARABLE)
	{
		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			bilateralFilterSP_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			bilateralFilterSP_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}
	/*
	else if(method==BILATERAL_ORDER2)
	{

		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			if(sigma_space<0)
				bilateralBoxFilterORDER2_8u(src,dst,kernelSize,sigma_color,borderType);// box space weight
			else
				bilateralFilterORDER2_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			if(sigma_space<0)
				bilateralBoxFilterORDER2_32f(src,dst,kernelSize,sigma_color,borderType);// box space weight
			else
				bilateralFilterORDER2_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}*/
	else if(FILTER_SLOWEST)
	{
		bilateralFilterSlowest(src,dst,kernelSize,sigma_color,sigma_space);
	}
}

void weightedBilateralFilterSP_8u( const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	Mat dst2 = dst.clone();
	weightedBilateralFilter_8u( src, weight, dst2, Size(kernelSize.width,1), sigma_color, sigma_space, borderType );
	weightedBilateralFilter_8u( dst2, weight, dst, Size(1,kernelSize.width), sigma_color, sigma_space, borderType );
}

void weightedBilateralFilterSP_32f( const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	Mat dst2 = dst.clone();
	weightedBilateralFilter_32f( src, weight, dst2, Size(kernelSize.width,1), sigma_color, sigma_space, borderType );
	weightedBilateralFilter_32f( dst2, weight, dst, Size(1,kernelSize.width), sigma_color, sigma_space, borderType );
}

void weightedBilateralFilter(const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType)
{
	if(dst.empty())dst.create(src.size(),src.type());
	if(method==FILTER_CIRCLE|| method== FILTER_DEFAULT)
	{
		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			weightedBilateralFilter_8u(src,weight,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			weightedBilateralFilter_32f(src,weight,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}
	else if(method==FILTER_SEPARABLE)
	{
		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			weightedBilateralFilterSP_8u(src,weight,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			weightedBilateralFilterSP_32f(src,weight,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}
}

void bilateralWeightMapSP_8u( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	Mat t1(src.size(),CV_32F);
	bilateralWeightMap_8u(src, t1, Size(kernelSize.width,1), sigma_color, sigma_space, borderType );
	bilateralWeightMap_8u(src, dst, Size(1,kernelSize.width), sigma_color, sigma_space, borderType );
	multiply(t1,dst,dst);
}

void bilateralWeightMapSP_32f( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	Mat t1(src.size(),CV_32F);
	bilateralWeightMap_32f(src, t1, Size(kernelSize.width,1), sigma_color, sigma_space, borderType );
	bilateralWeightMap_32f(src, dst, Size(1,kernelSize.height), sigma_color, sigma_space, borderType );
	multiply(t1,dst,dst);
}

void bilateralWeightMap(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType)
{
	if(dst.empty())dst.create(src.size(),CV_32F);
	if(method==FILTER_CIRCLE)
	{
		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			bilateralWeightMap_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			bilateralWeightMap_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}
	else if(method==FILTER_SEPARABLE)
	{
		if(src.type()==CV_MAKE_TYPE(CV_8U,src.channels()))
		{
			bilateralWeightMapSP_8u(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
		else if(src.type()==CV_MAKE_TYPE(CV_32F,src.channels()))
		{
			bilateralWeightMapSP_32f(src,dst,kernelSize,sigma_color,sigma_space,borderType);
		}
	}
	else if(method==BILATERAL_ORDER2)
	{
		;
	}
}