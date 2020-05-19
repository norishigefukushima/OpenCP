#include "viewsynthesis.hpp"
#include "depthfilter.hpp"
#include "GaussianFilter.hpp"
#include "binalyWeightedRangeFilter.hpp"
#include "minmaxfilter.hpp"
#include "shiftImage.hpp"
#include "blend.hpp"
#include "metrics.hpp"
#include "draw.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	
	void boundaryBlur(Mat& src, int tap, int direction)
	{
		Mat val = src.clone();
		//l2r
		if (direction > 0)
		{
			for (int j = 0; j < src.rows; j++)
			{
				uchar* s = src.ptr<uchar>(j);
				for (int i = tap; i < src.cols - tap; i++)
				{
					if (s[3 * i] == 0)
					{
						s[3 * (i - 1) + 0] = (2 * s[3 * (i - 1) + 0] + s[3 * (i - 2) + 0]) / 3;
						s[3 * (i - 1) + 1] = (2 * s[3 * (i - 1) + 1] + s[3 * (i - 2) + 1]) / 3;
						s[3 * (i - 1) + 2] = (2 * s[3 * (i - 1) + 2] + s[3 * (i - 2) + 2]) / 3;
						/*s[3*(i-1)+0] = (s[3*(i-2)+0]+s[3*(i-2)+0])/2;
						s[3*(i-1)+1] = (s[3*(i-2)+1]+s[3*(i-2)+1])/2;
						s[3*(i-1)+2] = (s[3*(i-2)+2]+s[3*(i-2)+2])/2;*/

						for (; i < src.cols - tap; i++)
						{
							if (s[3 * i] != 0)
								break;
						}
					}
				}
			}
		}
		//r2l
		if (direction < 0)
		{
			for (int j = 0; j < src.rows; j++)
			{
				uchar* s = src.ptr<uchar>(j);
				for (int i = src.cols - tap - 1; i >= tap; i--)
				{
					if (s[3 * i] == 0)
					{
						s[3 * (i + 1) + 0] = ((int)2 * s[3 * (i + 1) + 0] + s[3 * (i + 2) + 0]) / 3;
						s[3 * (i + 1) + 1] = ((int)2 * s[3 * (i + 1) + 1] + s[3 * (i + 2) + 1]) / 3;
						s[3 * (i + 1) + 2] = ((int)2 * s[3 * (i + 1) + 2] + s[3 * (i + 2) + 2]) / 3;

						for (; i >= tap; i--)
						{
							if (s[3 * i] != 0)
								break;
						}
					}
				}
			}
		}
	}
	void getZeroMask(Mat& src, Mat& mask)
	{
		const int size = src.size().area();
		if (mask.empty())
			mask = Mat::zeros(src.size(), src.type());
		else
			mask.setTo(0);
		uchar* s = src.data;
		uchar* m = mask.data;
		for (int i = 0; i < size; i++)
		{
			if (s[0] == 0 && s[1] == 0 && s[2] == 0)
			{
				m[0] = 255;
			}
			m++, s += 3;
		}
	}

	template <class T>
	void shiftImInvNN_(const Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
	{
		if (amp > 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = (uchar*)srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 1; i >= 0; i--)
				{
					if (m[i] != 255)continue;

					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}

					const int dest = (int)(disp*amp + 0.5);
					//const int dest = (int)(disp*amp);

					if (i - dest >= 0)
					{
						dim[3 * (i)+0] = sim[3 * (i - dest) + 0];
						dim[3 * (i)+1] = sim[3 * (i - dest) + 1];
						dim[3 * (i)+2] = sim[3 * (i - dest) + 2];
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else if (amp < 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = (uchar*)srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					if (m[i] != 255)continue;
					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}
					const int dest = (int)((-amp*disp) + 0.5);
					//const int dest = (int)((-amp*disp));

					if (i + dest < srcdisp.cols)
					{
						dim[3 * (i)+0] = sim[3 * (i + dest) + 0];
						dim[3 * (i)+1] = sim[3 * (i + dest) + 1];
						dim[3 * (i)+2] = sim[3 * (i + dest) + 2];
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else
		{
			//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
			srcim.copyTo(destim);
			mask.setTo(Scalar(255));
		}
	}



	template <class T>
	void shiftImInvNN_(const Mat& srcim, Mat& srcdisp, Mat& destim, float amp, int invalid = 0)
	{

		if (amp > 0)
		{
			for (int j = 0; j < srcdisp.rows; j++)
			{
				Mat im; copyMakeBorder(srcim, im, 0, 0, 2, 0, BORDER_REPLICATE);
				uchar* sim = (uchar*)im.ptr<uchar>(j); sim += 6;
				uchar* dim = destim.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(disp*amp + 0.5f);
					dim[3 * (i)+0] = sim[3 * (i - dest) + 0];
					dim[3 * (i)+1] = sim[3 * (i - dest) + 1];
					dim[3 * (i)+2] = sim[3 * (i - dest) + 2];
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 0, 2, BORDER_REPLICATE);
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = (uchar*)im.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(-amp*disp + 0.5f);//符号

					dim[3 * (i)+0] = sim[3 * (i + dest) + 0];
					dim[3 * (i)+1] = sim[3 * (i + dest) + 1];
					dim[3 * (i)+2] = sim[3 * (i + dest) + 2];
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
		}
	}

	template <class T>
	void shiftImInvCubicLUT_(const Mat& srcim, Mat& srcdisp, Mat& destim, float amp, int disp_amp, int invalid = 0)
	{
		const float cubic = -1.0f;
		//const float cubic = -0.7f;
		const float c1 = cubic;
		const float c2 = -5.f*cubic;
		const float c3 = 8.f*cubic;
		const float c4 = -4.f*cubic;
		const float c5 = 2.f + cubic;
		const float c6 = -(cubic + 3.f);

		CV_DECL_ALIGNED(16) float lut[256 * 4];
		//CV_DECL_ALIGNED(16) int buff[4];
		//CV_DECL_ALIGNED(16) float bufff[4];
		if (amp > 0)
		{
			for (int i = 0; i < 256; i++)
			{
				const int dest = (int)(i*amp);
				const float ia = (float)((i*amp) - dest);
				const float a = 1.f - ia;

				lut[4 * i + 0] = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
				lut[4 * i + 1] = c5* a* a* a + c6* a* a + 1.f;
				lut[4 * i + 2] = c5*ia*ia*ia + c6*ia*ia + 1.f;
				lut[4 * i + 3] = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;
			}
		}
		else if (amp < 0)
		{
			for (int i = 0; i < 256; i++)
			{
				const int dest = (int)(i*-amp);
				const float ia = (float)((-amp*i) - dest);
				const float a = 1.f - ia;


				lut[4 * i + 0] = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
				lut[4 * i + 1] = c5* a* a* a + c6* a* a + 1.f;
				lut[4 * i + 2] = c5*ia*ia*ia + c6*ia*ia + 1.f;
				lut[4 * i + 3] = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;
			}
		}

		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 2, 1, BORDER_REPLICATE);

			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 6;
				uchar* dim = destim.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(disp*amp);
					const float ia = (float)((disp*amp) - dest);

					if (ia == 0.f)
					{
						memcpy(dim + 3 * i, sim + 3 * (i - dest), 3);
					}
					else
					{
						int idx = ((int)disp) * 4;
						const float viaa = lut[idx + 0];
						const float iaa = lut[idx + 1];
						const float aa = lut[idx + 2];
						const float vaa = lut[idx + 3];
						dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 0] + aa*sim[3 * (i - dest) + 0] + iaa*sim[3 * (i - dest - 1) + 0] + viaa*sim[3 * (i - dest - 2) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 1] + aa*sim[3 * (i - dest) + 1] + iaa*sim[3 * (i - dest - 1) + 1] + viaa*sim[3 * (i - dest - 2) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 2] + aa*sim[3 * (i - dest) + 2] + iaa*sim[3 * (i - dest - 1) + 2] + viaa*sim[3 * (i - dest - 2) + 2]);

						/*		dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+0] + aa*sim[3*(i-dest)+0]+iaa*sim[3*(i-dest-1)+0]+viaa*sim[3*(i-dest-2)+0]+0.5f);
							dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+1] + aa*sim[3*(i-dest)+1]+iaa*sim[3*(i-dest-1)+1]+viaa*sim[3*(i-dest-2)+1]+0.5f);
							dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+2] + aa*sim[3*(i-dest)+2]+iaa*sim[3*(i-dest-1)+2]+viaa*sim[3*(i-dest-2)+2]+0.5f);*/


						/*__m128 mlut = _mm_load_ps(lut+idx);

						__m128 mcoeff = _mm_shuffle_ps(mlut, mlut, 0x00);
						__m128 mdst =           _mm_mul_ps(mcoeff, _mm_set_ps(0.f, sim[3*(i-dest+1)+2],sim[3*(i-dest+1)+1],sim[3*(i-dest+1)+0]));

						mcoeff = _mm_shuffle_ps(mlut, mlut, 0x55);
						mdst = _mm_add_ps(mdst, _mm_mul_ps(mcoeff, _mm_set_ps(0.f, sim[3*(i-dest  )+2],sim[3*(i-dest  )+1],sim[3*(i-dest  )+0])));

						mcoeff = _mm_shuffle_ps(mlut, mlut, 0xAA);
						mdst = _mm_add_ps(mdst, _mm_mul_ps(mcoeff, _mm_set_ps(0.f, sim[3*(i-dest-1)+2],sim[3*(i-dest-1)+1],sim[3*(i-dest-1)+0])));

						mcoeff = _mm_shuffle_ps(mlut, mlut, 0xFF);
						mdst = _mm_add_ps(mdst, _mm_mul_ps(mcoeff, _mm_set_ps(0.f, sim[3*(i-dest-2)+2],sim[3*(i-dest-2)+1],sim[3*(i-dest-2)+0])));

						//__m128i midst =  _mm_cvtps_epi32(mdst);
						//_mm_store_si128((__m128i*) buff, midst);
						_mm_store_ps( bufff, mdst);

						dim[3*(i)+0]=saturate_cast<uchar>(bufff[0]);
						dim[3*(i)+1]=saturate_cast<uchar>(bufff[1]);
						dim[3*(i)+2]=saturate_cast<uchar>(bufff[2]);*/
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 0, 2, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(-amp*disp);//符号
					const float ia = (float)((-amp*disp) - dest);
					if (ia == 0.f)
					{
						memcpy(dim + 3 * i, sim + 3 * (i + dest), 3);
					}
					else
					{
						int idx = ((int)disp) * 4;
						const float viaa = lut[idx + 0];
						const float iaa = lut[idx + 1];
						const float aa = lut[idx + 2];
						const float vaa = lut[idx + 3];

						dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 0] + aa*sim[3 * (i + dest) + 0] + iaa*sim[3 * (i + dest + 1) + 0] + viaa*sim[3 * (i + dest + 2) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 1] + aa*sim[3 * (i + dest) + 1] + iaa*sim[3 * (i + dest + 1) + 1] + viaa*sim[3 * (i + dest + 2) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 2] + aa*sim[3 * (i + dest) + 2] + iaa*sim[3 * (i + dest + 1) + 2] + viaa*sim[3 * (i + dest + 2) + 2]);
						/*dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+0] + aa*sim[3*(i+dest)+0]+iaa*sim[3*(i+dest+1)+0]+viaa*sim[3*(i+dest+2)+0]+0.5f);
						dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+1] + aa*sim[3*(i+dest)+1]+iaa*sim[3*(i+dest+1)+1]+viaa*sim[3*(i+dest+2)+1]+0.5f);
						dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+2] + aa*sim[3*(i+dest)+2]+iaa*sim[3*(i+dest+1)+2]+viaa*sim[3*(i+dest+2)+2]+0.5f);*/
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
		}
	}

	template <class T>
	void shiftImInvCubic_(const Mat& srcim, Mat& srcdisp, Mat& destim, double amp, int invalid = 0)
	{
		const float cubic = -1.f;

		const float c1 = cubic;
		const float c2 = -5.f*cubic;
		const float c3 = 8.f*cubic;
		const float c4 = -4.f*cubic;
		const float c5 = 2.f + cubic;
		const float c6 = -(cubic + 3.f);

		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 2, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 6;
				uchar* dim = destim.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(disp*amp);
					const float ia = (float)((disp*amp) - dest);
					const float a = 1.f - ia;

					if (ia == 0.0)
					{
						memcpy(dim + 3 * i, sim + 3 * (i - dest), 3);
					}
					else
					{
						const float viaa = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
						const float iaa = c5* a* a* a + c6* a* a + 1.f;
						const float aa = c5*ia*ia*ia + c6*ia*ia + 1.f;
						const float vaa = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;

						dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 0] + aa*sim[3 * (i - dest) + 0] + iaa*sim[3 * (i - dest - 1) + 0] + viaa*sim[3 * (i - dest - 2) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 1] + aa*sim[3 * (i - dest) + 1] + iaa*sim[3 * (i - dest - 1) + 1] + viaa*sim[3 * (i - dest - 2) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 2] + aa*sim[3 * (i - dest) + 2] + iaa*sim[3 * (i - dest - 1) + 2] + viaa*sim[3 * (i - dest - 2) + 2]);
						//dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+0] + aa*sim[3*(i-dest)+0]+iaa*sim[3*(i-dest-1)+0]+viaa*sim[3*(i-dest-2)+0]+0.5);
						//dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+1] + aa*sim[3*(i-dest)+1]+iaa*sim[3*(i-dest-1)+1]+viaa*sim[3*(i-dest-2)+1]+0.5);
						//dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+2] + aa*sim[3*(i-dest)+2]+iaa*sim[3*(i-dest-1)+2]+viaa*sim[3*(i-dest-2)+2]+0.5);
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 0, 2, BORDER_REPLICATE);
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(-amp*disp);//符号
					const float ia = (float)((-amp*disp) - dest);
					const float a = 1.f - ia;

					if (ia == 0.0)
					{
						memcpy(dim + 3 * i, sim + 3 * (i + dest), 3);
					}
					else
					{
						const float viaa = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
						const float iaa = c5* a* a* a + c6* a* a + 1.f;
						const float aa = c5*ia*ia*ia + c6*ia*ia + 1.f;
						const float vaa = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;

						dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 0] + aa*sim[3 * (i + dest) + 0] + iaa*sim[3 * (i + dest + 1) + 0] + viaa*sim[3 * (i + dest + 2) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 1] + aa*sim[3 * (i + dest) + 1] + iaa*sim[3 * (i + dest + 1) + 1] + viaa*sim[3 * (i + dest + 2) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 2] + aa*sim[3 * (i + dest) + 2] + iaa*sim[3 * (i + dest + 1) + 2] + viaa*sim[3 * (i + dest + 2) + 2]);
						/*dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+0] + aa*sim[3*(i+dest)+0]+iaa*sim[3*(i+dest+1)+0]+viaa*sim[3*(i+dest+2)+0]+0.5);
						dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+1] + aa*sim[3*(i+dest)+1]+iaa*sim[3*(i+dest+1)+1]+viaa*sim[3*(i+dest+2)+1]+0.5);
						dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+2] + aa*sim[3*(i+dest)+2]+iaa*sim[3*(i+dest+1)+2]+viaa*sim[3*(i+dest+2)+2]+0.5);*/
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
		}
	}

	template <class T>
	void shiftImInvCubic_(const Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
	{
		const float cubic = -1.f;

		const float c1 = cubic;
		const float c2 = -5.f*cubic;
		const float c3 = 8.f*cubic;
		const float c4 = -4.f*cubic;
		const float c5 = 2.f + cubic;
		const float c6 = -(cubic + 3.f);

		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 2, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 6;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					//if(m[i]!=255)continue;

					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}

					const int dest = (int)(disp*amp);
					const float ia = (float)((disp*amp) - dest);
					const float a = 1.f - ia;
					if (i >= dest)
					{
						if (ia == 0.0)
						{
							dim[3 * (i)+0] = sim[3 * (i - dest) + 0];
							dim[3 * (i)+1] = sim[3 * (i - dest) + 1];
							dim[3 * (i)+2] = sim[3 * (i - dest) + 2];
						}
						else
						{
							const float viaa = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
							const float iaa = c5* a* a* a + c6* a* a + 1.f;
							const float aa = c5*ia*ia*ia + c6*ia*ia + 1.f;
							const float vaa = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;

							dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 0] + aa*sim[3 * (i - dest) + 0] + iaa*sim[3 * (i - dest - 1) + 0] + viaa*sim[3 * (i - dest - 2) + 0]);
							dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 1] + aa*sim[3 * (i - dest) + 1] + iaa*sim[3 * (i - dest - 1) + 1] + viaa*sim[3 * (i - dest - 2) + 1]);
							dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i - dest + 1) + 2] + aa*sim[3 * (i - dest) + 2] + iaa*sim[3 * (i - dest - 1) + 2] + viaa*sim[3 * (i - dest - 2) + 2]);
						}
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 0, 2, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					//if(m[i]!=255)continue;
					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}

					const int dest = (int)(-amp*disp);//符号
					const float ia = (float)((-amp*disp) - dest);
					const float a = 1.f - ia;

					if (i + dest < srcdisp.cols)
					{
						if (ia == 0.0)
						{
							dim[3 * (i)+0] = sim[3 * (i + dest) + 0];
							dim[3 * (i)+1] = sim[3 * (i + dest) + 1];
							dim[3 * (i)+2] = sim[3 * (i + dest) + 2];
						}
						else
						{
							const float viaa = c1*(1.f + a)*(1.f + a)*(1.f + a) + c2*(1.f + a)*(1.f + a) + c3*(1.f + a) + c4;
							const float iaa = c5* a* a* a + c6* a* a + 1.f;
							const float aa = c5*ia*ia*ia + c6*ia*ia + 1.f;
							const float vaa = c1*(1.f + ia)*(1.f + ia)*(1.f + ia) + c2*(1.f + ia)*(1.f + ia) + c3*(1.f + ia) + c4;
							/*	const float viaa= lut[0];
							const float iaa = lut[1];
							const float aa  = lut[2];
							const float vaa = lut[3];*/

							dim[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 0] + aa*sim[3 * (i + dest) + 0] + iaa*sim[3 * (i + dest + 1) + 0] + viaa*sim[3 * (i + dest + 2) + 0]);
							dim[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 1] + aa*sim[3 * (i + dest) + 1] + iaa*sim[3 * (i + dest + 1) + 1] + viaa*sim[3 * (i + dest + 2) + 1]);
							dim[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * (i + dest - 1) + 2] + aa*sim[3 * (i + dest) + 2] + iaa*sim[3 * (i + dest + 1) + 2] + viaa*sim[3 * (i + dest + 2) + 2]);
						}
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
			mask.setTo(Scalar(255));
		}
	}


	template <class T>
	void shiftImInvLinear_(const Mat& srcim, Mat& srcdisp, Mat& destim, float amp, int invalid = 0)
	{
		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 0, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(disp*amp);


					const float ia = (float)((disp*amp) - dest);
					const float a = 1.f - ia;

					/*dim[3*(i)+0]=saturate_cast<uchar>(a*sim[3*(i-dest)+0]+ia*sim[3*(i-dest-1)+0]+0.5f);
					dim[3*(i)+1]=saturate_cast<uchar>(a*sim[3*(i-dest)+1]+ia*sim[3*(i-dest-1)+1]+0.5f);
					dim[3*(i)+2]=saturate_cast<uchar>(a*sim[3*(i-dest)+2]+ia*sim[3*(i-dest-1)+2]+0.5f);*/
					dim[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 0] + ia*sim[3 * (i - dest - 1) + 0]);
					dim[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 1] + ia*sim[3 * (i - dest - 1) + 1]);
					dim[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 2] + ia*sim[3 * (i - dest - 1) + 2]);
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 0, 2, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					if (disp == invalid)
					{
						memset(dim + 3 * i, 0, 3);
						continue;
					}

					const int dest = (int)(-amp*disp);//符号
					const float ia = (-amp*disp) - dest;
					const float a = 1.f - ia;

					/*dim[3*(i)+0]=saturate_cast<uchar>(a*sim[3*(i+dest)+0]+ia*sim[3*(i+dest+1)+0]+0.5f);
					dim[3*(i)+1]=saturate_cast<uchar>(a*sim[3*(i+dest)+1]+ia*sim[3*(i+dest+1)+1]+0.5f);
					dim[3*(i)+2]=saturate_cast<uchar>(a*sim[3*(i+dest)+2]+ia*sim[3*(i+dest+1)+2]+0.5f);*/

					dim[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 0] + ia*sim[3 * (i + dest + 1) + 0]);
					dim[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 1] + ia*sim[3 * (i + dest + 1) + 1]);
					dim[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 2] + ia*sim[3 * (i + dest + 1) + 2]);
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
		}
	}

	template <class T>
	void shiftImInvLinear_(const Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
	{
		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);

				for (int i = 0; i < srcdisp.cols; i++)
				{
					if (m[i] != 255)continue;

					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}

					const int dest = (int)(disp*amp);
					const double ia = (disp*amp) - dest;
					const double a = 1.0 - ia;

					if (i >= dest)
					{
						dim[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 0] + ia*sim[3 * (i - dest - 1) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 1] + ia*sim[3 * (i - dest - 1) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * (i - dest) + 2] + ia*sim[3 * (i - dest - 1) + 2]);
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					if (m[i] != 255)continue;
					const T disp = s[i];
					if (disp == invalid)
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
						continue;
					}

					const int dest = (int)(-amp*disp);//符号
					const double ia = (-amp*disp) - dest;
					const double a = 1.0 - ia;

					if (i + dest < srcdisp.cols)
					{
						dim[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 0] + ia*sim[3 * (i + dest + 1) + 0]);
						dim[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 1] + ia*sim[3 * (i + dest + 1) + 1]);
						dim[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * (i + dest) + 2] + ia*sim[3 * (i + dest + 1) + 2]);
					}
					else
					{
						dim[3 * (i)+0] = 0;
						dim[3 * (i)+1] = 0;
						dim[3 * (i)+2] = 0;
						m[i] = 0;
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
			mask.setTo(Scalar(255));
		}
	}

	template <class T>
	void shiftImInvWithMask_(const Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0, int inter_method = INTER_CUBIC)
	{
		destim.setTo(0);

		if (inter_method == INTER_CUBIC)
			shiftImInvCubic_<T>(srcim, srcdisp, destim, amp, mask, invalid);
		else if (inter_method == INTER_LINEAR)
			shiftImInvLinear_<T>(srcim, srcdisp, destim, amp, mask, invalid);
		else if (inter_method == INTER_NEAREST)
			shiftImInvNN_<T>(srcim, srcdisp, destim, amp, mask, invalid);
		else
			cout << "support types are NN, linear and cubic" << endl;
	}


	template <class T>
	void shiftImInv_(const Mat& srcim, Mat& srcdisp, Mat& destim, float amp, int invalid = 0, int inter_method = INTER_CUBIC)
	{
		destim.setTo(0);

		if (inter_method == INTER_CUBIC)
		{
			if (srcdisp.type() != CV_8U)shiftImInvCubic_<T>(srcim, srcdisp, destim, amp, invalid);
			else shiftImInvCubicLUT_<T>(srcim, srcdisp, destim, amp, invalid);
		}
		else if (inter_method == INTER_LINEAR)
			shiftImInvLinear_<T>(srcim, srcdisp, destim, amp, invalid);
		else if (inter_method == INTER_NEAREST)
			shiftImInvNN_<T>(srcim, srcdisp, destim, amp, invalid);
		else
			cout << "support types are NN, linear and cubic" << endl;
	}

	void shiftImInv(const Mat& srcim, Mat& srcdisp, Mat& destim, float amp, int invalid = 0, int inter_method = INTER_CUBIC)
	{
		int depth = srcdisp.depth();
		if (depth == CV_8U) shiftImInv_<uchar>(srcim, srcdisp, destim, amp, invalid, inter_method);
		if (depth == CV_16S) shiftImInv_<short>(srcim, srcdisp, destim, amp, invalid, inter_method);
		if (depth == CV_16U) shiftImInv_<ushort>(srcim, srcdisp, destim, amp, invalid, inter_method);
		if (depth == CV_32F) shiftImInv_<float>(srcim, srcdisp, destim, amp, invalid, inter_method);
	}

	//without mask mad jump
	template <class T>
	void shiftDisp_(const Mat& srcdisp, Mat& destdisp, float amp, const int sub_gap)
	{
		Mat dsp; copyMakeBorder(srcdisp, dsp, 0, 0, 1, 1, BORDER_REPLICATE);
		//const int offset = (int)(256*amp+0.5);
		const int offset = (int)(256);
		Mat dst = Mat::zeros(Size(destdisp.cols + offset, 1), destdisp.type());

		if (amp > 0)
		{
			for (int j = 0; j < srcdisp.rows; j++)
			{
				dst.setTo(0);
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0); d += offset;

				for (int i = srcdisp.cols; i >= 0; i--)
				{
					const T disp = s[i];
					const int sub = (int)(abs(disp - s[i - 1]));

					bool issub = (sub <= sub_gap && sub>0) ? true : false;
					const int dest = (int)(disp*amp);

					if (disp > d[i - dest])
					{
						d[i - dest] = disp;
						if (issub)
						{
							if (disp > d[i - dest - 1])
							{
								d[i - dest - 1] = (T)((disp + s[i - 1])*0.5);
								//d[i-dest-1]=disp;
							}
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
			}
		}
		else if (amp < 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				dst.setTo(0);
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					const int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap && sub>0) ? true : false;
					//bool issub = (sub<=sub_gap)?true:false;

					const int dest = (int)(-amp*disp);

					if (disp > d[i + dest])
					{
						d[i + dest] = (T)disp;
						if (issub)
						{
							if (disp > d[i + dest + 1])
							{
								d[i + dest + 1] = (T)((disp + s[i + 1])*0.5);
								//d[i+dest+1]=(disp);
							}
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
			}
		}
		else
		{
			srcdisp.copyTo(destdisp);
		}
	}

	//without mask
	template <class T>
	void shiftDisp_(const Mat& srcdisp, Mat& destdisp, float amp, const int large_jump, const int sub_gap)
	{
		Mat dsp; copyMakeBorder(srcdisp, dsp, 0, 0, 1, 1, BORDER_REPLICATE);
		//const int offset = (int)(256*amp);
		const int offset = (int)(256);
		Mat dst = Mat::zeros(Size(destdisp.cols + offset, 1), destdisp.type());

		int ij = 0;
		const int ljump = max(large_jump, 1);

		if (amp > 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				dst.setTo(0);
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0); d += offset;

				for (int i = srcdisp.cols; i >= 0; i--)
				{
					const T disp = s[i];
					const int sub = (int)(abs(disp - s[i - 1]));

					bool issub = (sub <= sub_gap && sub>0) ? true : false;
					const int dest = (int)(disp*amp);

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
						//if(s[i-1]-disp>ljump)
					{
						i -= ij;
						continue;
					}

					if (disp > d[i - dest])
					{
						d[i - dest] = disp;
						if (issub)
						{
							if (disp > d[i - dest - 1])
							{
								d[i - dest - 1] = (T)((disp + s[i - 1])*0.5);
								//d[i-dest-1]=disp;
							}
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
			}
		}
		else if (amp<0)
		{
			for (int j = 0; j<srcdisp.rows; j++)
			{
				dst.setTo(0);
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					const int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap && sub>0) ? true : false;

					const int dest = (int)(-amp*disp);

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (disp > d[i + dest])
					{
						d[i + dest] = (T)disp;
						if (issub)
						{
							if (disp > d[i + dest + 1])
							{
								d[i + dest + 1] = (T)((disp + s[i + 1])*0.5);
							}
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
			}
		}
		else
		{
			srcdisp.copyTo(destdisp);
		}
	}


	//with mask
	template <class T>
	void shiftDisp_(const Mat& srcdisp, Mat& destdisp, float amp, Mat& mask, const int large_jump, const int sub_gap)
	{
		Mat dsp; copyMakeBorder(srcdisp, dsp, 0, 0, 1, 1, BORDER_REPLICATE);
		const int offset = 256;
		Mat dst = Mat::zeros(Size(destdisp.cols + offset, 1), destdisp.type());
		Mat msk = Mat::zeros(Size(destdisp.cols + offset, 1), CV_8U);
		int ij = 0;
		const int ljump = max(large_jump, 1);
		//const int iamp=cvRound(amp);
		if (amp > 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				dst.setTo(0); msk.setTo(0);
				uchar* m = msk.ptr<uchar>(0); m += offset;
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0); d += offset;

				for (int i = srcdisp.cols; i >= 0; i--)
					//for(int i=0;i<srcdisp.cols;i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));

					bool issub = (sub <= sub_gap && sub>0) ? true : false;
					//bool issub = (sub<=sub_gap)?true:false;
					const int dest = (int)(disp*amp);

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
						//if(s[i-1]-disp>ljump)
					{
						i -= ij;
						continue;
					}

					if (disp > d[i - dest])
					{
						m[i - dest] = 255;
						d[i - dest] = disp;
					}
					if (issub)
					{
						if (disp > d[i - dest - 1])
						{
							m[i - dest - 1] = 255;
							d[i - dest - 1] = (T)((disp + s[i - 1])*0.5);
							//d[i-dest-1]=disp;
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
				memcpy(mask.ptr<uchar>(j), m, sizeof(uchar)*destdisp.cols);
			}
			//imshowNormalize("amp>0",destdisp);
		}
		else if (amp<0)
		{
			//#pragma omp parallel for
			for (int j = 0; j<srcdisp.rows; j++)
			{
				dst.setTo(0); msk.setTo(0);
				uchar* m = msk.ptr<uchar>(0);
				T* s = dsp.ptr<T>(j); s += 1;
				T* d = dst.ptr<T>(0);
				for (int i = 0; i < srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap && sub>0) ? true : false;
					//bool issub = (sub<=sub_gap)?true:false;

					const int dest = (int)(-amp*disp);

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (disp > d[i + dest])
					{
						m[i + dest] = 255;
						d[i + dest] = (T)disp;
					}
					if (issub)
					{
						if (disp > d[i + dest + 1])
						{
							m[i + dest + 1] = 255;
							d[i + dest + 1] = (T)((disp + s[i + 1])*0.5);
							//d[i+dest+1]=(disp);
						}
					}
				}
				memcpy(destdisp.ptr<T>(j), d, sizeof(T)*destdisp.cols);
				memcpy(mask.ptr<uchar>(j), m, sizeof(uchar)*destdisp.cols);
			}
			//imshowNormalize("amp<0",destdisp);
		}
		else
		{
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}


	template <class T>
	void shiftImDispNN_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
	{
		int ij = 0;
		const int ljump = max(large_jump, 1);

		if (amp > 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 1; i >= 1; i--)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(disp*amp + 0.5);
					//const int dest = (int)(disp*amp);

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
					{
						i -= ij;
						continue;
					}

					if (i > dest)
					{
						if (disp > d[i - dest])
						{
							m[i - dest] = 255;
							d[i - dest] = disp;
							dim[3 * (i - dest) + 0] = sim[3 * i + 0];
							dim[3 * (i - dest) + 1] = sim[3 * i + 1];
							dim[3 * (i - dest) + 2] = sim[3 * i + 2];
							if (issub)
							{

								m[i - dest - 1] = 255;
								d[i - dest - 1] = disp;
								dim[3 * (i - dest - 1) + 0] = sim[3 * i - 3];
								dim[3 * (i - dest - 1) + 1] = sim[3 * i - 2];
								dim[3 * (i - dest - 1) + 2] = sim[3 * i - 1];
							}
						}
					}
				}
			}
		}
		else if (amp < 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j<srcdisp.rows; j++)
			{
				uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);
				for (int i = 0; i<srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)((-amp*disp) + 0.5);
					//const int dest = (int)((-amp*disp));

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (i + dest + 1 < srcdisp.cols)
					{
						if (disp > d[i + dest])
						{
							m[i + dest] = 255;
							d[i + dest] = (T)disp;

							dim[3 * (i + dest) + 0] = sim[3 * i + 0];
							dim[3 * (i + dest) + 1] = sim[3 * i + 1];
							dim[3 * (i + dest) + 2] = sim[3 * i + 2];

							if (issub)
							{
								m[i + dest + 1] = 255;
								d[i + dest + 1] = (T)disp;
								dim[3 * (i + dest + 1) + 0] = sim[3 * i + 3];
								dim[3 * (i + dest + 1) + 1] = sim[3 * i + 4];
								dim[3 * (i + dest + 1) + 2] = sim[3 * i + 5];
							}
						}
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}

	template <class T>
	void shiftImDispCubicS_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask)
	{
		//	cout<<"Cubic"<<endl;
		const double cubic = -1.0;
		const double c1 = cubic;
		const double c2 = -5.0*cubic;
		const double c3 = 8.0*cubic;
		const double c4 = -4.0*cubic;
		const double c5 = 2.0 + cubic;
		const double c6 = -(cubic + 3.0);

		if (amp > 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 2; i >= 0; i--)
				{
					const T disp = s[i];

					const int dest = (int)(disp*amp);
					const double ia = ((double)disp*amp) - dest;
					const double a = 1.0 - ia;


					if (i - dest - 1 >= 0 && i - dest - 1 < srcdisp.cols - 1)
					{
						if (disp > d[i - dest])
						{
							if (ia == 0.0)
							{
								m[i - dest] = 255;
								d[i - dest] = disp;
								dim[3 * (i - dest) + 0] = sim[3 * i + 0];
								dim[3 * (i - dest) + 1] = sim[3 * i + 1];
								dim[3 * (i - dest) + 2] = sim[3 * i + 2];
							}
							else
							{
								const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
								const double iaa = c5* a* a* a + c6* a* a + 1.0;
								const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
								const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;

								m[i - dest] = 255;
								d[i - dest] = disp;
								dim[3 * (i - dest) + 0] = saturate_cast<uchar>(vaa*sim[3 * i - 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i + 3] + viaa*sim[3 * i + 6]);
								dim[3 * (i - dest) + 1] = saturate_cast<uchar>(vaa*sim[3 * i - 2] + aa*sim[3 * i + 1] + iaa*sim[3 * i + 4] + viaa*sim[3 * i + 7]);
								dim[3 * (i - dest) + 2] = saturate_cast<uchar>(vaa*sim[3 * i - 1] + aa*sim[3 * i + 2] + iaa*sim[3 * i + 5] + viaa*sim[3 * i + 8]);
							}
						}
					}
				}
			}
		}
		else if (amp < 0)
		{
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);
				for (int i = 1; i < srcdisp.cols; i++)
				{
					const T disp = s[i];

					const int dest = (int)(-amp*disp);//符号
					const double ia = (-(double)disp*amp) - dest;
					const double a = 1.0 - ia;

					if (i + dest + 1 >= 0 && i + dest + 1 < srcdisp.cols - 1)
					{
						if (disp > d[i + dest])
						{
							if (ia == 0.0)
							{
								m[i + dest] = 255;
								d[i + dest] = (T)disp;
								dim[3 * (i + dest) + 0] = sim[3 * i + 0];
								dim[3 * (i + dest) + 1] = sim[3 * i + 1];
								dim[3 * (i + dest) + 2] = sim[3 * i + 2];
							}
							else
							{
								const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
								const double iaa = c5* a* a* a + c6* a* a + 1.0;
								const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
								const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;
								m[i + dest] = 255;
								d[i + dest] = disp;

								dim[3 * (i + dest) + 0] = saturate_cast<uchar>(vaa*sim[3 * i + 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i - 3] + viaa*sim[3 * i - 6]);
								dim[3 * (i + dest) + 1] = saturate_cast<uchar>(vaa*sim[3 * i + 4] + aa*sim[3 * i + 1] + iaa*sim[3 * i - 2] + viaa*sim[3 * i - 5]);
								dim[3 * (i + dest) + 2] = saturate_cast<uchar>(vaa*sim[3 * i + 5] + aa*sim[3 * i + 2] + iaa*sim[3 * i - 1] + viaa*sim[3 * i - 4]);
							}
						}
					}
				}
			}
		}
		else
		{
			//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
			srcim.copyTo(destim);
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}

	template <class T>
	void shiftImDispCubic_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
	{
		const double cubic = -1.0;
		const double c1 = cubic;
		const double c2 = -5.0*cubic;
		const double c3 = 8.0*cubic;
		const double c4 = -4.0*cubic;
		const double c5 = 2.0 + cubic;
		const double c6 = -(cubic + 3.0);

		int ij = 0;
		const int ljump = max(large_jump, 1);

		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 2, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 1; i >= 1; i--)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(disp*amp);
					const double ia = ((double)disp*amp) - dest;
					const double a = 1.0 - ia;

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
					{
						i -= ij;
						continue;
					}

					if (i > dest)
					{
						if (disp > d[i - dest])
						{
							if (ia == 0.0)
							{
								m[i - dest] = 255;
								d[i - dest] = disp;
								dim[3 * (i - dest) + 0] = sim[3 * i + 0];
								dim[3 * (i - dest) + 1] = sim[3 * i + 1];
								dim[3 * (i - dest) + 2] = sim[3 * i + 2];
								if (issub)
								{
									m[i - dest - 1] = 255;
									d[i - dest - 1] = disp;
									dim[3 * (i - dest - 1) + 0] = sim[3 * i - 3];
									dim[3 * (i - dest - 1) + 1] = sim[3 * i - 2];
									dim[3 * (i - dest - 1) + 2] = sim[3 * i - 1];
								}
							}
							else
							{
								const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
								const double iaa = c5* a* a* a + c6* a* a + 1.0;
								const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
								const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;

								m[i - dest] = 255;
								d[i - dest] = disp;
								dim[3 * (i - dest) + 0] = saturate_cast<uchar>(vaa*sim[3 * i - 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i + 3] + viaa*sim[3 * i + 6]);
								dim[3 * (i - dest) + 1] = saturate_cast<uchar>(vaa*sim[3 * i - 2] + aa*sim[3 * i + 1] + iaa*sim[3 * i + 4] + viaa*sim[3 * i + 7]);
								dim[3 * (i - dest) + 2] = saturate_cast<uchar>(vaa*sim[3 * i - 1] + aa*sim[3 * i + 2] + iaa*sim[3 * i + 5] + viaa*sim[3 * i + 8]);
								if (issub)
								{
									m[i - dest - 1] = 255;
									d[i - dest - 1] = disp;
									dim[3 * (i - dest - 1) + 0] = saturate_cast<uchar>(vaa*sim[3 * i - 6] + aa*sim[3 * i - 3] + iaa*sim[3 * i + 0] + viaa*sim[3 * i + 3]);
									dim[3 * (i - dest - 1) + 1] = saturate_cast<uchar>(vaa*sim[3 * i - 5] + aa*sim[3 * i - 2] + iaa*sim[3 * i + 1] + viaa*sim[3 * i + 4]);
									dim[3 * (i - dest - 1) + 2] = saturate_cast<uchar>(vaa*sim[3 * i - 4] + aa*sim[3 * i - 1] + iaa*sim[3 * i + 2] + viaa*sim[3 * i + 5]);
								}
							}
						}
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 2, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j<srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 6;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);
				for (int i = 0; i<srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(-amp*disp);//符号
					const double ia = (-(double)disp*amp) - dest;
					const double a = 1.0 - ia;

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (i + dest + 1 < srcdisp.cols)
					{
						if (disp > d[i + dest])
						{
							if (ia == 0.0)
							{
								m[i + dest] = 255;
								d[i + dest] = (T)disp;
								dim[3 * (i + dest) + 0] = sim[3 * i + 0];
								dim[3 * (i + dest) + 1] = sim[3 * i + 1];
								dim[3 * (i + dest) + 2] = sim[3 * i + 2];
								if (issub)
								{
									m[i + dest + 1] = 255;
									d[i + dest + 1] = (T)disp;
									dim[3 * (i + dest + 1) + 0] = sim[3 * i + 3];
									dim[3 * (i + dest + 1) + 1] = sim[3 * i + 4];
									dim[3 * (i + dest + 1) + 2] = sim[3 * i + 5];
								}
							}
							else
							{
								const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
								const double iaa = c5* a* a* a + c6* a* a + 1.0;
								const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
								const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;
								m[i + dest] = 255;
								d[i + dest] = disp;

								dim[3 * (i + dest) + 0] = saturate_cast<uchar>(vaa*sim[3 * i + 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i - 3] + viaa*sim[3 * i - 6]);
								dim[3 * (i + dest) + 1] = saturate_cast<uchar>(vaa*sim[3 * i + 4] + aa*sim[3 * i + 1] + iaa*sim[3 * i - 2] + viaa*sim[3 * i - 5]);
								dim[3 * (i + dest) + 2] = saturate_cast<uchar>(vaa*sim[3 * i + 5] + aa*sim[3 * i + 2] + iaa*sim[3 * i - 1] + viaa*sim[3 * i - 4]);
								if (issub)
								{
									m[i + dest + 1] = 255;
									d[i + dest + 1] = (T)disp;

									dim[3 * (i + dest + 1) + 0] = saturate_cast<uchar>(vaa*sim[3 * i + 6] + aa*sim[3 * i + 3] + iaa*sim[3 * i + 0] + viaa*sim[3 * i - 3]);
									dim[3 * (i + dest + 1) + 1] = saturate_cast<uchar>(vaa*sim[3 * i + 7] + aa*sim[3 * i + 4] + iaa*sim[3 * i + 1] + viaa*sim[3 * i - 2]);
									dim[3 * (i + dest + 1) + 2] = saturate_cast<uchar>(vaa*sim[3 * i + 8] + aa*sim[3 * i + 5] + iaa*sim[3 * i + 2] + viaa*sim[3 * i - 1]);
								}
							}
						}
					}
				}
			}
		}
		else
		{
			//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
			srcim.copyTo(destim);
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}

	template <class T>
	void shiftImDispLinear_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
	{
		int ij = 0;
		const int ljump = max(large_jump, 1);
		//const int iamp=cvRound(amp);
		if (amp > 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 1; i >= 1; i--)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(disp*amp);
					const double ia = (double)(disp*amp) - dest;
					const double a = 1.0 - ia;

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
					{
						i -= ij;
						continue;
					}

					if (i > dest)
					{
						if (disp > d[i - dest])
						{
							m[i - dest] = 255;
							d[i - dest] = disp;

							dim[3 * (i - dest) + 0] = saturate_cast<uchar>(a*sim[3 * i + 0] + ia*sim[3 * i + 3]);
							dim[3 * (i - dest) + 1] = saturate_cast<uchar>(a*sim[3 * i + 1] + ia*sim[3 * i + 4]);
							dim[3 * (i - dest) + 2] = saturate_cast<uchar>(a*sim[3 * i + 2] + ia*sim[3 * i + 5]);

							if (issub)
							{

								m[i - dest - 1] = 255;
								d[i - dest - 1] = disp;

								dim[3 * (i - dest - 1) + 0] = saturate_cast<uchar>(a*sim[3 * i - 3] + ia*sim[3 * i + 0]);
								dim[3 * (i - dest - 1) + 1] = saturate_cast<uchar>(a*sim[3 * i - 2] + ia*sim[3 * i + 1]);
								dim[3 * (i - dest - 1) + 2] = saturate_cast<uchar>(a*sim[3 * i - 1] + ia*sim[3 * i + 2]);
							}
						}
					}
				}
			}
		}
		else if (amp < 0)
		{
			Mat im; copyMakeBorder(srcim, im, 0, 0, 1, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j<srcdisp.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);
				for (int i = 0; i<srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(-amp*disp);//符号
					const double ia = (-(double)disp*amp) - dest;
					const double a = 1.0 - ia;

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (i + dest + 1 < srcdisp.cols)
					{
						if (disp > d[i + dest])
						{
							m[i + dest] = 255;
							d[i + dest] = (T)disp;

							dim[3 * (i + dest) + 0] = saturate_cast<uchar>(a*sim[3 * i + 0] + ia*sim[3 * i - 3]);
							dim[3 * (i + dest) + 1] = saturate_cast<uchar>(a*sim[3 * i + 1] + ia*sim[3 * i - 2]);
							dim[3 * (i + dest) + 2] = saturate_cast<uchar>(a*sim[3 * i + 2] + ia*sim[3 * i - 1]);

							if (issub)
							{
								m[i + dest + 1] = 255;
								d[i + dest + 1] = (T)disp;

								dim[3 * (i + dest + 1) + 0] = saturate_cast<uchar>(a*sim[3 * i + 3] + ia*sim[3 * i + 0]);
								dim[3 * (i + dest + 1) + 1] = saturate_cast<uchar>(a*sim[3 * i + 4] + ia*sim[3 * i + 1]);
								dim[3 * (i + dest + 1) + 2] = saturate_cast<uchar>(a*sim[3 * i + 5] + ia*sim[3 * i + 2]);
							}
						}
					}
				}
			}
		}
		else
		{
			//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
			srcim.copyTo(destim);
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}

	//input mask should be set to 0
	template <class T>
	void shiftImDisp(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, double sub_gap, const int large_jump = 3, Mat& mask = Mat(), int warpInterpolationMethod = INTER_CUBIC)
	{
		Mat mask_ = mask;
		if (mask_.empty())mask_ = Mat::zeros(srcim.size(), CV_8U);

		if (srcdisp.depth() == CV_8U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_8U);
			else destdisp.setTo(0);
			if (destim.empty())destim = Mat::zeros(srcim.size(), CV_8UC3);
			else destim.setTo(0);
		}
		else if (srcdisp.depth() == CV_16S)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16S);
			else destdisp.setTo(0);
			if (destim.empty())destim = Mat::zeros(srcim.size(), CV_8UC3);
			else destim.setTo(0);
		}
		else if (srcdisp.depth() == CV_16U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16U);
			else destdisp.setTo(0);
			if (destim.empty())destim = Mat::zeros(srcim.size(), CV_8UC3);
			else destim.setTo(0);
		}
		else if (srcdisp.depth() == CV_32F)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_32F);
			else destdisp.setTo(0);
			if (destim.empty())destim = Mat::zeros(srcim.size(), CV_8UC3);
			else destim.setTo(0);
		}

		if (warpInterpolationMethod == INTER_NEAREST)
			shiftImDispNN_<T>(srcim, srcdisp, destim, destdisp, amp, mask_, large_jump, (int)sub_gap);
		else if (warpInterpolationMethod == INTER_LINEAR)
			shiftImDispLinear_<T>(srcim, srcdisp, destim, destdisp, amp, mask_, large_jump, (int)sub_gap);
		else if (warpInterpolationMethod == INTER_CUBIC)
		{
			shiftImDispCubic_<T>(srcim, srcdisp, destim, destdisp, amp, mask_, large_jump, (int)sub_gap);
			//shiftImDispCubicS_<T>(srcim,srcdisp,destim,destdisp,amp,mask_);
		}
		else
		{
			cout << "support types are NN, linear and cubic" << endl;
		}
		mask_.copyTo(mask);
	}

	void shiftDisp(const Mat& srcdisp, Mat& destdisp, float amp, float sub_gap, const int large_jump = 3, Mat& mask = Mat())
	{
		if (srcdisp.depth() == CV_8U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_8U);
			else destdisp.setTo(0);

			if (mask.empty())
			{
				if (large_jump == 0)
				{
					shiftDisp_<uchar>(srcdisp, destdisp, amp, (int)sub_gap);
				}
				else
				{
					shiftDisp_<uchar>(srcdisp, destdisp, amp, large_jump, (int)sub_gap);
				}
			}
			else
			{
				shiftDisp_<uchar>(srcdisp, destdisp, amp, mask, large_jump, (int)sub_gap);
			}
		}
		else if (srcdisp.depth() == CV_16S)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16S);
			else destdisp.setTo(0);

			if (mask.empty())
			{
				if (large_jump == 0)
				{
					shiftDisp_<short>(srcdisp, destdisp, amp, (int)sub_gap);
				}
				else
				{
					shiftDisp_<short>(srcdisp, destdisp, amp, large_jump, (int)sub_gap);
				}
			}
			else
			{
				shiftDisp_<short>(srcdisp, destdisp, amp, mask, large_jump, (int)sub_gap);
			}
		}
		else if (srcdisp.depth() == CV_16U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16U);
			else destdisp.setTo(0);

			if (mask.empty())
			{
				if (large_jump == 0)
				{
					shiftDisp_<ushort>(srcdisp, destdisp, amp, (int)sub_gap);
				}
				else
				{
					shiftDisp_<ushort>(srcdisp, destdisp, amp, large_jump, (int)sub_gap);
				}
			}
			else
			{
				shiftDisp_<ushort>(srcdisp, destdisp, amp, mask, large_jump, (int)sub_gap);
			}
		}
		else if (srcdisp.depth() == CV_32F)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_32F);
			else destdisp.setTo(0);

			if (mask.empty())
			{
				if (large_jump == 0)
				{
					shiftDisp_<float>(srcdisp, destdisp, amp, (int)sub_gap);
				}
				else
				{
					shiftDisp_<float>(srcdisp, destdisp, amp, large_jump, (int)sub_gap);
				}
			}
			else
			{
				shiftDisp_<float>(srcdisp, destdisp, amp, mask, large_jump, (int)sub_gap);
			}
		}
	}

	template <class T>
	void fillOcclusionImDisp2_(Mat& im, Mat& src, T invalidvalue, int maxlength = 1000)
	{
		//Mat mask=Mat::zeros(im.size(),CV_8U);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			//	uchar* m = mask.ptr<uchar>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = 255;//可能性のある最大値を入力
			s[src.cols - 1] = 255;//可能性のある最大値を入力
			//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
			if (j > 0 && j < src.rows - 1)
			{
				for (int i = 1; i < src.cols; i++)
				{
					if (s[i] <= invalidvalue)
					{
						int t = i;
						int count = 0;
						do
						{
							count++;
							t++;
							if (count > maxlength)break;
							if (t > src.cols - 2)break;
						} while (s[t] <= invalidvalue);
						if (count > maxlength)break;


						uchar ce[3];
						T dd;
						if (s[i - 1]<s[t])
						{
							dd = s[i - 1];

							int count = 1;
							int r = ims[3 * i - 3];
							int g = ims[3 * i - 2];
							int b = ims[3 * i - 1];
							if (s[i - 1 - src.cols]>invalidvalue)
							{
								r += ims[3 * (i - 1 - src.cols) + 0];
								g += ims[3 * (i - 1 - src.cols) + 1];
								b += ims[3 * (i - 1 - src.cols) + 2];
								count++;
							}
							if (s[i - 1 + src.cols] > invalidvalue)
							{
								r += ims[3 * (i - 1 + src.cols) + 0];
								g += ims[3 * (i - 1 + src.cols) + 1];
								b += ims[3 * (i - 1 + src.cols) + 2];
								count++;
							}
							ce[0] = cvRound(r / (double)count);
							ce[1] = cvRound(g / (double)count);
							ce[2] = cvRound(b / (double)count);
						}
						else
						{
							dd = s[t];
							int count = 1;
							int r = ims[3 * t + 0];
							int g = ims[3 * t + 1];
							int b = ims[3 * t + 2];
							if (s[t - src.cols] > invalidvalue)
							{
								r += ims[3 * (t - src.cols) + 0];
								g += ims[3 * (t - src.cols) + 1];
								b += ims[3 * (t - src.cols) + 2];
								count++;
							}
							if (s[t + src.cols] > invalidvalue)
							{
								r += ims[3 * (i + src.cols) + 0];
								g += ims[3 * (i + src.cols) + 1];
								b += ims[3 * (i + src.cols) + 2];
								count++;
							}
							ce[0] = r / count;
							ce[1] = g / count;
							ce[2] = b / count;
						}

						if (t - i > src.cols - 3)
						{
							for (int n = 0; n < src.cols; n++)
							{
								s[i] = invalidvalue;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
						else
						{
							for (; i < t; i++)
							{
								s[i] = dd;
								//m[i]=255;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
					}
				}
			}
			else
			{
				for (int i = 1; i < src.cols; i++)
				{
					if (s[i] <= invalidvalue)
					{
						uchar cs[3];
						cs[0] = ims[3 * i - 3];
						cs[1] = ims[3 * i - 2];
						cs[2] = ims[3 * i - 1];
						int t = i;
						int count = 0;
						do
						{
							count++;
							t++;
							if (count > maxlength)break;
							if (t > src.cols - 2)break;
						} while (s[t] <= invalidvalue);
						if (count > maxlength)break;

						uchar ce[3];

						T dd;
						if (s[i - 1] < s[t])
						{
							dd = s[i - 1];
							ce[0] = cs[0];
							ce[1] = cs[1];
							ce[2] = cs[2];
						}
						else
						{
							dd = s[t];
							ce[0] = ims[3 * t + 0];
							ce[1] = ims[3 * t + 1];
							ce[2] = ims[3 * t + 2];
						}

						if (t - i > src.cols - 3)
						{
							for (int n = 0; n < src.cols; n++)
							{
								s[i] = invalidvalue;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
						else
						{
							for (; i < t; i++)
							{
								s[i] = dd;
								//m[i]=255;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
					}
				}
			}

			s[0] = st;//もとに戻す
			if (st <= invalidvalue)
			{
				s[0] = s[1];
				ims[0] = ims[3];
				ims[1] = ims[4];
				ims[2] = ims[5];
			}
			s[src.cols - 1] = ed;
			if (ed <= invalidvalue)
			{
				s[src.cols - 1] = s[src.cols - 2];
				ims[3 * src.cols - 3] = ims[3 * src.cols - 6];
				ims[3 * src.cols - 2] = ims[3 * src.cols - 5];
				ims[3 * src.cols - 1] = ims[3 * src.cols - 4];
			}
		}
	}

	template <class T>
	void fillOcclusionImDispH_(Mat& im, Mat& src, T invalidvalue, Mat& distance, int maxlength = 1000)
	{
		distance = Mat::zeros(im.size(), CV_16U);
		T maxval;
		if (sizeof(T) == 1)maxval = 255;
		else maxval = (T)SHRT_MAX;
		//	Mat mask=Mat::zeros(im.size(),CV_8U);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			ushort* dist = distance.ptr<ushort>(j);
			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			//	uchar* m = mask.ptr<uchar>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = maxval;
			s[src.cols - 1] = maxval;

			for (int i = 1; i < src.cols; i++)
			{
				if (s[i] <= invalidvalue)
				{
					uchar CV_DECL_ALIGNED(16) cs[3];
					/*cs[0]=ims[3*i-3];
					cs[1]=ims[3*i-2];
					cs[2]=ims[3*i-1];*/
					cs[0] = (ims[3 * i - 3] + ims[3 * i - 6]) >> 1;
					cs[1] = (ims[3 * i - 2] + ims[3 * i - 5]) >> 1;
					cs[2] = (ims[3 * i - 1] + ims[3 * i - 4]) >> 1;
					int t = i;
					int count = 0;
					do
					{
						count++;
						t++;
						//if(count>maxlength)break;

					} while (s[t] <= invalidvalue);
					//if(count>maxlength)break;

					uchar CV_DECL_ALIGNED(16) ce[3];

					T dd;
					bool flag = true;
					if (s[i - 1] < s[t])
					{
						dd = s[i - 1];
						ce[0] = cs[0];
						ce[1] = cs[1];
						ce[2] = cs[2];
					}
					else
					{
						flag = false;
						dd = s[t];
						/*ce[0]=ims[3*t+0];
						ce[1]=ims[3*t+1];
						ce[2]=ims[3*t+2];*/
						ce[0] = (ims[3 * t + 0] + ims[3 * t + 3]) >> 1;
						ce[1] = (ims[3 * t + 1] + ims[3 * t + 4]) >> 1;
						ce[2] = (ims[3 * t + 2] + ims[3 * t + 5]) >> 1;
					}

					if (t - i > src.cols - 3)
					{
						for (int n = 0; n < src.cols; n++)
						{
							s[i] = invalidvalue;
							ims[3 * i + 0] = ce[0];
							ims[3 * i + 1] = ce[1];
							ims[3 * i + 2] = ce[2];
						}
					}
					else
					{
						if (flag)
						{
							int count = 0;
							for (; i < t; i++)
							{
								s[i] = dd;
								dist[i] = count++;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
						else
						{
							int count = t - i;
							for (; i < t; i++)
							{
								s[i] = dd;
								dist[i] = count--;
								ims[3 * i + 0] = ce[0];
								ims[3 * i + 1] = ce[1];
								ims[3 * i + 2] = ce[2];
							}
						}
					}
				}
			}
			s[0] = st;
			if (st <= invalidvalue)
			{
				s[0] = s[1];
				ims[0] = ims[3];
				ims[1] = ims[4];
				ims[2] = ims[5];
			}
			s[src.cols - 1] = ed;
			if (ed <= invalidvalue)
			{
				s[src.cols - 1] = s[src.cols - 2];
				ims[3 * src.cols - 3] = ims[3 * src.cols - 6];
				ims[3 * src.cols - 2] = ims[3 * src.cols - 5];
				ims[3 * src.cols - 1] = ims[3 * src.cols - 4];
			}
		}
	}


	template <class T>
	void fillOcclusionImDispHV_(Mat& im, Mat& src, T invalidvalue, int maxlength = 1000)
	{
		//Mat mask;comapre(src,invalidvalue,mask,CMP_EQ);
		Mat imH = im.clone();
		Mat dispH = src.clone();

		Mat imV; transpose(im, imV);
		Mat dispV; transpose(src, dispV);

		Mat distH;
		Mat distV;

		fillOcclusionImDispH_<T>(imH, dispH, invalidvalue, distH, maxlength);
		fillOcclusionImDispH_<T>(imV, dispV, invalidvalue, distV, maxlength);
		transpose(dispV, src);
		transpose(imV, im);

		uchar* sh = imH.data;
		T* dh = dispH.ptr<T>(0);

		uchar* s = im.data;
		T* d = src.ptr<T>(0);

		for (int i = 0; i < im.size().area(); i++)
		{
			if (d[i] > dh[i])
			{
				s[3 * i + 0] = sh[3 * i + 0];
				s[3 * i + 1] = sh[3 * i + 1];
				s[3 * i + 2] = sh[3 * i + 2];
				d[i] = dh[i];
			}
		}
	}




	template <class T>
	void fillOcclusionImDisp_(Mat& im, Mat& src, T invalidvalue, int maxlength = 1000)
	{
		T maxval;
		if (sizeof(T) == 1)maxval = 255;
		else maxval = (T)32000;
		//	Mat mask=Mat::zeros(im.size(),CV_8U);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			//	uchar* m = mask.ptr<uchar>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = maxval;
			s[src.cols - 1] = maxval;

			for (int i = 1; i < src.cols - 1; i++)
			{
				if (s[i] <= invalidvalue)
				{
					uchar CV_DECL_ALIGNED(16) cs[3];
					memcpy(cs, ims + 3 * (i - 1), 3);
					int t = i;
					int count = 0;
					do
					{
						count++;
						t++;
						//if(count>maxlength)break;
					} while (s[t] <= invalidvalue);
					//if(count>maxlength)break;

					uchar CV_DECL_ALIGNED(16) ce[3];
					T dd;
					if (s[i - 1] < s[t])
					{
						dd = s[i - 1];
						memcpy(ce, cs, 3);
					}
					else
					{
						dd = s[t];
						memcpy(ce, ims + 3 * t, 3);
					}

					/*memset(s+i, dd, t-i);
					for(;i<t;i++)
					{
					memcpy(ims+3*i, ce, 3);
					}*/

					for (; i < t; i++)
					{
						s[i] = dd;
						ims[3 * i + 0] = ce[0];
						ims[3 * i + 1] = ce[1];
						ims[3 * i + 2] = ce[2];
					}
				}
			}
			s[0] = st;//もとに戻す
			if (st <= invalidvalue)
			{
				s[0] = s[1];
				ims[0] = ims[3];
				ims[1] = ims[4];
				ims[2] = ims[5];
			}
			s[src.cols - 1] = ed;
			if (ed <= invalidvalue)
			{
				s[src.cols - 1] = s[src.cols - 2];
				ims[3 * src.cols - 3] = ims[3 * src.cols - 6];
				ims[3 * src.cols - 2] = ims[3 * src.cols - 5];
				ims[3 * src.cols - 1] = ims[3 * src.cols - 4];
			}
		}
	}

	template <class T>
	void fillOcclusionImDispReflect_(Mat& im, Mat& src, T invalidvalue, int maxlength = 1000)
	{
		Mat imc;
		Mat sc;
		im.copyTo(imc);
		src.copyTo(sc);
		fillOcclusionImDisp_<T>(imc, sc, invalidvalue);

		//Mat mask=Mat::zeros(im.size(),CV_8U);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			T* dref = sc.ptr<T>(j);
			uchar* imref = imc.ptr<uchar>(j);

			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			//uchar* m = mask.ptr<uchar>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = 255;//可能性のある最大値を入力
			s[src.cols - 1] = 255;//可能性のある最大値を入力
			//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
			for (int i = 1; i < src.cols; i++)
			{
				if (s[i] <= invalidvalue)
				{
					int t = i;
					int count = 0;
					do
					{
						count++;
						t++;
						if (count > maxlength)break;
						if (t > src.cols - 2)break;
					} while (s[t] <= invalidvalue);
					if (count > maxlength)break;
					//外枠は例外
					if (t == src.cols - 1)
					{
						memcpy(ims + 3 * i, imref + 3 * i, 3 * (src.cols - 1 - i));
						memcpy(s + sizeof(T)*i, dref + sizeof(T)*i, sizeof(T)*(src.cols - 1 - i));
						continue;
					}
					if (i == 1)
					{
						memcpy(ims, imref, 3 * t);
						memcpy(s, dref, sizeof(T)*t);
						i = t;
						continue;
					}

					T dd;

					if (s[i - 1] < s[t])
					{
						dd = s[i - 1];
						int p = i;
						int count = 1;
						for (; i < t; i++)
						{
							s[i] = dd;
							//m[i]=255;
							ims[3 * i + 0] = imref[3 * (p - count) + 0];
							ims[3 * i + 1] = imref[3 * (p - count) + 1];
							ims[3 * i + 2] = imref[3 * (p - count++) + 2];
							/*ims[3*i+0]=0;
							ims[3*i+1]=255;
							ims[3*i+2]=0;*/
						}
					}
					else
					{
						dd = s[t];
						int count = t - i + 1;
						for (int k = 1; k < count; k++)
						{
							s[t - k] = dd;
							//m[t-k]=255;
							ims[3 * (t - k) + 0] = imref[3 * (t + k) + 0];
							ims[3 * (t - k) + 1] = imref[3 * (t + k) + 1];
							ims[3 * (t - k) + 2] = imref[3 * (t + k) + 2];
							/*ims[3*(t-k)+0]=0;
							ims[3*(t-k)+1]=0;
							ims[3*(t-k)+2]=255;*/
						}
					}
				}
			}
			s[0] = st;//もとに戻す
			if (st <= invalidvalue)
			{
				s[0] = s[1];
				ims[0] = ims[3];
				ims[1] = ims[4];
				ims[2] = ims[5];
			}
			s[src.cols - 1] = ed;
			if (ed <= invalidvalue)
			{
				s[src.cols - 1] = s[src.cols - 2];
				ims[3 * src.cols - 3] = ims[3 * src.cols - 6];
				ims[3 * src.cols - 2] = ims[3 * src.cols - 5];
				ims[3 * src.cols - 1] = ims[3 * src.cols - 4];
			}
		}
	}

	template <class T>
	void fillOcclusionImDispStretch_(Mat& im, Mat& src, T invalidvalue, int maxlength = 1000)
	{
		Mat imc;
		Mat sc;
		im.copyTo(imc);
		src.copyTo(sc);
		fillOcclusionImDisp_<T>(imc, sc, invalidvalue);
		//Mat mask=Mat::zeros(im.size(),CV_8U);
		//#pragma omp parallel for
		for (int j = 0; j < src.rows - 1; j++)
		{
			T* dref = sc.ptr<T>(j);
			uchar* imref = imc.ptr<uchar>(j);
			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			//uchar* m = mask.ptr<uchar>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = 255;//可能性のある最大値を入力
			s[src.cols - 1] = 255;//可能性のある最大値を入力
			//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
			for (int i = 0; i < src.cols; i++)
			{
				if (s[i] <= invalidvalue)
				{
					int t = i;
					int count = 0;
					do
					{
						count++;
						t++;
						if (count > maxlength)break;
						if (t > src.cols - 2)break;
					} while (s[t] <= invalidvalue);
					if (count > maxlength)break;
					//外枠は例外
					if (t == src.cols - 1)
					{
						memcpy(ims + 3 * i, imref + 3 * i, 3 * (src.cols - 1 - i));
						memcpy(s + sizeof(T)*i, dref + sizeof(T)*i, sizeof(T)*(src.cols - 1 - i));
						continue;
					}
					if (i == 1)
					{
						memcpy(ims, imref, 3 * t);
						memcpy(s, dref, sizeof(T)*t);
						i = t;
						continue;
					}

					T dd;

					if (s[i - 1] < s[t])
					{
						dd = s[i - 1];

						int count = 2 * (t - i);
						int p = i - (t - i);
						for (int k = 0; k < count; k += 2)
						{
							s[p + k] = dd;
							//m[p+k]=255;
							int f = (k >> 1);
							ims[3 * (p + k) + 0] = imref[3 * (p + f) + 0];
							ims[3 * (p + k) + 1] = imref[3 * (p + f) + 1];
							ims[3 * (p + k) + 2] = imref[3 * (p + f) + 2];

							/*ims[3*(p+k)+0]=0;
							ims[3*(p+k)+1]=255;
							ims[3*(p+k)+2]=0;*/
						}
						for (int k = 1; k < count; k += 2)
						{
							s[p + k] = dd;
							//	m[p+k]=255;
							int f = (k >> 1);
							ims[3 * (p + k) + 0] = (imref[3 * (p + f) + 0] + imref[3 * (p + f + 1) + 0]) >> 1;
							ims[3 * (p + k) + 1] = (imref[3 * (p + f) + 1] + imref[3 * (p + f + 1) + 1]) >> 1;
							ims[3 * (p + k) + 2] = (imref[3 * (p + f) + 2] + imref[3 * (p + f + 1) + 2]) >> 1;

							/*ims[3*(p+k)+0]=0;
							ims[3*(p+k)+1]=255;
							ims[3*(p+k)+2]=0;*/
						}

						i = t;
					}
					else
					{
						dd = s[t];
						int p = i;
						int count = (t - i - 1) * 2;
						i += (t - i);
						for (int k = 0; k < count; k += 2)
						{
							s[p + k] = dd;
							//	m[p+k]=255;
							int f = (k >> 1);
							ims[3 * (p + k) + 0] = imref[3 * (t + f) + 0];
							ims[3 * (p + k) + 1] = imref[3 * (t + f) + 1];
							ims[3 * (p + k) + 2] = imref[3 * (t + f) + 2];
							//ims[3*(p+k)+0]=0;
							//ims[3*(p+k)+1]=0;
							//ims[3*(p+k)+2]=255;
						}
						for (int k = 1; k < count; k += 2)
						{
							s[p + k] = dd;
							//	m[p+k]=255;
							int f = (k >> 1);
							ims[3 * (p + k) + 0] = (imref[3 * (t + f) + 0] + imref[3 * (t + f + 1) + 0]) >> 1;
							ims[3 * (p + k) + 1] = (imref[3 * (t + f) + 1] + imref[3 * (t + f + 1) + 1]) >> 1;
							ims[3 * (p + k) + 2] = (imref[3 * (t + f) + 2] + imref[3 * (t + f + 1) + 2]) >> 1;
							//ims[3*(p+k)+0]=0;
							//ims[3*(p+k)+1]=0;
							//ims[3*(p+k)+2]=255;
						}
					}
				}
			}
			s[0] = st;//もとに戻す
			if (st <= invalidvalue)
			{
				s[0] = s[1];
				ims[0] = ims[3];
				ims[1] = ims[4];
				ims[2] = ims[5];
			}
			s[src.cols - 1] = ed;
			if (ed <= invalidvalue)
			{
				s[src.cols - 1] = s[src.cols - 2];
				ims[3 * src.cols - 3] = ims[3 * src.cols - 6];
				ims[3 * src.cols - 2] = ims[3 * src.cols - 5];
				ims[3 * src.cols - 1] = ims[3 * src.cols - 4];
			}
		}
	}
	template <class T>
	void fillOcclusionImDispBlur_(Mat& im, Mat& src, T invalidvalue)
	{
		int bb = 1;
		//#pragma omp parallel for
		for (int j = bb; j < src.rows - bb; j++)
		{
			uchar* ims = im.ptr<uchar>(j);
			T* s = src.ptr<T>(j);
			const T st = s[0];
			const T ed = s[src.cols - 1];

			s[0] = 255;//可能性のある最大値を入力
			s[src.cols - 1] = 255;//可能性のある最大値を入力
			//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
			for (int i = 0; i < src.cols; i++)
			{
				if (s[i] <= invalidvalue)
				{
					uchar cs[3];
					cs[0] = ims[3 * i - 3];
					cs[1] = ims[3 * i - 2];
					cs[2] = ims[3 * i - 1];
					int t = i;
					do
					{
						t++;
						if (t > src.cols - 1)break;
					} while (s[t] <= invalidvalue);
					uchar ce[3];
					ce[0] = ims[3 * t + 0];
					ce[1] = ims[3 * t + 1];
					ce[2] = ims[3 * t + 2];

					T dd;
					if (s[i - 1] < s[t])
					{
						dd = s[i - 1];
						ce[0] = cs[0];
						ce[1] = cs[1];
						ce[2] = cs[2];
					}
					else
					{
						dd = s[t];
					}

					if (t - i > src.cols - 3)
					{
						for (int n = 0; n < src.cols; n++)
						{
							s[i] = invalidvalue;
							ims[3 * i + 0] = ce[0];
							ims[3 * i + 1] = ce[1];
							ims[3 * i + 2] = ce[2];
						}
					}
					else
					{
						for (; i < t; i++)
						{
							s[i] = dd;
							ims[3 * i + 0] = ce[0];
							ims[3 * i + 1] = ce[1];
							ims[3 * i + 2] = ce[2];
						}
					}

				}
			}
			s[0] = st;//もとに戻す
			s[src.cols - 1] = ed;
		}
	}

	template <class T>
	void fillOcclusionImDispLRMax_(Mat& im, Mat& disp, Mat& lref, Mat& rref, T invalidvalue, double alpha, double amp, int mode)
	{
		double da = alpha / amp;
		double ida = (1.0 - alpha) / amp;
		Mat dispm;

		maxFilter(disp, dispm, Size(3, 3));
		for (int j = 1; j < im.rows - 1; j++)
		{
			uchar* img = im.ptr<uchar>(j);
			uchar* imgl = lref.ptr<uchar>(j);
			uchar* imgr = rref.ptr<uchar>(j);

			T* d = disp.ptr<T>(j);
			T* dm = dispm.ptr<T>(j);
			for (int i = 0; i < im.cols; i++)
			{
				if (d[i] == invalidvalue && dm[i] != invalidvalue)
				{
					int dl = +3 * cvRound(dm[i] * da);
					int dr = -3 * cvRound(dm[i] * ida);
					d[i] = dm[i];
					img[0] = (imgl[dl + 0] + imgr[dr + 0]) >> 1;
					img[1] = (imgl[dl + 1] + imgr[dr + 1]) >> 1;
					img[2] = (imgl[dl + 2] + imgr[dr + 2]) >> 1;
					/*img[0] = (imgl[dl+0]);
					img[1] = (imgl[dl+1]);
					img[2] = (imgl[dl+2]);

					img[0] = (imgr[dr+0]);
					img[1] = (imgr[dr+1]);
					img[2] = (imgr[dr+2]);*/

				}
				img += 3; imgl += 3; imgr += 3;
			}
		}
		//fillOcclusionImDisp(im, disp, invalidvalue, mode);
	}


	void fillOcclusionImDispLRMax(Mat& im, Mat& disp, Mat& lref, Mat& rref, double invalidvalue, double alpha, double amp, int mode)
	{
		if (disp.depth() == CV_8U)
		{
			fillOcclusionImDispLRMax_<uchar>(im, disp, lref, rref, (uchar)invalidvalue, alpha, amp, mode);
		}
		else if (disp.depth() == CV_16S)
		{
			fillOcclusionImDispLRMax_<short>(im, disp, lref, rref, (short)invalidvalue, alpha, amp, mode);
		}
		else if (disp.depth() == CV_16U)
		{
			fillOcclusionImDispLRMax_<ushort>(im, disp, lref, rref, (ushort)invalidvalue, alpha, amp, mode);
		}
		else if (disp.depth() == CV_32F)
		{
			fillOcclusionImDispLRMax_<float>(im, disp, lref, rref, (float)invalidvalue, alpha, amp, mode);
		}
	}

	void fillOcclusionImDisp(InputOutputArray im_, InputOutputArray disp_, int invalidvalue, int mode)
	{
		Mat im = im_.getMat();
		Mat disp = disp_.getMat();
		if (mode == FILL_OCCLUSION_LINE)
		{
			if (disp.depth() == CV_8U)
			{
				fillOcclusionImDisp_<uchar>(im, disp, (uchar)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_16S)
			{
				fillOcclusionImDisp_<short>(im, disp, (short)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_16U)
			{
				fillOcclusionImDisp_<unsigned short>(im, disp, (unsigned short)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_32S)
			{
				fillOcclusionImDisp_<int>(im, disp, (short)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_32F)
			{
				fillOcclusionImDisp_<float>(im, disp, (float)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_64F)
			{
				fillOcclusionImDisp_<double>(im, disp, (double)invalidvalue, 10000);
			}
		}
		else if (mode == FILL_OCCLUSION_REFLECT)
		{
			//reflect interpolation
			if (disp.depth() == CV_8U)
			{
				fillOcclusionImDispReflect_<uchar>(im, disp, (uchar)invalidvalue);
			}
			else if (disp.depth() == CV_16S)
			{
				fillOcclusionImDispReflect_<short>(im, disp, (short)invalidvalue);
			}
			else if (disp.depth() == CV_16U)
			{
				fillOcclusionImDispReflect_<unsigned short>(im, disp, (unsigned short)invalidvalue);
			}
			else if (disp.depth() == CV_32S)
			{
				fillOcclusionImDispReflect_<int>(im, disp, (int)invalidvalue);
			}
			else if (disp.depth() == CV_32F)
			{
				fillOcclusionImDispReflect_<float>(im, disp, (float)invalidvalue);
			}
			else if (disp.depth() == CV_64F)
			{
				fillOcclusionImDispReflect_<double>(im, disp, (double)invalidvalue);
			}
		}
		else if (mode == FILL_OCCLUSION_STRETCH)
		{
			//stretch interpolation
			if (disp.depth() == CV_8U)
			{
				fillOcclusionImDispStretch_<uchar>(im, disp, (uchar)invalidvalue);
			}
			else if (disp.depth() == CV_16S)
			{
				fillOcclusionImDispStretch_<short>(im, disp, (short)invalidvalue);
			}
			else if (disp.depth() == CV_16U)
			{
				fillOcclusionImDispStretch_<unsigned short>(im, disp, (unsigned short)invalidvalue);
			}
			else if (disp.depth() == CV_32S)
			{
				fillOcclusionImDispStretch_<int>(im, disp, (int)invalidvalue);
			}
			else if (disp.depth() == CV_32F)
			{
				fillOcclusionImDispStretch_<float>(im, disp, (float)invalidvalue);
			}
			else if (disp.depth() == CV_64F)
			{
				fillOcclusionImDispStretch_<double>(im, disp, (double)invalidvalue);
			}
		}
		//else if(mode==2)
		//{
		//	//stretch interpolation
		//	if(disp.depth()==CV_8U)
		//	{
		//		fillOcclusionImDisp2_<uchar>(im,disp, (uchar)invalidvalue);
		//	}
		//	else if(disp.depth()==CV_16S)
		//	{
		//		fillOcclusionImDisp2_<short>(im,disp, (short)invalidvalue);
		//	}
		//	else if(disp.depth()==CV_16U)
		//	{
		//		fillOcclusionImDisp2_<unsigned short>(im,disp, (short)invalidvalue);
		//	}
		//	else if(disp.depth()==CV_32F)
		//	{
		//		fillOcclusionImDisp2_<float>(im,disp, (float)invalidvalue);
		//	}
		//}
		else if (mode == FILL_OCCLUSION_HV)//FILL_OCCLUSION_HV
		{
			if (disp.depth() == CV_8U)
			{
				fillOcclusionImDispHV_<uchar>(im, disp, (uchar)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_16S)
			{
				fillOcclusionImDispHV_<short>(im, disp, (short)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_16U)
			{
				fillOcclusionImDispHV_<unsigned short>(im, disp, (short)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_32S)
			{
				fillOcclusionImDispHV_<int>(im, disp, (int)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_32F)
			{
				fillOcclusionImDispHV_<float>(im, disp, (float)invalidvalue, 10000);
			}
			else if (disp.depth() == CV_64F)
			{
				fillOcclusionImDispHV_<double>(im, disp, (int)invalidvalue, 10000);
			}
		}
	}

	template <class T>
	void setRectficatedInvalidMask_(Mat& disp, Mat& image, T invalidvalue)
	{
		//#pragma omp parallel for
		for (int j = 0; j < disp.rows; j++)
		{
			uchar* im = image.ptr<uchar>(j);
			T* d = disp.ptr<T>(j);
			for (int i = 0; i < disp.cols; i++)
			{
				if (im[i] == 0)
					d[i] = invalidvalue;
			}
		}
	}
	void setRectficatedInvalidMask(Mat& disp, Mat& image, int invalidvalue)
	{
		Mat im;
		if (image.channels() != 1)cvtColor(image, im, COLOR_BGR2GRAY);
		else im = image;


		if (disp.depth() == CV_8U)
		{
			setRectficatedInvalidMask_<uchar>(disp, im, (uchar)invalidvalue);
		}
		else if (disp.depth() == CV_16S)
		{
			setRectficatedInvalidMask_<short>(disp, im, (short)invalidvalue);

		}
		else if (disp.depth() == CV_32F)
		{
			setRectficatedInvalidMask_<float>(disp, im, (float)invalidvalue);
		}
	}

	template <class T>
	void blendLR2(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, Mat& maskL, Mat& maskR, double a)
	{
		int dth = 5;
		a = a > 1.0 ? 1.0 : a;
		a = a < 0.0 ? 0.0 : a;

		/*
		a =  a<0.1 ? a=0:a;
		a =  a>0.9 ? a=1:a;*/
		/*
		double aa=a;
		if(aa<0.5)a=2*a*a;
		else a = -2.0*(a-1)*(a-1)+1.0;*/

		double ia = 1.0 - a;
		//#pragma omp parallel for
		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			uchar* ml = maskL.ptr<uchar>(j);
			uchar* mr = maskR.ptr<uchar>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (ml[i] == 255 && mr[i] == 255)
				{
					if (abs(dl[i] - dr[i]) < dth)
					{
						dd[i] = (T)((dl[i] + dr[i])*0.5);
						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0]);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1]);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2]);
					}
					else if (dl[i] > dr[i])
					{
						dd[i] = dl[i];
						d[3 * i + 0] = l[3 * i + 0];
						d[3 * i + 1] = l[3 * i + 1];
						d[3 * i + 2] = l[3 * i + 2];
					}
					else
					{
						dd[i] = dr[i];
						d[3 * i + 0] = r[3 * i + 0];
						d[3 * i + 1] = r[3 * i + 1];
						d[3 * i + 2] = r[3 * i + 2];
					}

				}
				else if (ml[i] == 255)
				{
					//d[3*i+0] = 0.75*l[3*i+0]+0.25*r[3*i+0];
					//d[3*i+1] = 0.75*l[3*i+1]+0.25*r[3*i+1];
					//d[3*i+2] = 0.75*l[3*i+2]+0.25*r[3*i+2];

					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (mr[i] == 255)
				{
					//d[3*i+0] = 0.25*l[3*i+0]+0.75*r[3*i+0];
					//d[3*i+1] = 0.25*l[3*i+1]+0.75*r[3*i+1];
					//d[3*i+2] = 0.25*l[3*i+2]+0.75*r[3*i+2];
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}


	// simplest without mask
	template <class T>
	void blendLRS(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double a, T invalid)
	{
		a = max(0.0, min(a, 1.0));

		double ia = 1.0 - a;

		//	iml.copyTo(dest);
		//	dispL.copyTo(destdisp);

		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (dl[i] != invalid && dr[i] != invalid)
				{
					dd[i] = (T)((dl[i] + dr[i])*0.5);
					d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5f);
					d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5f);
					d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5f);
				}
				else if (dl[i] != invalid)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (dr[i] != invalid)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}


	void blendLRS_8u_leftisout(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, double a, uchar invalid)
	{
		a = max(0.0, min(a, 1.0));
		double ia = 1.0 - a;


		const int shift = 10;
		int base = 1 << shift;
		int A = (int)(base * a);
		int IA = base - A;

		for (int j = 0; j < iml.rows; j++)
		{
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			uchar* dl = dispL.ptr<uchar>(j);
			uchar* dr = dispR.ptr<uchar>(j);
			int i = 0;
			for (; i < iml.cols; i++)
			{
				if (dl[i] != invalid && dr[i] != invalid)
				{
					dl[i] = ((dl[i] + dr[i]) >> 1);
					/*l[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]);
					l[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]);
					l[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]);*/
					/*l[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]+0.5);
					l[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]+0.5);
					l[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]+0.5);*/
					l[3 * i + 0] = saturate_cast<uchar>((IA*l[3 * i + 0] + A*r[3 * i + 0]) >> shift);
					l[3 * i + 1] = saturate_cast<uchar>((IA*l[3 * i + 1] + A*r[3 * i + 1]) >> shift);
					l[3 * i + 2] = saturate_cast<uchar>((IA*l[3 * i + 2] + A*r[3 * i + 2]) >> shift);
				}
				else if (dr[i] != invalid)
				{
					dl[i] = dr[i];
					l[3 * i + 0] = r[3 * i + 0];
					l[3 * i + 1] = r[3 * i + 1];
					l[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}

	// simplest
	template <class T>
	void blendLRS(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, Mat& maskL, Mat& maskR, double a)
	{
		a = max(0.0, min(a, 1.0));

		double ia = 1.0 - a;

		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			uchar* ml = maskL.ptr<uchar>(j);
			uchar* mr = maskR.ptr<uchar>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (ml[i] == 255 && mr[i] == 255)
				{
					dd[i] = (T)((dl[i] + dr[i])*0.5);
					d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
					d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
					d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
				}
				else if (ml[i] == 255)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (mr[i] == 255)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}



	//without mask
	template <class T>
	void blendLR_NearestMax(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double a, T dth, T invalid)
	{
		a = max(0.0, min(a, 1.0));

		Mat maxL, maxR;
		maxFilter(dispL, maxL, 1);
		maxFilter(dispR, maxR, 1);
		double ia = 1.0 - a;
		//#pragma omp parallel for
		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			T* mdl = maxL.ptr<T>(j);
			T* mdr = maxR.ptr<T>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (dl[i] != invalid &&dr[i] != invalid)
				{
					if (abs(dr[i] - dl[i]) <= dth)
					{
						dd[i] = (T)((dl[i] + dr[i])*0.5);

						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
					}
					else if (abs(mdr[i] - dl[i]) <= dth)
					{
						dd[i] = (T)((dl[i] + mdr[i])*0.5);

						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
					}
					else  if (abs(dr[i] - mdl[i]) <= dth)
					{
						dd[i] = (T)((mdl[i] + dr[i])*0.5);

						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
					}

					else if (dl[i] - dr[i] > dth)
					{
						dd[i] = dl[i];
						d[3 * i + 0] = l[3 * i + 0];
						d[3 * i + 1] = l[3 * i + 1];
						d[3 * i + 2] = l[3 * i + 2];
					}
					else if (dr[i] - dl[i] > dth)
					{
						dd[i] = dr[i];
						d[3 * i + 0] = r[3 * i + 0];
						d[3 * i + 1] = r[3 * i + 1];
						d[3 * i + 2] = r[3 * i + 2];
					}
				}
				else if (dl[i] != invalid)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (dr[i] != invalid)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}

	//without mask
	template <class T>
	void blendLR(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double a, T dth, T invalid)
	{
		a = max(0.0, min(a, 1.0));

		double ia = 1.0 - a;
		//#pragma omp parallel for
		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);


			for (int i = 0; i<iml.cols; i++)
			{
				if (dl[i] != invalid &&dr[i] != invalid)
				{
					if (dl[i] - dr[i]>dth)
					{
						dd[i] = dl[i];
						d[3 * i + 0] = l[3 * i + 0];
						d[3 * i + 1] = l[3 * i + 1];
						d[3 * i + 2] = l[3 * i + 2];
					}
					else if (dr[i] - dl[i] > dth)
					{
						dd[i] = dr[i];
						d[3 * i + 0] = r[3 * i + 0];
						d[3 * i + 1] = r[3 * i + 1];
						d[3 * i + 2] = r[3 * i + 2];
					}
					else
					{
						dd[i] = (T)((dl[i] + dr[i])*0.5);
						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
					}
				}
				else if (dl[i] != invalid)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (dr[i] != invalid)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
	}


	template <class T>
	void blendLR(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, Mat& maskL, Mat& maskR, double a, T dth)
	{
		a = max(0.0, min(a, 1.0));

		double ia = 1.0 - a;
		//#pragma omp parallel for
		for (int j = 0; j < iml.rows; j++)
		{
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			uchar* ml = maskL.ptr<uchar>(j);
			uchar* mr = maskR.ptr<uchar>(j);

			for (int i = 0; i<iml.cols; i++)
			{
				if (ml[i] == 255 && mr[i] == 255)
				{
					if (dl[i] - dr[i]>dth)
					{
						dd[i] = dl[i];
						mr[i] = 0;
						d[3 * i + 0] = l[3 * i + 0];
						d[3 * i + 1] = l[3 * i + 1];
						d[3 * i + 2] = l[3 * i + 2];
					}
					else if (dr[i] - dl[i] > dth)
					{
						dd[i] = dr[i];
						ml[i] = 0;
						d[3 * i + 0] = r[3 * i + 0];
						d[3 * i + 1] = r[3 * i + 1];
						d[3 * i + 2] = r[3 * i + 2];
					}
					else
					{
						dd[i] = (T)((dl[i] + dr[i])*0.5);
						d[3 * i + 0] = saturate_cast<uchar>(ia*l[3 * i + 0] + a*r[3 * i + 0] + 0.5);
						d[3 * i + 1] = saturate_cast<uchar>(ia*l[3 * i + 1] + a*r[3 * i + 1] + 0.5);
						d[3 * i + 2] = saturate_cast<uchar>(ia*l[3 * i + 2] + a*r[3 * i + 2] + 0.5);
					}
				}
				else if (ml[i] == 255)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (mr[i] == 255)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}

		//boundary post filter
		/*Mat mk =Mat::zeros(maskL.size(),CV_8U);
		for(int j=0;j<iml.rows;j++)
		{
		uchar* d=dest.ptr<uchar>(j);
		uchar* l=iml.ptr<uchar>(j);
		uchar* r=imr.ptr<uchar>(j);

		T* dd=destdisp.ptr<T>(j);
		T* dl=dispL.ptr<T>(j);
		T* dr=dispR.ptr<T>(j);

		uchar* ml=maskL.ptr<uchar>(j);
		uchar* mr=maskR.ptr<uchar>(j);

		for(int i=0;i<iml.cols;i++)
		{

		if((ml[i]==0 && mr[i]==255) && (ml[i+1]==255 && mr[i+1]==255))
		{
		mk.at<uchar>(j,i)=255;
		mk.at<uchar>(j,i+1)=255;
		uchar r = (d[3*i+0]+d[3*i+3])>>1;
		uchar g = (d[3*i+1]+d[3*i+4])>>1;
		uchar b = (d[3*i+2]+d[3*i+5])>>1;
		d[3*i+0] =(d[3*i+0]+r)>>1;
		d[3*i+1] =(d[3*i+1]+g)>>1;
		d[3*i+2] =(d[3*i+2]+b)>>1;
		d[3*i+3] =(d[3*i+3]+r)>>1;
		d[3*i+4] =(d[3*i+4]+g)>>1;
		d[3*i+5] =(d[3*i+5]+b)>>1;
		}
		if((ml[i]==0 && mr[i]==255) && (ml[i-1]==255 && mr[i-1]==255))
		{
		mk.at<uchar>(j,i)=255;
		mk.at<uchar>(j,i-1)=255;

		uchar r = (d[3*i+0]+d[3*i-3])>>1;
		uchar g = (d[3*i+1]+d[3*i-2])>>1;
		uchar b = (d[3*i+2]+d[3*i-1])>>1;
		d[3*i+0] =(d[3*i+0]+r)>>1;
		d[3*i+1] =(d[3*i+1]+g)>>1;
		d[3*i+2] =(d[3*i+2]+b)>>1;
		d[3*i-3] =(d[3*i-3]+r)>>1;
		d[3*i-2] =(d[3*i-2]+g)>>1;
		d[3*i-1] =(d[3*i-1]+b)>>1;
		}

		if((ml[i]==255 && mr[i]==0) && (ml[i+1]==255 && mr[i+1]==255))
		{
		mk.at<uchar>(j,i)=255;
		mk.at<uchar>(j,i+1)=255;
		uchar r = (d[3*i+0]+d[3*i+3])>>1;
		uchar g = (d[3*i+1]+d[3*i+4])>>1;
		uchar b = (d[3*i+2]+d[3*i+5])>>1;
		d[3*i+0] =(d[3*i+0]+r)>>1;
		d[3*i+1] =(d[3*i+1]+g)>>1;
		d[3*i+2] =(d[3*i+2]+b)>>1;
		d[3*i+3] =(d[3*i+3]+r)>>1;
		d[3*i+4] =(d[3*i+4]+g)>>1;
		d[3*i+5] =(d[3*i+5]+b)>>1;

		}
		if((ml[i]==255 && mr[i]==0) && (ml[i-1]==255 && mr[i-1]==255))
		{
		mk.at<uchar>(j,i)=255;
		mk.at<uchar>(j,i-1)=255;

		uchar r = (d[3*i+0]+d[3*i-3])>>1;
		uchar g = (d[3*i+1]+d[3*i-2])>>1;
		uchar b = (d[3*i+2]+d[3*i-1])>>1;
		d[3*i+0] =(d[3*i+0]+r)>>1;
		d[3*i+1] =(d[3*i+1]+g)>>1;
		d[3*i+2] =(d[3*i+2]+b)>>1;
		d[3*i-3] =(d[3*i-3]+r)>>1;
		d[3*i-2] =(d[3*i-2]+g)>>1;
		d[3*i-1] =(d[3*i-1]+b)>>1;
		}
		}
		}*/
	}


	void warpShiftSubpix_cubic(Mat& src, Mat& dest, double a)
	{
		const double cubic = -1.0;
		const double c1 = cubic;
		const double c2 = -5.0*cubic;
		const double c3 = 8.0*cubic;
		const double c4 = -4.0*cubic;
		const double c5 = 2.0 + cubic;
		const double c6 = -(cubic + 3.0);

		dest.create(src.size(), src.type());

		if (a > 0)
		{
			a = 1.0 - a;
			const double ia = 1.0 - a;
			const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
			const double iaa = c5* a* a* a + c6* a* a + 1.0;
			const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
			const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;

			Mat im; cv::copyMakeBorder(src, im, 0, 0, 1, 2, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < src.cols; i++)
				{
					d[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * i - 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i + 3] + viaa*sim[3 * i + 6]);
					d[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * i - 2] + aa*sim[3 * i + 1] + iaa*sim[3 * i + 4] + viaa*sim[3 * i + 7]);
					d[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * i - 1] + aa*sim[3 * i + 2] + iaa*sim[3 * i + 5] + viaa*sim[3 * i + 8]);
				}
			}
		}
		else if (a < 0)
		{
			a = 1.0 + a;
			const double ia = 1.0 - a;
			const double viaa = c1*(1.0 + a)*(1.0 + a)*(1.0 + a) + c2*(1.0 + a)*(1.0 + a) + c3*(1.0 + a) + c4;
			const double iaa = c5* a* a* a + c6* a* a + 1.0;
			const double aa = c5*ia*ia*ia + c6*ia*ia + 1.0;
			const double vaa = c1*(1.0 + ia)*(1.0 + ia)*(1.0 + ia) + c2*(1.0 + ia)*(1.0 + ia) + c3*(1.0 + ia) + c4;

			Mat im; cv::copyMakeBorder(src, im, 0, 0, 2, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 6;
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < src.cols; i++)
				{
					d[3 * (i)+0] = saturate_cast<uchar>(vaa*sim[3 * i + 3] + aa*sim[3 * i + 0] + iaa*sim[3 * i - 3] + viaa*sim[3 * i - 6]);
					d[3 * (i)+1] = saturate_cast<uchar>(vaa*sim[3 * i + 4] + aa*sim[3 * i + 1] + iaa*sim[3 * i - 2] + viaa*sim[3 * i - 5]);
					d[3 * (i)+2] = saturate_cast<uchar>(vaa*sim[3 * i + 5] + aa*sim[3 * i + 2] + iaa*sim[3 * i - 1] + viaa*sim[3 * i - 4]);
				}
			}
		}
		else
		{
			src.copyTo(dest);
		}
	}
	void warpShiftSubpix_linear(Mat& src, Mat& dest, double a)
	{
		dest.create(src.size(), src.type());

		if (a > 0)
		{
			a = 1.0 - a;
			const double ia = 1.0 - a;
			Mat im; cv::copyMakeBorder(src, im, 0, 0, 0, 1, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j);
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < src.cols; i++)
				{
					d[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * i + 0] + ia*sim[3 * i + 3]);
					d[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * i + 1] + ia*sim[3 * i + 4]);
					d[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * i + 2] + ia*sim[3 * i + 5]);
				}
			}
		}
		else if (a < 0)
		{
			a = 1.0 + a;
			const double ia = 1.0 - a;
			Mat im; cv::copyMakeBorder(src, im, 0, 0, 1, 0, BORDER_REPLICATE);
			//#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				uchar* sim = im.ptr<uchar>(j); sim += 3;
				uchar* d = dest.ptr<uchar>(j);

				for (int i = 0; i < src.cols; i++)
				{
					d[3 * (i)+0] = saturate_cast<uchar>(a*sim[3 * i + 0] + ia*sim[3 * i - 3]);
					d[3 * (i)+1] = saturate_cast<uchar>(a*sim[3 * i + 1] + ia*sim[3 * i - 2]);
					d[3 * (i)+2] = saturate_cast<uchar>(a*sim[3 * i + 2] + ia*sim[3 * i - 1]);
				}
			}
		}
		else
		{
			src.copyTo(dest);
		}
	}

	template <class T>
	void blendLRRes(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, Mat& maskL, Mat& maskR, double a, int dth = 32)
	{
		Mat resmap = Mat::zeros(iml.size(), CV_8U);
		vector<Mat> lsub(21);
		vector<Mat> rsub(21);
		vector<uchar*> ls(21);
		vector<uchar*> rs(21);
		{
			Mat lref = iml.clone();
			Mat ldref = dispL.clone();
			Mat rref = imr.clone();
			Mat rdref = dispR.clone();
			fillOcclusionImDisp(lref, ldref);
			fillOcclusionImDisp(rref, rdref);
			for (int i = -10; i <= 10; i++)
			{
				double move = i*0.05;
				warpShiftSubpix_cubic(lref, lsub[i + 10], move);
				warpShiftSubpix_cubic(rref, rsub[i + 10], -move);
				//gui(lsub[i+5],rsub[i+5]);
				/*imshow("l",lsub[i+5]);
				imshow("r",rsub[i+5]);
				waitKey();*/
			}
		}
		a = max(0.0, min(a, 1.0));


		double ia = 1.0 - a;
		//#pragma omp parallel for

		for (int j = 0; j < iml.rows; j++)
		{
			for (int n = 0; n < 11; n++)
			{
				ls[n] = lsub[n].ptr<uchar>(j);
				rs[n] = rsub[n].ptr<uchar>(j);
			}
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			uchar* ml = maskL.ptr<uchar>(j);
			uchar* mr = maskR.ptr<uchar>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (ml[i] == 255 && mr[i] == 255)
				{
					dd[i] = (T)((dl[i] + dr[i])*0.5);
					int emax = INT_MAX;
					int argn = 0;
					for (int n = 0; n < 11; n++)
					{
						int e = (int)abs(ls[n][3 * i + 0] - rs[n][3 * i + 0]) + 3 * (int)abs(ls[n][3 * i + 1] - rs[n][3 * i + 1]) + (int)abs(ls[n][3 * i + 2] - rs[n][3 * i + 2]);
						e += (int)abs(ls[n][3 * i + 3] - rs[n][3 * i + 3]) + (int)abs(ls[n][3 * i + 4] - rs[n][3 * i + 4]) + (int)abs(ls[n][3 * i + 5] - rs[n][3 * i + 5]);
						e += (int)abs(ls[n][3 * i - 1] - rs[n][3 * i - 1]) + (int)abs(ls[n][3 * i - 2] - rs[n][3 * i - 2]) + (int)abs(ls[n][3 * i - 3] - rs[n][3 * i - 3]);
						if (e < emax)
						{
							emax = e;
							argn = n;
						}
					}
					resmap.at<uchar>(j, i) = argn * 10;
					//argn=5;
					d[3 * i + 0] = saturate_cast<uchar>(ia*ls[argn][3 * i + 0] + a*rs[argn][3 * i + 0]);
					d[3 * i + 1] = saturate_cast<uchar>(ia*ls[argn][3 * i + 1] + a*rs[argn][3 * i + 1]);
					d[3 * i + 2] = saturate_cast<uchar>(ia*ls[argn][3 * i + 2] + a*rs[argn][3 * i + 2]);

					/*d[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]);
					d[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]);
					d[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]);*/
				}
				else if (ml[i] == 255)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (mr[i] == 255)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
		imshow("resmap", resmap);
	}

	template <class T>
	void blendLRRes2(Mat& iml, Mat& imr, Mat& imlP, Mat& imrP, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, Mat& maskL, Mat& maskR, double a, int dth = 32)
	{
		Mat resmap = Mat::zeros(iml.size(), CV_8U);
		vector<Mat> lsub(21);
		vector<Mat> rsub(21);
		vector<uchar*> ls(21);
		vector<uchar*> rs(21);
		const int wstep = 3 * iml.cols;
		{
			Mat il;
			Mat ir;
			copyMakeBorder(imrP, ir, 1, 1, 0, 0, BORDER_REPLICATE);
			copyMakeBorder(imlP, il, 1, 1, 0, 0, BORDER_REPLICATE);
			for (int i = -10; i <= 10; i++)
			{
				double move = i*0.05;

				warpShiftSubpix_cubic(il, lsub[i + 10], move);
				warpShiftSubpix_cubic(ir, rsub[i + 10], -move);
				//gui(lsub[i+5],rsub[i+5]);
				/*imshow("l",lsub[i+5]);
				imshow("r",rsub[i+5]);
				waitKey();*/
			}
		}
		a = max(0.0, min(a, 1.0));


		double ia = 1.0 - a;
		//#pragma omp parallel for

		for (int j = 0; j < iml.rows; j++)
		{
			for (int n = 0; n < 21; n++)
			{
				ls[n] = lsub[n].ptr<uchar>(j + 1);
				rs[n] = rsub[n].ptr<uchar>(j + 1);
			}
			uchar* d = dest.ptr<uchar>(j);
			uchar* l = iml.ptr<uchar>(j);
			uchar* r = imr.ptr<uchar>(j);

			T* dd = destdisp.ptr<T>(j);
			T* dl = dispL.ptr<T>(j);
			T* dr = dispR.ptr<T>(j);

			uchar* ml = maskL.ptr<uchar>(j);
			uchar* mr = maskR.ptr<uchar>(j);

			for (int i = 0; i < iml.cols; i++)
			{
				if (ml[i] == 255 && mr[i] == 255)
				{
					dd[i] = (T)((dl[i] + dr[i])*0.5);

					//cout<<dl[i]<<","<<dr[i]<<endl;
					int disp = dd[i] / 16.0;
					int emax = INT_MAX;
					int argn = 0;
					for (int n = 0; n < 21; n++)
					{
						int e = INT_MAX;
						if (i + disp<iml.cols&& i>disp)
						{

							e = (int)abs(ls[n][3 * (i + disp) + 0] - rs[n][3 * (i - disp) + 0]) + 3 * (int)abs(ls[n][3 * (i + disp) + 1] - rs[n][3 * (i - disp) + 1]) + (int)abs(ls[n][3 * (i + disp) + 2] - rs[n][3 * (i - disp) + 2]);
							e += (int)abs(ls[n][3 * (i + disp) + 3] - rs[n][3 * (i - disp) + 3]) + 3 * (int)abs(ls[n][3 * (i + disp) + 4] - rs[n][3 * (i - disp) + 4]) + (int)abs(ls[n][3 * (i + disp) + 2] - rs[n][3 * (i - disp) + 5]);
							e += (int)abs(ls[n][3 * (i + disp) - 3] - rs[n][3 * (i - disp) - 3]) + 3 * (int)abs(ls[n][3 * (i + disp) - 2] - rs[n][3 * (i - disp) - 2]) + (int)abs(ls[n][3 * (i + disp) + 2] - rs[n][3 * (i - disp) - 1]);
							e += (int)abs(ls[n][3 * (i + disp) + wstep + 3] - rs[n][3 * (i - disp) + wstep + 3]) + 3 * (int)abs(ls[n][3 * (i + disp) + wstep + 4] - rs[n][3 * (i - disp) + wstep + 4]) + (int)abs(ls[n][3 * (i + disp) + wstep + 2] - rs[n][3 * (i - disp) + wstep + 5]);
							e += (int)abs(ls[n][3 * (i + disp) - wstep - 3] - rs[n][3 * (i - disp) - wstep - 3]) + 3 * (int)abs(ls[n][3 * (i + disp) - wstep - 2] - rs[n][3 * (i - disp) - wstep - 2]) + (int)abs(ls[n][3 * (i + disp) - wstep + 2] - rs[n][3 * (i - disp) - wstep - 1]);
						}


						if (e < emax)
						{
							emax = e;
							argn = n;
						}
					}

					resmap.at<uchar>(j, i) = argn * 10;
					//cout<<"!"<<i<<","<<disp<<","<<argn<<endl;
					d[3 * i + 0] = saturate_cast<uchar>(ia*ls[argn][3 * (i + disp) + 0] + a*rs[argn][3 * (i - disp) + 0]);
					d[3 * i + 1] = saturate_cast<uchar>(ia*ls[argn][3 * (i + disp) + 1] + a*rs[argn][3 * (i - disp) + 1]);
					d[3 * i + 2] = saturate_cast<uchar>(ia*ls[argn][3 * (i + disp) + 2] + a*rs[argn][3 * (i - disp) + 2]);

					/*d[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]);
					d[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]);
					d[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]);*/
				}
				else if (ml[i] == 255)
				{
					dd[i] = dl[i];
					d[3 * i + 0] = l[3 * i + 0];
					d[3 * i + 1] = l[3 * i + 1];
					d[3 * i + 2] = l[3 * i + 2];
				}
				else if (mr[i] == 255)
				{
					dd[i] = dr[i];
					d[3 * i + 0] = r[3 * i + 0];
					d[3 * i + 1] = r[3 * i + 1];
					d[3 * i + 2] = r[3 * i + 2];
				}
			}
		}
		imshow("resmap", resmap);
	}
	template <class T>
	void fillBoundingBoxDepthIm(Mat& src_im, Mat& src_dp, int occflag)
	{
		//#pragma omp parallel for
		for (int j = 0; j < src_im.rows; j++)
		{
			uchar* sim = src_im.ptr<uchar>(j);
			T* sdp = src_dp.ptr<T>(j);

			int k = 0;
			int i = 0;

			while (sdp[i] == occflag)
			{
				i++;
				k++;
				if (i == src_im.cols)continue;
			}
			for (i = 0; i < k; i++)
			{
				sdp[i] = sdp[k];
				sim[3 * i + 0] = sim[3 * k + 0];
				sim[3 * i + 1] = sim[3 * k + 1];
				sim[3 * i + 2] = sim[3 * k + 2];
			}

			i = src_im.cols - 2;
			k = 0;

			while (sdp[i] == occflag)
			{
				i--;
				k++;
				if (i == -1)continue;
			}
			for (i = src_im.cols - k - 1; i < src_im.cols; i++)
			{
				sdp[i] = sdp[src_im.cols - k - 2];
				sim[3 * i + 0] = sim[3 * (src_im.cols - 2 - k) + 0];
				sim[3 * i + 1] = sim[3 * (src_im.cols - 2 - k) + 1];
				sim[3 * i + 2] = sim[3 * (src_im.cols - 2 - k) + 2];
			}
		}
	}
	template <class T>
	void fillBoundingBoxDepth(Mat& src_dp, int occflag)
	{
		//#pragma omp parallel for
		for (int j = 0; j < src_dp.rows; j++)
		{
			T* sdp = src_dp.ptr<T>(j);

			int k = 0;
			int i = 0;

			while (sdp[i] == occflag)
			{
				i++;
				k++;
				if (i == src_dp.cols)continue;
			}
			for (i = 0; i < k; i++)
			{
				sdp[i] = sdp[k];
			}

			i = src_dp.cols - 2;
			k = 0;

			while (sdp[i] == occflag)
			{
				i--;
				k++;
				if (i == -1)continue;
			}
			for (i = src_dp.cols - k - 1; i < src_dp.cols; i++)
			{
				sdp[i] = sdp[src_dp.cols - k - 2];
			}
		}
	}


	template <class T>
	void depthBasedInpaint(Mat& src_im, Mat& src_dp, Mat& dest_im, Mat& dest_dp, T OCC_FLAG)
	{
		const T DISPINF = 255;
		int bs = 20;
		//erode(src_dp,dest_dp,Mat());
		src_dp.copyTo(dest_dp);
		src_im.copyTo(dest_im);

		bool loop = true;
		for (int iter = 0; iter < 10; iter++)
		{
			//printf("iter %d\n",iter);
			if (loop == false)break;
			loop = false;
			dest_im.copyTo(src_im);
			dest_dp.copyTo(src_dp);

			//#pragma omp parallel for
			for (int j = bs; j < src_dp.rows - bs; j++)
			{
				uchar* dim = dest_im.ptr<uchar>(j);
				T* ddp = dest_dp.ptr<T>(j);
				T* sdp = src_dp.ptr<T>(j);

				for (int i = bs; i < src_dp.cols - bs; i++)
				{
					if (sdp[i] == OCC_FLAG)
					{
						loop = true;
						T dmin = DISPINF;
						for (int l = -bs; l <= bs; l++)
						{
							T* bddp = src_dp.ptr<T>(j + l);
							for (int k = -bs; k <= bs; k++)
							{
								if (bddp[k + i] != OCC_FLAG)
									dmin = min(dmin, bddp[k + i]);
							}
						}
						if (dmin == DISPINF)
						{
							continue;
						}

						int r = 0;
						int g = 0;
						int b = 0;
						int count = 0;
						for (int l = -bs; l <= bs; l++)
						{
							T* bddp = src_dp.ptr<T>(j + l);
							uchar* sim = src_im.ptr<uchar>(j + l);
							for (int k = -bs; k <= bs; k++)
							{
								if (bddp[k + i] < dmin + 2 && bddp[k + i] >= dmin)
								{
									r += sim[3 * (k + i) + 0];
									g += sim[3 * (k + i) + 1];
									b += sim[3 * (k + i) + 2];
									count++;
								}
							}
						}
						dim[3 * i] = r / count;
						dim[3 * i + 1] = g / count;
						dim[3 * i + 2] = b / count;
						ddp[i] = dmin;
					}

				}
			}
		}
	}

	template <class T>
	void shiftDisparity_(Mat& srcdisp, Mat& destdisp, double amp, const int large_jump, const int sub_gap)
	{
		const int ljump = max(large_jump, 1);
		if (amp > 0)
		{
			const int step = srcdisp.cols;
			T* s = srcdisp.ptr<T>(0);
			T* d = destdisp.ptr<T>(0);
			for (int j = 0; j<srcdisp.rows; j++)
			{
				for (int i = srcdisp.cols - 1; i >= 0; i--)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(disp*amp);

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1]) > ljump)
					{
						continue;
					}

					if (i - dest - 1 >= 0 && i - dest - 1 < srcdisp.cols - 1)
					{
						if (disp > d[i - dest])
						{
							d[i - dest] = disp;
							if (issub)
								d[i - dest + 1] = disp;
						}
					}
				}
				s += step;
				d += step;
			}
		}
		else if (amp < 0)
		{
			const int step = srcdisp.cols;
			T* s = srcdisp.ptr<T>(0);
			T* d = destdisp.ptr<T>(0);
			for (int j = 0; j<srcdisp.rows; j++)
			{
				for (int i = 0; i<srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(-disp*amp);

					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						continue;
					}

					if (i + dest + 1 >= 0 && i + dest + 1 < srcdisp.cols + 1)
					{
						if (disp > d[i + dest])
						{
							d[i + dest] = disp;
							if (issub)
								d[i + dest - 1] = disp;
						}
					}
				}
				s += step;
				d += step;
			}
		}
		else
		{
			srcdisp.copyTo(destdisp);
		}
	}

	void shiftDisparity(Mat& srcdisp, Mat& destdisp, double amp, const int large_jump, const int sub_gap)
	{
		if (srcdisp.depth() == CV_8U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_8U);
			else destdisp.setTo(0);
			shiftDisparity_<uchar>(srcdisp, destdisp, amp, large_jump, sub_gap);
			//fillBoundingBoxDepth<uchar>(destdisp,0);
			//fillOcclusion(destdisp,0);
		}
		else if (srcdisp.depth() == CV_16S)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16S);
			else destdisp.setTo(0);
			shiftDisparity_<short>(srcdisp, destdisp, amp, large_jump, sub_gap);
			//fillOcclusion(destdisp,0);
		}
		else if (srcdisp.depth() == CV_16U)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_16U);
			else destdisp.setTo(0);
			shiftDisparity_<ushort>(srcdisp, destdisp, amp, large_jump, sub_gap);
			//fillOcclusion(destdisp,0);
		}
		else if (srcdisp.depth() == CV_32F)
		{
			if (destdisp.empty())destdisp = Mat::zeros(srcdisp.size(), CV_32F);
			else destdisp.setTo(0);
			shiftDisparity_<float>(srcdisp, destdisp, amp, large_jump, sub_gap);
			fillBoundingBoxDepth<float>(destdisp, 0);
			fillOcclusion(destdisp, 0);
		}
	}


	template<class T>
	void crackRemove_(Mat& depth, Mat& depth_dest, T invalidvalue)
	{

		depth.copyTo(depth_dest);
		for (int j = 0; j < depth.rows; j++)
		{
			T* s = depth.ptr<T>(j);
			T* d = depth_dest.ptr<T>(j);

			for (int i = 1; i < depth.cols - 1; i++)
			{
				if (s[i] == invalidvalue)
				{
					if (s[i - 1] != invalidvalue && s[i + 1] != invalidvalue)
					{
						d[i] = (T)((s[i - 1] + s[i + 1])*0.5);
					}
				}
			}
		}
	}
	void crackRemove(Mat& depth, Mat& depth_dest, double invalidvalue)
	{
		if (depth.depth() == CV_8U)
			crackRemove_<uchar>(depth, depth_dest, (uchar)invalidvalue);
		else if (depth.depth() == CV_16S)
			crackRemove_<short>(depth, depth_dest, (short)invalidvalue);
		else if (depth.depth() == CV_16U)
			crackRemove_<ushort>(depth, depth_dest, (ushort)invalidvalue);
		else if (depth.depth() == CV_32F)
			crackRemove_<float>(depth, depth_dest, (float)invalidvalue);
	}

	void filterDepthSlant(Mat& depth, Mat& depth2, Mat& mask2, int kernelSize = 3)
	{
		depth.copyTo(depth2);
		fillOcclusion(depth2, 0);
		compare(depth, depth2, mask2, cv::CMP_NE);
	}

	void filterDepth2(Mat& depth, Mat& depth2, Mat& mask2, int kernelSize = 3)
	{
		//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
		//crackRemove(depth,depth2,mask2);
		//guiAlphaBlend(depth2,mask2);
		medianBlur(depth, depth2, kernelSize);
		medianBlur(depth2, depth2, kernelSize);

		Mat temp;
		depth2.convertTo(temp, CV_16SC1);
		filterSpeckles(temp, 0, 20, 5);
		temp.convertTo(depth2, CV_8U);

		//imshow("depth",depth2)
		/*medianBlur(depth2,depth2,kernelSize);
		medianBlur(depth2,depth2,kernelSize);
		medianBlur(depth2,depth2,kernelSize);*/
		compare(depth, depth2, mask2, cv::CMP_NE);
	}

	void filterDepth(Mat& depth, Mat& depth2, Mat& mask2, int kernelSize, int viewstep)
	{
		//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
		//crackRemove(depth,depth2,mask2);
		//guiAlphaBlend(depth2,mask2);
		medianBlur(depth, depth2, kernelSize);
		medianBlur(depth2, depth2, kernelSize);
		if (viewstep > 0)
		{
			maxFilter(depth2, depth2, Size(2 * viewstep + 1, 1));
			minFilter(depth2, depth2, Size(2 * viewstep + 1, 1));
		}

		//imshow("depth",depth2)
		/*medianBlur(depth2,depth2,kernelSize);
		medianBlur(depth2,depth2,kernelSize);
		medianBlur(depth2,depth2,kernelSize);*/
		compare(depth, depth2, mask2, cv::CMP_NE);
	}


	void StereoViewSynthesis::depthfilter(Mat& depth, Mat& depth2, Mat& mask, int viewstep, double disp_amp)
	{
		if (mask.empty())
		{
			if (depthfiltermode == DEPTH_FILTER_SPECKLE)
			{
				medianBlur(depth, depth2, warpedMedianKernel);
				filterSpeckles(depth2, 0, warpedSpeckesWindow, (int)(warpedSpeckesRange*disp_amp));
			}
			else if (depthfiltermode == DEPTH_FILTER_MEDIAN)
			{
				medianBlur(depth, depth2, warpedMedianKernel);
			}
			else if (depthfiltermode == DEPTH_FILTER_MEDIAN_ERODE)
			{
				medianBlur(depth, depth2, warpedMedianKernel);

				/*Mat temp;
				compare(depth2, 0 ,mask,cv::CMP_EQ);
				maxFilter(depth2,temp,Size(5,1));
				temp.copyTo(depth2,mask);*/

				Mat temp;
				minFilter(depth2, temp, Size(3, 1));
				//maxFilter(depth2,temp,Size(3,1));
				compare(temp, 0, mask, cv::CMP_EQ);
				depth2.setTo(0, mask);
			}
			else if (depthfiltermode == DEPTH_FILTER_CRACK)
			{
				crackRemove(depth, depth2, 0);
			}
			else //DEPTH_FILTER_NONE
			{
				depth.copyTo(depth2);
			}
		}
		else
		{
			if (depthfiltermode == DEPTH_FILTER_SPECKLE)
			{
				medianBlur(depth, depth2, warpedMedianKernel);
				filterSpeckles(depth2, 0, warpedSpeckesWindow, (int)(warpedSpeckesRange*disp_amp));
				compare(depth, depth2, mask, cv::CMP_NE);
			}
			else if (depthfiltermode == DEPTH_FILTER_MEDIAN)
			{
				medianBlur(depth, depth2, warpedMedianKernel);
			}
			else if (depthfiltermode == DEPTH_FILTER_MEDIAN_ERODE)
			{
				medianBlur(depth, depth2, warpedMedianKernel);

				/*Mat temp;
				compare(depth2, 0 ,mask,cv::CMP_EQ);
				maxFilter(depth2,temp,Size(5,1));
				temp.copyTo(depth2,mask);*/

				Mat temp;
				minFilter(depth2, temp, Size(3, 1));
				//maxFilter(depth2,temp,Size(3,1));
				compare(temp, 0, mask, cv::CMP_EQ);
				depth2.setTo(0, mask);
			}
			else if (depthfiltermode == DEPTH_FILTER_CRACK)
			{
				crackRemove(depth, depth2, 0);
			}
			else //DEPTH_FILTER_NONE
			{
				depth.copyTo(depth2);
			}
			compare(depth, depth2, mask, cv::CMP_NE);
		}
	}


	void StereoViewSynthesis::init(int preset)
	{
		warpMethod = WAPR_IMG_INV;
		warpSputtering = false;
		warpInterpolationMethod = INTER_CUBIC;
		depthfiltermode = DEPTH_FILTER_MEDIAN;
		postFilterMethod = POST_GAUSSIAN_FILL;
		inpaintMethod = FILL_OCCLUSION_HV;

		large_jump = 0;
		warpedMedianKernel = 3;
		warpedSpeckesWindow = 100;
		warpedSpeckesRange = 1;

		bilateral_r = 2;
		bilateral_sigma_space = 5;
		bilateral_sigma_color = 8;

		blendMethod = 0;
		blend_z_thresh = 32.0;

		occBlurSize = Size(3, 3);

		canny_t1 = 18;
		canny_t2 = 30;
		boundaryKernelSize = Size(3, 3);
		boundarySigma = 3.0;
		boundaryGaussianRatio = 1.0;

		inpaintr = 2.0;
		if (preset == StereoViewSynthesis::PRESET_FASTEST)
		{
			blend_z_thresh = 0.0;
			warpSputtering = false;
			warpInterpolationMethod = INTER_NEAREST;
			depthfiltermode = DEPTH_FILTER_MEDIAN;
			postFilterMethod = POST_FILL;
			inpaintMethod = FILL_OCCLUSION_LINE;
		}
		if (preset == StereoViewSynthesis::PRESET_SLOWEST)
		{
			warpSputtering = true;
			warpInterpolationMethod = INTER_CUBIC;
			depthfiltermode = DEPTH_FILTER_MEDIAN;
			postFilterMethod = POST_GAUSSIAN_FILL;
			inpaintMethod = FILL_OCCLUSION_HV;
		}
	}
	StereoViewSynthesis::StereoViewSynthesis(int preset)
	{
		init(preset);
	}

	StereoViewSynthesis::StereoViewSynthesis()
	{
		init(PRESET_SLOWEST);
	}

	void StereoViewSynthesis::check(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref)
	{
		if (ref.empty())ref = Mat::zeros(srcL.size(), srcL.type());
		string wname = "Stereo ViewSynthesis";
		namedWindow(wname);

		int a = (int)(100 * alpha); createTrackbar("a", wname, &a, 100);
		createTrackbar("l jump", wname, &large_jump, 500);
		createTrackbar("w med", wname, &warpedMedianKernel, 5);
		createTrackbar("sp window", wname, &warpedSpeckesWindow, 1024);
		createTrackbar("sp range", wname, &warpedSpeckesRange, 255);

		int occb = 1;
		createTrackbar("occblur", wname, &occb, 30);

		createTrackbar("canny t1", wname, &canny_t1, 255);
		createTrackbar("canny t2", wname, &canny_t2, 255);

		int boundk = 1;
		createTrackbar("boundk", wname, &boundk, 30);
		int bounds = 30;
		createTrackbar("bounds", wname, &bounds, 30);
		int bb = 0;
		createTrackbar("psnrbb", wname, &bb, 100);

		Mat dshow;
		Mat show;
		int key = 0;
		while (key != 'q')
		{
			warpedMedianKernel = (warpedMedianKernel / 2) * 2 + 1;

			occBlurSize = Size(2 * occb + 1, 2 * occb + 1);
			boundaryKernelSize = Size(2 * boundk + 1, 2 * boundk + 1);
			boundarySigma = bounds / 10.0;

			{
				this->operator()(srcL, srcR, dispL, dispR, dest, destdisp, a / 100.0, invalidvalue, disp_amp);
				//alphaSynth(srcL,srcR,dispL,dispR, dest, destdisp, alpha, invalidvalue, disp_amp);
			}
			std::cout << getPSNR(dest, ref) << std::endl;
			/*
			double minv,maxv;
			minMaxLoc(dst,&minv,&maxv);
			cout<<format("%f %f\n",minv,maxv);
			int minDisparity=(int)(minv+0.5);
			int numberOfDisparities=(int)(maxv-minv+0.5);
			cvtDisparityColor(dst,dshow,minDisparity,numberOfDisparities,isColor,1);

			addWeightedOMP(joint,1.0-(alpha/100.0),dshow,(alpha/100.0),0.0,show);
			*/
			imshow(wname, dest);
			key = waitKey(1);
		}
	}



	void StereoViewSynthesis::check(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref)
	{
		if (ref.empty())ref = Mat::zeros(src.size(), src.type());

		string wname = "Single ViewSynthesis";
		namedWindow(wname);
		int alphav = 0;
		createTrackbar("alpha", wname, &alphav, 100);

		createTrackbar("l jump", wname, &large_jump, 50);
		createTrackbar("w med", wname, &warpedMedianKernel, 5);
		createTrackbar("sp window", wname, &warpedSpeckesWindow, 1024);
		createTrackbar("sp range", wname, &warpedSpeckesRange, 255);

		int occb = 1;
		createTrackbar("occblur", wname, &occb, 30);

		createTrackbar("canny t1", wname, &canny_t1, 255);
		createTrackbar("canny t2", wname, &canny_t2, 255);

		int boundk = 1;
		createTrackbar("boundk", wname, &boundk, 30);
		int bounds = 30;
		createTrackbar("bounds", wname, &bounds, 30);
		int bb = 0;
		createTrackbar("psnrbb", wname, &bb, 100);

		int maxk = 1;
		createTrackbar("maxK", wname, &maxk, 10);
		Mat dshow;
		Mat show;
		int key = 0;
		Mat disp2;
		disp.copyTo(disp2);

		double minv, maxv;
		minMaxLoc(disp, &minv, &maxv);
		double damp = 255.0 / maxv;
		while (key != 'q')
		{
			disp2.copyTo(disp);

			maxFilter(disp, disp, Size(2 * maxk + 1, 2 * maxk + 1));

			occBlurSize = Size(2 * occb + 1, 2 * occb + 1);
			boundaryKernelSize = Size(2 * boundk + 1, 2 * boundk + 1);
			boundarySigma = bounds / 10.0;

			{
				this->operator()(src, disp, dest, destdisp, alpha, invalidvalue, disp_amp);
				//alphaSynth(srcL,srcR,dispL,dispR, dest, destdisp, alpha, invalidvalue, disp_amp);
			}
			//cout<<"PSNR:"<<calcPSNRBB(ref,dest,bb,bb)<<endl;;
			/*
			double minv,maxv;
			minMaxLoc(dst,&minv,&maxv);
			cout<<format("%f %f\n",minv,maxv);
			int minDisparity=(int)(minv+0.5);
			int numberOfDisparities=(int)(maxv-minv+0.5);
			cvtDisparityColor(dst,dshow,minDisparity,numberOfDisparities,isColor,1);

			addWeightedOMP(joint,1.0-(alpha/100.0),dshow,(alpha/100.0),0.0,show);
			*/
			destdisp.convertTo(dshow, CV_8U, damp);

			imshow(wname, dest);
			key = waitKey(1);
		}
	}
	void StereoViewSynthesis::noFilter(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
	{
		;
	}
	void StereoViewSynthesis::alphaSynth(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
	{
		if (alpha == 0.0)
		{
			srcL.copyTo(dest);
			dispL.copyTo(destdisp);
			return;
		}
		else if (alpha == 1.0)
		{
			srcR.copyTo(dest);
			dispR.copyTo(destdisp);
			return;
		}

		if (dest.empty())dest.create(srcL.size(), CV_8UC3);
		else dest.setTo(0);

		if (destdisp.empty())destdisp.create(srcL.size(), CV_8U);
		else destdisp.setTo(0);

		Mat maskL(srcL.size(), CV_8U, Scalar(0));
		Mat maskL2(srcL.size(), CV_8U, Scalar(0));
		Mat maskR(srcL.size(), CV_8U, Scalar(0));
		Mat destR(srcL.size(), CV_8UC3);
		Mat destdispR(srcL.size(), CV_8U);
		Mat temp(srcL.size(), CV_8U);
		Mat temp2(srcL.size(), CV_8U);
		Mat swap(srcL.size(), CV_8UC3);
		{
			//CalcTime t("warp");

			shiftImDisp<uchar>(srcL, dispL, dest, temp, alpha*disp_amp, disp_amp, large_jump, maskL);
			//			fillOcclusion(temp);////////////
			depthfilter(temp, destdisp, maskL2, cvRound(abs(alpha)), disp_amp);
			Mat m;  compare(destdisp, 0, m, cv::CMP_EQ);
			dest.setTo(0, m);
			maskL.setTo(0, m);
			maskL2.setTo(0, m);

			shiftImInvWithMask_<uchar>(srcL, destdisp, dest, -alpha*disp_amp, maskL2);
			maskL = maskL + maskL2;

			temp.setTo(0);
			shiftImDisp<uchar>(srcR, dispR, destR, temp, -disp_amp*(1.0 - alpha), disp_amp, large_jump, maskR);

			depthfilter(temp, destdispR, maskL2, cvRound(abs(alpha)), disp_amp);
			//filterDepth(temp,destdispR,maskL2);
			Mat m2;  compare(destdispR, 0, m2, cv::CMP_EQ);
			destR.setTo(0, m2);
			maskR.setTo(0, m2);
			maskL2.setTo(0, m2);
			shiftImInvWithMask_<uchar>(srcR, destdispR, destR, disp_amp*(1.0 - alpha), maskL2);
			filterSpeckles(destR, 0, 255, 255);

			maskR = maskR + maskL2;
		}

	{
		//	CalcTime t("blend");			
		//blendLRS<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
		blendLR<uchar>(dest, destR, destdisp, destdispR, destR, destdispR, maskL, maskR, alpha, 32);
		//guiAlphaBlend(maskL,maskR);
		//	imwrite("im.bmp",destR);
		//			imwrite("binp_dp.bmp",destdispR);
	}
	if (postFilterMethod == POST_GAUSSIAN_FILL)
	{
		{
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			Mat m;  compare(destdispR, 0, m, cv::CMP_EQ);
			fillOcclusionImDisp(destR, destdispR, invalidvalue, FILL_OCCLUSION_LINE);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);

			Mat dest2;
			boxFilter(dest, dest2, -1, occBlurSize);
			dest2.copyTo(dest, m);
			//	imwrite("im_occ.bmp",dest);
		}
		Mat edge;
		cv::Canny(destdisp, edge, canny_t1, canny_t2);
		//imshow("ee",edge);
		//dilate(edge,edge,Mat(),Point(-1,-1),2);

		Mat a;
		GaussianBlur(dest, a, boundaryKernelSize, boundarySigma);
		a.copyTo(destR, edge);
		double aa = (alpha > 1.0) ? 1.0 : alpha;
		aa = (alpha < 0.0) ? 0.0 : aa;
		aa = (0.5 - abs(aa - 0.5))*2.0;
		addWeighted(dest, 1.0 - aa, destR, aa, 0.0, dest);
	}
	else if (postFilterMethod == POST_FILL)
	{
		Mat m;  compare(destdispR, 0, m, cv::CMP_EQ);
		fillOcclusionImDisp(destR, destdispR, invalidvalue, FILL_OCCLUSION_LINE);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		Mat dest2;
		boxFilter(dest, dest2, -1, occBlurSize);
		dest2.copyTo(dest, m);
	}
	else
	{
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);
	}
	}

	void StereoViewSynthesis::analyzeSynthesizedView(Mat& srcsynth, Mat& ref)
	{
		;

	}

	void StereoViewSynthesis::makeMask(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, double alpha, int invalidvalue, double disp_amp)
	{
		if (dispL.depth() == CV_8U)
			makeMask_<uchar>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp);
		if (dispL.depth() == CV_16U)
			makeMask_<ushort>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp);
		if (dispL.depth() == CV_16S)
			makeMask_<short>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp);
		if (dispL.depth() == CV_32F)
			makeMask_<float>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp);
	}

	void StereoViewSynthesis::makeMask(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, double alpha, int invalidvalue, double disp_amp, Mat& srcsynth, Mat& ref)
	{
		makeMask(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp);
		analyzeSynthesizedView(srcsynth, ref);
	}

	template <class T>
	void StereoViewSynthesis::analyzeSynthesizedViewDetail_(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, double alpha, int invalidvalue, double disp_amp, Mat& srcsynth, Mat& ref)
	{
		vector<Mat> draw;
		allMask = Mat::ones(srcL.size(), CV_8U); allMask *= 255;//all mask 
		nonOcclusionMask = Mat::zeros(srcL.size(), CV_8U);
		occlusionMask = Mat::zeros(srcL.size(), CV_8U);//half and full occlusion
		fullOcclusionMask = Mat::zeros(srcL.size(), CV_8U);//full occlusion
		halfOcclusionMask = Mat::zeros(srcL.size(), CV_8U);//left and right half ooclusion

		boundaryMask = Mat::zeros(srcL.size(), CV_8U);//disparity boundary
		nonFullOcclusionMask = Mat::zeros(srcL.size(), CV_8U); //bar of full occlusion
		Mat vis = Mat::zeros(srcL.size(), CV_8UC3);
		Mat disp8Ubuff;
		double sub_gap = (warpSputtering) ? disp_amp : -1.0;

		Mat maskL(srcL.size(), CV_8U, Scalar(0));
		Mat maskR(srcL.size(), CV_8U, Scalar(0));
		Mat maskTemp(srcL.size(), CV_8U, Scalar(0));

		Mat dest(srcL.size(), CV_8UC3);
		Mat destdisp(srcL.size(), dispL.depth());
		Mat destR(srcL.size(), CV_8UC3);
		Mat destdispR(srcL.size(), dispL.depth());
		Mat temp(srcL.size(), dispL.depth());


		shiftDisp(dispL, temp, (float)(alpha / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp), maskL);
		depthfilter(temp, destdisp, maskTemp, cvRound(abs(alpha)), disp_amp);
		compare(destdisp, 0, maskL, cv::CMP_NE);
		dest.setTo(0);
		shiftImInvWithMask_<T>(srcL, destdisp, dest, -alpha / disp_amp, maskL, 0, warpInterpolationMethod);

		draw.push_back(dest.clone());

		shiftDisp(dispR, temp, (float)((alpha - 1.0) / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp), maskR);
		depthfilter(temp, destdispR, maskTemp, cvRound(abs(alpha)), disp_amp);
		compare(destdispR, 0, maskR, cv::CMP_NE);
		destR.setTo(0);
		shiftImInvWithMask_<T>(srcR, destdispR, destR, (1.0 - alpha) / disp_amp, maskR, 0, warpInterpolationMethod);

		draw.push_back(destR.clone());

		bitwise_and(maskL, maskR, nonOcclusionMask);
		bitwise_not(nonOcclusionMask, occlusionMask);
		bitwise_and(~maskL, ~maskR, fullOcclusionMask);
		bitwise_and(occlusionMask, ~fullOcclusionMask, halfOcclusionMask);

		bitwise_not(fullOcclusionMask, nonFullOcclusionMask);

		blendLRS<T>(dest, destR, destdisp, destdispR, destR, destdispR, maskL, maskR, alpha);

		draw.push_back(destR.clone());

		fillOcclusionImDisp(destR, destdispR, 0, FILL_OCCLUSION_HV);
		destdispR.convertTo(disp8Ubuff, CV_8U, 1.0 / disp_amp);
		cv::Canny(disp8Ubuff, boundaryMask, canny_t1, canny_t2);  //imshow("ee",edge);//waitKey();


		draw.push_back(ref);
		//imshowAnalysis("signal",draw);

		maxFilter(boundaryMask, boundaryMask, Size(3, 3));

		analyzeSynthesizedView(srcsynth, ref);
	}

	void StereoViewSynthesis::analyzeSynthesizedViewDetail(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, double alpha, int invalidvalue, double disp_amp, Mat& srcsynth, Mat& ref)
	{
		if (dispL.depth() == CV_8U)
			analyzeSynthesizedViewDetail_<uchar>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp, srcsynth, ref);
		if (dispL.depth() == CV_16U)
			analyzeSynthesizedViewDetail_<ushort>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp, srcsynth, ref);
		if (dispL.depth() == CV_16S)
			analyzeSynthesizedViewDetail_<short>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp, srcsynth, ref);
		if (dispL.depth() == CV_32F)
			analyzeSynthesizedViewDetail_<float>(srcL, srcR, dispL, dispR, alpha, invalidvalue, disp_amp, srcsynth, ref);
	}

	template <class T>
	void StereoViewSynthesis::makeMask_(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, double alpha, int invalidvalue, double disp_amp)
	{
		vector<Mat> draw;
		allMask = Mat::ones(srcL.size(), CV_8U); allMask *= 255;//all mask 
		nonOcclusionMask = Mat::zeros(srcL.size(), CV_8U);
		occlusionMask = Mat::zeros(srcL.size(), CV_8U);//half and full occlusion
		fullOcclusionMask = Mat::zeros(srcL.size(), CV_8U);//full occlusion
		halfOcclusionMask = Mat::zeros(srcL.size(), CV_8U);//left and right half ooclusion

		boundaryMask = Mat::zeros(srcL.size(), CV_8U);//disparity boundary
		nonFullOcclusionMask = Mat::zeros(srcL.size(), CV_8U); //bar of full occlusion
		Mat vis = Mat::zeros(srcL.size(), CV_8UC3);
		Mat disp8Ubuff;
		double sub_gap = (warpSputtering) ? disp_amp : -1.0;

		Mat maskL(srcL.size(), CV_8U, Scalar(0));
		Mat maskR(srcL.size(), CV_8U, Scalar(0));
		Mat maskTemp(srcL.size(), CV_8U, Scalar(0));

		Mat dest(srcL.size(), CV_8UC3);
		Mat destdisp(srcL.size(), dispL.depth());
		Mat destR(srcL.size(), CV_8UC3);
		Mat destdispR(srcL.size(), dispL.depth());
		Mat temp(srcL.size(), dispL.depth());


		shiftDisp(dispL, temp, (float)(alpha / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp), maskL);
		depthfilter(temp, destdisp, maskTemp, cvRound(abs(alpha)), disp_amp);
		compare(destdisp, 0, maskL, cv::CMP_NE);
		dest.setTo(0);
		shiftImInvWithMask_<T>(srcL, destdisp, dest, -alpha / disp_amp, maskL, 0, warpInterpolationMethod);

		draw.push_back(dest.clone());

		shiftDisp(dispR, temp, (float)((alpha - 1.0) / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp), maskR);
		depthfilter(temp, destdispR, maskTemp, cvRound(abs(alpha)), disp_amp);
		compare(destdispR, 0, maskR, cv::CMP_NE);
		destR.setTo(0);
		shiftImInvWithMask_<T>(srcR, destdispR, destR, (1.0 - alpha) / disp_amp, maskR, 0, warpInterpolationMethod);

		draw.push_back(destR.clone());

		bitwise_and(maskL, maskR, nonOcclusionMask);
		bitwise_not(nonOcclusionMask, occlusionMask);
		bitwise_and(~maskL, ~maskR, fullOcclusionMask);
		bitwise_and(occlusionMask, ~fullOcclusionMask, halfOcclusionMask);

		bitwise_not(fullOcclusionMask, nonFullOcclusionMask);

		blendLRS<T>(dest, destR, destdisp, destdispR, destR, destdispR, maskL, maskR, alpha);

		draw.push_back(destR.clone());

		fillOcclusionImDisp(destR, destdispR, 0, FILL_OCCLUSION_HV);
		destdispR.convertTo(disp8Ubuff, CV_8U, 1.0 / disp_amp);
		cv::Canny(disp8Ubuff, boundaryMask, canny_t1, canny_t2);  //imshow("ee",edge);//waitKey();

		maxFilter(boundaryMask, boundaryMask, Size(3, 3));
	}

	//#define VIS_SYNTH_INFO 0
	template <class T>
	void StereoViewSynthesis::viewsynth(const Mat& srcL, const Mat& srcR, const Mat& dispL, const Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype)
	{
		Mat disp8Ubuff;
		Mat edge;

		double sub_gap = (warpSputtering) ? disp_amp : -1.0;

		if (alpha == 0.0)
		{
			srcL.copyTo(dest);
			dispL.copyTo(destdisp);
			return;
		}
		else if (alpha == 1.0)
		{
			srcR.copyTo(dest);
			dispR.copyTo(destdisp);
			return;
		}

		if (dest.empty())dest.create(srcL.size(), CV_8UC3);
		else dest.setTo(0);

		if (destdisp.empty() || destdisp.type() != disptype)destdisp.create(srcL.size(), disptype);
		else destdisp.setTo(0);

		/*Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));

		Mat maskTemp(srcL.size(),CV_8U,Scalar(0));*/

		Mat destR(srcL.size(), CV_8UC3);
		Mat destdispR(srcL.size(), disptype);
		Mat temp(srcL.size(), disptype);
		Mat m;

#ifdef VIS_SYNTH_INFO
		Mat vis = Mat::zeros(dest.size(), CV_8UC3);
#endif
		if (warpMethod == WAPR_IMG_FWD_SUB_INV)
		{
			/*

			#ifdef VIS_SYNTH_INFO
			CalcTime t("warp");
			#endif
			shiftImDisp<T>(srcL,dispL,dest,temp,alpha/disp_amp,sub_gap,(int)(large_jump*disp_amp),maskL,warpInterpolationMethod);
			depthfilter(temp,destdisp,maskTemp,cvRound(abs(alpha)),disp_amp);
			compare(destdisp,0,m,cv::CMP_EQ);
			dest.setTo(0,m);
			maskL.setTo(0,m);
			maskTemp.setTo(0,m);
			shiftImInv_<T>(srcL,destdisp,dest,-alpha/disp_amp,maskTemp,0,warpInterpolationMethod);
			bitwise_or(maskL,maskTemp,maskL);

			temp.setTo(0);
			shiftImDisp<T>(srcR,dispR,destR,temp,(alpha-1.0)/disp_amp,sub_gap,(int)(large_jump*disp_amp),maskR,warpInterpolationMethod);
			depthfilter(temp,destdispR,maskTemp,cvRound(abs(alpha)),disp_amp);
			compare(destdispR,0,m,cv::CMP_EQ);
			destR.setTo(0,m);
			maskR.setTo(0,m);
			maskTemp.setTo(0,m);
			shiftImInv_<T>(srcR,destdispR,destR,(1.0-alpha)/disp_amp,maskTemp,0,warpInterpolationMethod);
			bitwise_or(maskR,maskTemp,maskR);
			*/
		}
		else if (warpMethod == WAPR_IMG_INV)
		{
#ifdef VIS_SYNTH_INFO
			CalcTime t("warp");
#endif
			shiftDisp(dispL, temp, (float)(alpha / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp));
			depthfilter(temp, destdisp, Mat(), cvRound(abs(alpha)), disp_amp);
			shiftImInv(srcL, destdisp, dest, (float)(-alpha / disp_amp), 0, warpInterpolationMethod);


			{
				//	CalcTime t("shift");
				shiftDisp(dispR, temp, (float)((alpha - 1.0) / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp));
			}
		{
			//	CalcTime t("filter");
			depthfilter(temp, destdispR, Mat(), cvRound(abs(alpha)), disp_amp);
		}
		{
			//CalcTime t("inter");
			//shiftImInv_<T>(srcR,destdispR,destR,(1.0-alpha)/disp_amp,maskR,0,warpInterpolationMethod);
			shiftImInv(srcR, destdispR, destR, (float)((1.0 - alpha) / disp_amp), 0, warpInterpolationMethod);
		}

		//with mask
		/*
		//	shiftDisp<T>(dispL,temp,alpha/disp_amp,sub_gap,large_jump*disp_amp,maskL);
		shiftDisp<T>(dispL,temp,alpha/disp_amp,sub_gap,(int)(large_jump*disp_amp));

		depthfilter(temp,destdisp,maskTemp,cvRound(abs(alpha)),disp_amp);

		compare(destdisp,0,maskL,cv::CMP_NE);
		dest.setTo(0);

		shiftImInv_<T>(srcL,destdisp,dest,-alpha/disp_amp,maskL,0,warpInterpolationMethod);

		//shiftDisp<T>(dispR,temp,(alpha-1.0)/disp_amp,sub_gap,large_jump*disp_amp,maskR);
		shiftDisp<T>(dispR,temp,(alpha-1.0)/disp_amp,sub_gap,large_jump*disp_amp);


		depthfilter(temp,destdispR,maskTemp,cvRound(abs(alpha)),disp_amp);
		compare(destdispR,0,maskR,cv::CMP_NE);

		destR.setTo(0);

		shiftImInv_<T>(srcR,destdispR,destR,(1.0-alpha)/disp_amp,maskR,0,warpInterpolationMethod);
		*/
	}

		{
#ifdef VIS_SYNTH_INFO
			CalcTime t("blend");
#endif

			if (blend_z_thresh <= 0.0)
			{
				//blendLRS<T>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
				//	blendLRS<T>(dest,destR,destdisp,destdispR,destR,destdisp,alpha,invalidvalue);
				if (blendMethod == 0) blendLRS<T>(dest, destR, destdisp, destdispR, destR, destdisp, alpha, invalidvalue);
				else
				{
					blendLRS_8u_leftisout(dest, destR, destdisp, destdispR, alpha, invalidvalue);//faster, but not accurate.
					dest.copyTo(destR);
				}
			}
			else
			{
				if (blendMethod == 0) blendLR<T>(dest, destR, destdisp, destdispR, destR, destdisp, alpha, (T)(blend_z_thresh*disp_amp), invalidvalue);//
				else blendLR_NearestMax<T>(dest, destR, destdisp, destdispR, destR, destdisp, alpha, (T)(blend_z_thresh*disp_amp), invalidvalue);//
				//blendLR<T>(dest,destR,destdisp,destdispR,destR,destdisp,alpha, (T)(blend_z_thresh*disp_amp),invalidvalue);//
				//blendLR<T>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha, (T)(blend_z_thresh*disp_amp));//

				//blendLRRes2<T>(dest,destR, srcL, srcR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			}
}

		if (postFilterMethod == POST_GAUSSIAN_FILL)
		{
#ifdef VIS_SYNTH_INFO

			CalcTime t("inpaint");
#endif
			compare(destdisp, 0, m, cv::CMP_EQ);

			{
				//CalcTime t("fill");
				if (inpaintMethod < FILL_OCCLUSION_INPAINT_NS)
					fillOcclusionImDisp(destR, destdisp, invalidvalue, inpaintMethod);
				else
				{

					if (inpaintMethod == FILL_OCCLUSION_INPAINT_NS) inpaint(destR, m, destR, inpaintr, INPAINT_NS);
					if (inpaintMethod == FILL_OCCLUSION_INPAINT_TELEA) inpaint(destR, m, destR, inpaintr, INPAINT_TELEA);

					fillOcclusion(destdisp);
				}
			}

		{
			//CalcTime t("blur");
			blur(destR, dest, occBlurSize);
			dest.copyTo(destR, m);
		}

		{
			//CalcTime t("canny");
			destdisp.convertTo(disp8Ubuff, CV_8U, 1.0 / disp_amp);
			cv::Canny(disp8Ubuff, edge, canny_t1, canny_t2);
		}


		//imshow("ee",edge);//waitKey();
#ifdef VIS_SYNTH_INFO
		vis.setTo(128, edge);
#endif


		if (alpha == 0.5 &&boundaryGaussianRatio == 1.0)
		{
			GaussianFilterwithMask(destR, dest, boundaryKernelSize.width / 2, (float)boundarySigma, FILTER_SLOWEST, edge);
		}
		else
		{
			Mat dst = destR.clone(); //used for destR as temp buffer

			GaussianFilterwithMask(dst, dst, boundaryKernelSize.width / 2, (float)boundarySigma, FILTER_SLOWEST, edge);
			//	GaussianBlur(dst,dest,boundaryKernelSize, boundarySigma);
			//dest.copyTo(dst,edge);

			double aa = (alpha > 1.0) ? 1.0 : alpha;
			aa = (alpha < 0.0) ? 0.0 : aa;
			aa = (0.5 - abs(aa - 0.5))*2.0;
			aa *= boundaryGaussianRatio;
			addWeighted(destR, 1.0 - aa, dst, aa, 0.0, dest);
		}

	}
		else if (postFilterMethod == POST_FILL)
		{
#ifdef VIS_SYNTH_INFO
			CalcTime t("inpaint");
#endif
			compare(destdisp, 0, m, cv::CMP_EQ);

			fillOcclusionImDisp(destR, destdisp, invalidvalue, inpaintMethod);

			blur(destR, dest, occBlurSize);
			dest.copyTo(destR, m); //imshow("ee",m);//waitKey();

			destR.copyTo(dest);
	}
		else
		{
#ifdef VIS_SYNTH_INFO
			CalcTime t("inpaint");
#endif
			destR.copyTo(dest);
		}
#ifdef VIS_SYNTH_INFO
		imshow("vis", vis);
#endif
}


	template <class T>
	void shiftImDispNN3_(const Mat& srcim, const Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
	{
		int ij = 0;
		const int ljump = large_jump*sub_gap * 1;
		//const int iamp=cvRound(amp);
		if (amp > 0)
		{
			for (int j = 0; j < srcdisp.rows; j++)
			{
				const uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);
				const T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);

				for (int i = srcdisp.cols - 2; i >= 0; i--)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i - 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(disp*amp);
					//				const double ia = (disp*amp)-dest;
					//				const double a = 1.0-ia;

					if (sub > ljump || abs(disp - s[i + 1]) > ljump)
					{
						i -= ij;
						continue;
					}


					if (i - dest - 1 >= 0 && i - dest - 1 < srcdisp.cols - 1)
					{
						if (disp > d[i - dest])
						{
							m[i - dest] = 255;
							d[i - dest] = disp;
							dim[3 * (i - dest) + 0] = sim[3 * i + 0];
							dim[3 * (i - dest) + 1] = sim[3 * i + 1];
							dim[3 * (i - dest) + 2] = sim[3 * i + 2];
							if (issub)
							{

								m[i - dest - 1] = 255;
								d[i - dest - 1] = disp;
								dim[3 * (i - dest - 1) + 0] = sim[3 * i - 3];
								dim[3 * (i - dest - 1) + 1] = sim[3 * i - 2];
								dim[3 * (i - dest - 1) + 2] = sim[3 * i - 1];
							}
						}
					}
				}
			}
		}
		else if (amp < 0)
		{
			for (int j = 0; j<srcdisp.rows; j++)
			{
				const uchar* sim = srcim.ptr<uchar>(j);
				uchar* dim = destim.ptr<uchar>(j);
				uchar* m = mask.ptr<uchar>(j);

				const T* s = srcdisp.ptr<T>(j);
				T* d = destdisp.ptr<T>(j);
				for (int i = 1; i<srcdisp.cols; i++)
				{
					const T disp = s[i];
					int sub = (int)(abs(disp - s[i + 1]));
					bool issub = (sub <= sub_gap) ? true : false;
					const int dest = (int)(-amp*disp);//符号
					//				const double ia = (-amp*disp)-dest;
					//				const double a = 1.0-ia;


					if (abs(disp - s[i - 1])>ljump || abs(disp - s[i + 1])>ljump)
					{
						i += ij;
						continue;
					}

					if (i + dest + 1 >= 0 && i + dest + 1 < srcdisp.cols - 1)
					{
						if (disp > d[i + dest])
						{
							m[i + dest] = 255;
							d[i + dest] = (T)disp;

							dim[3 * (i + dest) + 0] = sim[3 * i + 0];
							dim[3 * (i + dest) + 1] = sim[3 * i + 1];
							dim[3 * (i + dest) + 2] = sim[3 * i + 2];

							if (issub)
							{
								m[i + dest + 1] = 255;
								d[i + dest + 1] = (T)disp;
								dim[3 * (i + dest + 1) + 0] = sim[3 * i + 3];
								dim[3 * (i + dest + 1) + 1] = sim[3 * i + 4];
								dim[3 * (i + dest + 1) + 2] = sim[3 * i + 5];
							}
						}
					}
				}
			}
		}
		else
		{
			srcim.copyTo(destim);
			srcdisp.copyTo(destdisp);
			mask.setTo(Scalar(255));
		}
	}

	void StereoViewSynthesis::viewsynthSingleAlphaMap(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype)
	{
		typedef uchar T;
		if (alpha == 0.0)
		{
			src.copyTo(dest);
			disp.copyTo(destdisp);
			return;
		}

		double sub_gap = (warpSputtering) ? disp_amp : -1.0;
		//large_jump = large_jump<1 ?1:large_jump;

		if (dest.empty())dest = Mat::zeros(src.size(), CV_8UC3);
		else dest.setTo(0);

		if (destdisp.empty() || destdisp.depth() != disptype)destdisp = Mat::zeros(src.size(), disptype);
		else destdisp.setTo(0);

		Mat mask = Mat::zeros(src.size(), CV_8U);
		Mat maskTemp = Mat::zeros(src.size(), CV_8U);
		Mat temp = Mat::zeros(src.size(), disptype);

		/*shiftDisp<T>(disp,temp,alpha/disp_amp,disp_amp,large_jump,mask);
		depthfilter(temp,destdisp,maskTemp,cvRound(abs(alpha)),disp_amp);
		compare(destdisp,0,mask,cv::CMP_NE);
		shiftImInv_<T>(src,destdisp,dest,-alpha/disp_amp,mask,0,INTER_NEAREST); */


		Mat zero_ = Mat::zeros(src.size(), CV_8U);

		shiftDisp(disp, temp, (float)(alpha / disp_amp), (float)sub_gap, (int)(large_jump*disp_amp));
		depthfilter(temp, destdisp, Mat(), cvRound(abs(alpha)), disp_amp);
		//shiftImInv_<T>(src,destdisp,dest,(float)(-alpha/disp_amp),0,warpInterpolationMethod); 
		shiftImInv(src, destdisp, dest, (float)(-alpha / disp_amp), 0, INTER_NEAREST);

		//
		//shiftImDispNN3_<T>(src,disp,dest,temp,alpha/disp_amp,zero_,large_jump,(int)disp_amp);// warpInterpolationMethod
		//depthfilter(temp,destdisp,mask,cvRound(abs(alpha)),disp_amp);
		//shiftImInvWithMask_<T>(src,destdisp,dest,-alpha/disp_amp,mask,0, INTER_NEAREST);

		//guiAlphaBlend(dest,destdisp);
	}

	template <class T>
	void StereoViewSynthesis::viewsynthSingle(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype)
	{
		Mat disp8Ubuff;
		Mat edge;
		if (alpha == 0.0)
		{
			src.copyTo(dest);
			disp.copyTo(destdisp);
			return;
		}

		large_jump = large_jump < 1 ? 1 : large_jump;

		if (dest.empty())dest.create(src.size(), CV_8UC3);
		else dest.setTo(0);

		if (destdisp.empty() || destdisp.type() != disptype)destdisp.create(src.size(), disptype);
		else destdisp.setTo(0);


		Mat mask(src.size(), CV_8U);
		Mat destR(src.size(), CV_8UC3);
		Mat destdispR(src.size(), disptype);
		Mat temp(src.size(), disptype);


		{
			//CalcTime t("warp");
			shiftImDisp<T>(src, disp, dest, temp, alpha / disp_amp, disp_amp, large_jump);

			depthfilter(temp, destdisp, mask, cvRound(abs(alpha)), disp_amp);

			shiftImInvWithMask_<T>(src, destdisp, dest, -alpha / disp_amp, mask);
			//destdisp.convertTo(temp,CV_8U,5.0/disp_amp);imshow("a",temp);//waitKey();
		}


		if (postFilterMethod == POST_GAUSSIAN_FILL)
		{
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);
			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			compare(destdisp, 0, diskMask, cv::CMP_EQ);
			fillOcclusionImDisp(dest, destdisp, invalidvalue, FILL_OCCLUSION_LINE);

			//imshow("ee",dest);waitKey();
			boxFilter(dest, destR, -1, occBlurSize);
			destR.copyTo(dest, diskMask);

			destdisp.convertTo(disp8Ubuff, CV_8U, 1.0 / disp_amp);

			cv::Canny(disp8Ubuff, edge, canny_t1, canny_t2);

			//imshow("ee",edge);waitKey();
			//dilate(edge,edge,Mat(),Point(-1,-1),2);
			Mat a;
			dest.copyTo(destR);
			GaussianBlur(dest, a, boundaryKernelSize, boundarySigma);

			a.copyTo(dest, edge);

			//imshow("dest",destR);waitKey();

			double aa = (alpha > 1.0) ? 1.0 : alpha;
			aa = (alpha < 0.0) ? 0.0 : aa;
			aa = (0.5 - abs(aa - 0.5))*2.0;
			addWeighted(dest, aa, destR, 1.0 - aa, 0.0, dest);
		}
		else if (postFilterMethod == POST_FILL)
		{
			compare(destdisp, 0, diskMask, cv::CMP_EQ);

			fillOcclusionImDisp(dest, destdisp, invalidvalue, FILL_OCCLUSION_LINE);

			boxFilter(dest, destR, -1, occBlurSize);
			destR.copyTo(dest, diskMask);
		}
		else
		{
			;
		}
	}

	void StereoViewSynthesis::operator()(const Mat& srcL, const Mat& srcR, const Mat& dispL, const Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
	{
		int type = dispL.depth();
		if (type == CV_8U)
		{
			viewsynth<uchar>(srcL, srcR, dispL, dispR, dest, destdisp, alpha, invalidvalue, disp_amp, CV_8U);
		}
		else if (type == CV_16S)
		{
			viewsynth<short>(srcL, srcR, dispL, dispR, dest, destdisp, alpha, invalidvalue, disp_amp, CV_16S);
		}
		else if (type == CV_16U)
		{
			viewsynth<ushort>(srcL, srcR, dispL, dispR, dest, destdisp, alpha, invalidvalue, disp_amp, CV_16U);
		}
		else
		{
			cout << "not support" << endl;
		}
	}
	void StereoViewSynthesis::operator()(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
	{
		int type = disp.depth();
		if (type == CV_8U)
		{
			viewsynthSingle<uchar>(src, disp, dest, destdisp, alpha, invalidvalue, disp_amp, CV_8U);
		}
		else if (type == CV_16S)
		{
			viewsynthSingle<short>(src, disp, dest, destdisp, alpha, invalidvalue, disp_amp, CV_16S);
		}
		else if (type == CV_16U)
		{
			viewsynthSingle<ushort>(src, disp, dest, destdisp, alpha, invalidvalue, disp_amp, CV_16U);
		}
		else
		{
			cout << "not support" << endl;
		}
	}

	void StereoViewSynthesis::preview(Mat& srcL, Mat& srcR, Mat& dispL, Mat& dispR, int invalidvalue, double disp_amp)
	{
		string wname = "synth preview";
		namedWindow(wname);
		int x = 500;
		createTrackbar("x", wname, &x, 900);
		Mat dest, destdisp;
		int key = 0;
		while (key != 'q')
		{
			double vp = (x - 450) / 100.0;
			operator()(srcL, srcR, dispL, dispR, dest, destdisp, vp, invalidvalue, disp_amp);
			imshow(wname, dest);
			key = waitKey(1);
		}
		destroyWindow(wname);
	}

	void StereoViewSynthesis::preview(Mat& src, Mat& disp, int invalidvalue, double disp_amp)
	{
		string wname = "synth";
		namedWindow(wname);
		static int x = 1000;
		createTrackbar("x", wname, &x, 2000);

		static int alpha = 0; createTrackbar("alpha", wname, &alpha, 100);
		createTrackbar("mode", wname, &depthfiltermode, 1);
		static int maxr = 0;
		createTrackbar("max", wname, &maxr, 10);
		Mat dest, destdisp;
		int key = 0;

		double maxv, minv;
		minMaxLoc(disp, &minv, &maxv);

		while (key != 'q')
		{
			double vp = (x - 1000) / 100.0;
			Mat disp2;
			maxFilter(disp, disp2, Size(2 * maxr + 1, 2 * maxr + 1));
			operator()(src, disp2, dest, destdisp, vp, invalidvalue, disp_amp);
			//shiftViewSynthesisFilter(src,disp,dest,destdisp,vp,0,1.0/disp_amp);

			Mat dshow;
			Mat dtemp; Mat(destdisp*255.0 / maxv).convertTo(dtemp, CV_8U);
			//applyColorMap(dtemp,dshow,2);

			alphaBlend(dshow, dest, alpha / 100.0, dest);
			imshow(wname, dest);
			//imshowDisparity("disp",destdisp,2,0,48,(int)disp_amp);
			key = waitKey(1);
		}
		destroyWindow(wname);
	}










	DepthMapSubpixelRefinment::DepthMapSubpixelRefinment()
	{
		;
	}

	template <class S, class T>
	void DepthMapSubpixelRefinment::getDisparitySubPixel_Integer(Mat& src, Mat& dest, int disp_amp)
	{
		T* disp = dest.ptr<T>(0);
		S* s = src.ptr<S>(0);
		const int imsize = src.size().area();

		for (int j = 0; j < imsize; j++)
		{
			T d = (T)s[j];
			float f = cslice.at<float>(j);
			float p = pslice.at<float>(j);
			float m = mslice.at<float>(j);

			float md = ((p + m - (2.f*f))*2.f);
			if (md != 0)
			{
				float dd;
				float diff = (p - m) / md;
				if (abs(diff) <= 1.f)
					dd = (float)d - diff;
				else
					dd = (float)d;
				//cout<<d<<":"<<dd<<","<<md<<endl;getchar();
				disp[j] = (T)(disp_amp*dd + 0.5f);
			}
			else
			{
				disp[j] = (T)(disp_amp*d + 0.5f);
			}
		}
	}

	void DepthMapSubpixelRefinment::bluidCostSlice(const Mat& baseim, const Mat& warpim, Mat& dest, int metric, int truncate)
	{
		if (dest.empty())dest = Mat::zeros(baseim.size(), CV_32F);

		uchar* s1 = (uchar*)baseim.ptr<uchar>(0);
		uchar* s2 = (uchar*)warpim.ptr<uchar>(0);
		float* d = dest.ptr<float>(0);
		const int size = baseim.size().area();
		if (metric == 1)
		{
			for (int i = 0; i < size; i++)
			{
				if (s2[3 * i] == 0) d[i] = 0;
				else
				{
					d[i] = (float)(min(abs(s1[3 * i + 0] - s2[3 * i + 0]), truncate)
						+ min(abs(s1[3 * i + 1] - s2[3 * i + 1]), truncate)
						+ min(abs(s1[3 * i + 2] - s2[3 * i + 2]), truncate));
				}
			}
		}
		else if (metric == 2)
		{
			int t2 = truncate*truncate;
			for (int i = 0; i < size; i++)
			{
				if (s2[3 * i] == 0) d[i] = 0;
				else
				{
					d[i] = (float)(min((s1[3 * i + 0] - s2[3 * i + 0])*(s1[3 * i + 0] - s2[3 * i + 0]), t2)
						+ min((s1[3 * i + 1] - s2[3 * i + 1])*(s1[3 * i + 1] - s2[3 * i + 1]), t2)
						+ min((s1[3 * i + 2] - s2[3 * i + 2])*(s1[3 * i + 2] - s2[3 * i + 2]), t2));
				}
			}
		}
	}

	double grayNorm(const Mat& src, const Mat& ref, int flag)
	{
		double ret = 0.0;
		Mat a;
		Mat b;
		cvtColor(src, a, COLOR_BGR2GRAY);
		cvtColor(ref, b, COLOR_BGR2GRAY);
		int v = 0;
		int count = 0;

		if (flag == NORM_L2)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				if (a.at<uchar>(i) != 0 && b.at<uchar>(i) != 0)
				{
					int e = a.at<uchar>(i)-b.at<uchar>(i);
					v += e*e;
					count++;
				}
			}
			ret = (double)v / (double)count;
		}
		else if (flag == NORM_L1)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				if (a.at<uchar>(i) != 0 && b.at<uchar>(i) != 0)
				{
					int e = a.at<uchar>(i)-b.at<uchar>(i);
					v += abs(e);
					count++;
				}
			}
			ret = (double)v / (double)count;
		}
		else
		{
			cout << "does not suppert the type of the norm" << endl;
		}
		return ret;
	}

	void warping(const Mat& im, const Mat& disp, Mat& warp, float amp, bool l2r)
	{
		Mat wdisp = Mat::zeros(disp.size(), disp.type());
		if (warp.empty()) warp.create(im.size(), im.type());
		int interpolation = INTER_NEAREST;

		if (l2r == true)
		{
			shiftDisp(disp, wdisp, amp, amp, 0);
			medianBlur(wdisp, wdisp, 3);
			shiftImInv(im, wdisp, warp, (float)(-amp), 0, interpolation);
		}
		else
		{
			shiftDisp(disp, wdisp, -amp, amp, 0);
			medianBlur(wdisp, wdisp, 3);
			shiftImInv(im, wdisp, warp, (float)(amp), 0, interpolation);
		}
	}

	void warpingMask(const Mat& disp, Mat& mask, float amp, bool l2r)
	{
		Mat wdisp = Mat::zeros(disp.size(), disp.type());

		if (l2r == true)
		{
			shiftDisp(disp, wdisp, amp, amp, 0);
			medianBlur(wdisp, wdisp, 3);
		}
		else
		{
			shiftDisp(disp, wdisp, -amp, amp, 0);
			medianBlur(wdisp, wdisp, 3);
		}
		compare(wdisp, 0, mask, CMP_EQ);
	}

	double DepthMapSubpixelRefinment::calcReprojectionError(const Mat& leftim, const Mat& rightim, const Mat& leftdisp, const Mat& rightdisp, int disp_amp, bool left2right)
	{
		double ret = 0.0;
		const int depth = leftdisp.depth();
		int interpolation = INTER_CUBIC;
		Mat disp = Mat::zeros(leftdisp.size(), leftdisp.type());
		Mat warp = Mat::zeros(leftim.size(), leftim.type());
		int nrm = NORM_L2;
		float amp = 1.f / (float)disp_amp;
		if (left2right)
		{
			warping(leftim, leftdisp, warp, amp, true);
			ret = grayNorm(rightim, warp, nrm);
		}
		else
		{
			warping(rightim, rightdisp, warp, amp, false);
			ret = grayNorm(leftim, warp, nrm);
		}

		return ret;
	}


	void DepthMapSubpixelRefinment::operator()(const Mat& leftim, const Mat& rightim, const Mat& leftdisp, const Mat& rightdisp, int disp_amp, Mat& leftdest, Mat& rightdest)
	{
		cout << "Pre E L: " << calcReprojectionError(leftim, rightim, leftdisp, rightdisp, disp_amp, true) << endl; cout << "Pre E R: " << calcReprojectionError(leftim, rightim, leftdisp, rightdisp, disp_amp, false) << endl;

		int AMP = 16;

		Mat leftdispf; leftdisp.convertTo(leftdispf, CV_32F);
		Mat rightdispf; rightdisp.convertTo(rightdispf, CV_32F);

		Mat ld, rd;
		leftdisp.convertTo(ld, CV_8U, 1.0 / (double)disp_amp);
		rightdisp.convertTo(rd, CV_8U, 1.0 / (double)disp_amp);

		//cout<<"cmp E L: "<<calcReprojectionError(leftim,rightim,ld,rd,1,true)<<endl;cout<<"cmp E R: "<<calcReprojectionError(leftim,rightim,ld,rd,1,false)<<endl;


		Mat wleftim = Mat::zeros(leftim.size(), leftim.type());
		Mat pwleftim = Mat::zeros(leftim.size(), leftim.type());
		Mat mwleftim = Mat::zeros(leftim.size(), leftim.type());


		Mat wrightim = Mat::zeros(leftim.size(), leftim.type());
		Mat pwrightim = Mat::zeros(leftim.size(), leftim.type());
		Mat mwrightim = Mat::zeros(leftim.size(), leftim.type());

		Mat temp;

		warping(rightim, rd, wrightim, 1.0, false);
		warpShift(wrightim, pwrightim, 1, 0, BORDER_REPLICATE);
		warpShift(wrightim, mwrightim, -1, 0, BORDER_REPLICATE);

		warping(leftim, ld, wleftim, 1.0, true);
		warpShift(wleftim, pwleftim, -1, 0, BORDER_REPLICATE);
		warpShift(wleftim, mwleftim, 1, 0, BORDER_REPLICATE);

		int d = 7;
		int nrm = 1;
		int trunc = 30;
		int t2 = (int)(trunc*trunc / 10.0);


		Mat srightdisp = Mat::zeros(leftdisp.size(), CV_16S);
		Mat sleftdisp = Mat::zeros(leftdisp.size(), CV_16S);

		bluidCostSlice(rightim, wleftim, cslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(cslice,wleftdisp,cslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(cslice,rightdispf,cslice,Size(d,d),(float)disp_amp);
		blur(cslice, cslice, Size(d, d));
		//	imshowScale("error",cslice,5.0);waitKey();
		bluidCostSlice(rightim, pwleftim, pslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(pslice,pwleftdisp,pslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(pslice,rightdispf,pslice,Size(d,d),(float)disp_amp);
		blur(pslice, pslice, Size(d, d));
		//	imshowScale("error",pslice,5.0);waitKey();
		bluidCostSlice(rightim, mwleftim, mslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(mslice,mwleftdisp,mslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(mslice,rightdispf,mslice,Size(d,d),(float)disp_amp);
		blur(mslice, mslice, Size(d, d));
		//	imshowScale("error",mslice,5.0);waitKey();


		getDisparitySubPixel_Integer<uchar, short>(rd, srightdisp, AMP);


		bluidCostSlice(leftim, wrightim, cslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(cslice,wleftdisp,cslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(cslice,leftdispf,cslice,Size(d,d),(float)disp_amp);
		blur(cslice, cslice, Size(d, d));
		bluidCostSlice(leftim, pwrightim, pslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(pslice,pwleftdisp,pslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(pslice,leftdispf,pslice,Size(d,d),(float)disp_amp);
		blur(pslice, pslice, Size(d, d));
		bluidCostSlice(leftim, mwrightim, mslice, nrm, trunc);
		//jointBinalyWeightedRangeFilter(mslice,mwleftdisp,mslice,Size(d,d),(float)disp_amp);
		//jointBinalyWeightedRangeFilter(mslice,leftdispf,mslice,Size(d,d),(float)disp_amp);
		blur(mslice, mslice, Size(d, d));

		getDisparitySubPixel_Integer<uchar, short>(ld, sleftdisp, AMP);

		cout << "ref E L: " << calcReprojectionError(leftim, rightim, sleftdisp, srightdisp, AMP, true) << endl;
		cout << "ref E R: " << calcReprojectionError(leftim, rightim, sleftdisp, srightdisp, AMP, false) << endl;
		binalyWeightedRangeFilter(sleftdisp, leftdest, Size(5, 5), 16);
		binalyWeightedRangeFilter(srightdisp, rightdest, Size(5, 5), 16);

		cout << "sub E L: " << calcReprojectionError(leftim, rightim, leftdest, rightdest, AMP, true) << endl;
		cout << "sub E R: " << calcReprojectionError(leftim, rightim, leftdest, rightdest, AMP, false) << endl;
		cout << endl;
	}

	void DepthMapSubpixelRefinment::naive(const Mat& leftim, const Mat& rightim, const Mat& leftdisp, const Mat& rightdisp, int disp_amp, Mat& leftdest, Mat& rightdest)
	{
		cout << "Pre E L: " << calcReprojectionError(leftim, rightim, leftdisp, rightdisp, disp_amp, true) << endl; cout << "Pre E R: " << calcReprojectionError(leftim, rightim, leftdisp, rightdisp, disp_amp, false) << endl;

		int AMP = 16;

		Mat leftdispf; leftdisp.convertTo(leftdispf, CV_32F);
		Mat rightdispf; rightdisp.convertTo(rightdispf, CV_32F);

		Mat ld, rd;
		leftdisp.convertTo(ld, CV_8U, 1.0 / (double)disp_amp);
		rightdisp.convertTo(rd, CV_8U, 1.0 / (double)disp_amp);

		//cout<<"cmp E L: "<<calcReprojectionError(leftim,rightim,ld,rd,1,true)<<endl;cout<<"cmp E R: "<<calcReprojectionError(leftim,rightim,ld,rd,1,false)<<endl;


		Mat wleftim = Mat::zeros(leftim.size(), leftim.type());
		Mat pwleftim = Mat::zeros(leftim.size(), leftim.type());
		Mat mwleftim = Mat::zeros(leftim.size(), leftim.type());


		Mat wrightim = Mat::zeros(leftim.size(), leftim.type());
		Mat pwrightim = Mat::zeros(leftim.size(), leftim.type());
		Mat mwrightim = Mat::zeros(leftim.size(), leftim.type());

		Mat temp;

		int d = 7;
		int nrm = NORM_L2;
		int trunc = 255;

		Mat mask;
		Mat cost;


		Mat costL = Mat::ones(leftdisp.size(), CV_32F)*FLT_MAX;
		Mat costR = Mat::ones(leftdisp.size(), CV_32F)*FLT_MAX;
		Mat dispL = Mat::ones(leftdisp.size(), CV_8U) * 10;
		Mat dispR = Mat::ones(leftdisp.size(), CV_8U) * 10;

		warping(rightim, rd, wrightim, 1.0, false);
		warping(leftim, ld, wleftim, 1.0, true);

		//bluidCostSlice(leftim,wrightim,costL, nrm, trunc);
		//bluidCostSlice(rightim,wleftim,costR, nrm, trunc);
		Mat lf, rf;
		ld.convertTo(lf, CV_32F);
		rd.convertTo(rf, CV_32F);
		for (int i = -10; i <= 10; i++)
		{
			warpShiftSubpix(wrightim, pwrightim, i / 10.0, 0.0, BORDER_REPLICATE);
			bluidCostSlice(leftim, pwrightim, cost, nrm, trunc);

			jointBinalyWeightedRangeFilter(cost, lf, cost, Size(d, d), 2);
			//blur(cost,cost,Size(d,d));

			compare(cost, costL, mask, CMP_LT);
			cost.copyTo(costL, mask);
			dispL.setTo(i + 10, mask);

			//imshow("mask",mask);waitKey();

			warpShiftSubpix(wleftim, pwleftim, -i / 10.0, 0.0, BORDER_REPLICATE);
			bluidCostSlice(rightim, pwleftim, cost, nrm, trunc);

			//blur(cost,cost,Size(d,d));
			jointBinalyWeightedRangeFilter(cost, rf, cost, Size(d, d), 2);

			compare(cost, costR, mask, CMP_LT);
			cost.copyTo(costR, mask);
			dispR.setTo(i + 10, mask);


		}



		imshowScale("derror", dispL, 10);
		imshowScale("derrorR", dispR, 10);
		Mat srightdisp = Mat::zeros(leftdisp.size(), CV_16S);
		Mat sleftdisp = Mat::zeros(leftdisp.size(), CV_16S);

		Mat maskL, maskR;
		warpingMask(ld, maskL, 1, true);

		warpingMask(rd, maskR, 1, false);

		//guiAlphaBlend(leftdisp,maskR);
		//guiAlphaBlend(rightdisp,maskL);
		for (int i = 0; i < leftdisp.size().area(); i++)
		{
			if (maskR.at<uchar>(i) != 0) srightdisp.at<short>(i) = (short)((rd.at<uchar>(i)+(dispR.at<uchar>(i)-10) / 10.0)*AMP + 0.5);
			else srightdisp.at<short>(i) = (short)(rightdisp.at<uchar>(i) / (double)disp_amp*AMP + 0.5);

			if (maskL.at<uchar>(i) != 0) sleftdisp.at<short>(i) = (short)((ld.at<uchar>(i)+(dispL.at<uchar>(i)-10) / 10.0)*AMP + 0.5);
			else sleftdisp.at<short>(i) = (short)(leftdisp.at<uchar>(i) / (double)disp_amp*AMP + 0.5);
		}

		cout << "ref E L: " << calcReprojectionError(leftim, rightim, sleftdisp, srightdisp, AMP, true) << endl;
		cout << "ref E R: " << calcReprojectionError(leftim, rightim, sleftdisp, srightdisp, AMP, false) << endl;
		binalyWeightedRangeFilter(sleftdisp, leftdest, Size(3, 3), 16);
		binalyWeightedRangeFilter(srightdisp, rightdest, Size(3, 3), 16);

		cout << "sub E L: " << calcReprojectionError(leftim, rightim, leftdest, rightdest, AMP, true) << endl;
		cout << "sub E R: " << calcReprojectionError(leftim, rightim, leftdest, rightdest, AMP, false) << endl;
		cout << endl;
	}
}