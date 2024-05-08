#include "pch.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void RANDOMSampling(const vector<Mat>& src, Mat& dest, double rate, Rect roi)
{
	RNG rng(cv::getTickCount());
	const int channels = src.size();
	const int sample_num = (roi.width * roi.height) * rate;
	dest.create(Size(sample_num, channels), CV_32F);

	AutoBuffer<const float*> sptr(channels);
	vector<float*> dptr(channels);
	for (int c = 0; c < channels; c++)
	{
		dptr[c] = dest.ptr<float>(c);
	}

	int sample = 0;
	for (int j = 0; j < roi.height; j++)
	{
		for (int c = 0; c < channels; c++)
		{
			sptr[c] = src[c].ptr<float>(j + roi.y, roi.x);
		}
		for (int i = 0; i < roi.width; i++)
		{
			float v = rng.uniform(0.f, 1.f);
			if (v > 1.0 - rate)
			{
				for (int c = 0; c < channels; c++)
				{
					dptr[c][sample] = sptr[c][i];
				}
				sample++;
				if (sample == sample_num) return;
			}
		}
	}
	for (int i = sample; i < sample_num; i++)
	{
		int v = rng.uniform(0, sample);
		for (int c = 0; c < channels; c++)
		{
			dptr[c][i] = dptr[c][v];
		}
	}
}

void NNSampling(const vector<Mat>& src, Mat& dest, int scale, Rect roi)
{
	const int channels = src.size();
	const int sample_num = (roi.width * roi.height) / (scale * scale);
	dest.create(Size(sample_num, channels), CV_32F);

	AutoBuffer<const float*> sptr(channels);
	vector<float*> dptr(channels);
	for (int c = 0; c < channels; c++)
	{
		dptr[c] = dest.ptr<float>(c);
	}
	int sample = 0;
	for (int j = 0; j < roi.height; j += scale)
	{
		for (int c = 0; c < channels; c++)
		{
			sptr[c] = src[c].ptr<float>(j + roi.y, roi.x);
		}
		for (int i = 0; i < roi.width; i += scale)
		{
			for (int c = 0; c < channels; c++)
			{
				dptr[c][sample] = sptr[c][i];
			}
			sample++;
		}
	}
}

void AREASampling(const vector<Mat>& src, Mat& dest, int scale, Rect roi)
{
	const float div = 1.f / (scale * scale);

	const int channels = src.size();
	const int sample_num = (roi.width * roi.height) / (scale * scale);
	dest.create(Size(sample_num, channels), CV_32F);

	AutoBuffer<const float*> sptr(channels);
	AutoBuffer<float> ave(channels);
	vector<float*> dptr(channels);
	for (int c = 0; c < channels; c++)
	{
		dptr[c] = dest.ptr<float>(c);
	}
	const int w = src[0].cols;
	int sample = 0;
	for (int j = 0; j < roi.height; j += scale)
	{
		for (int c = 0; c < channels; c++)
		{
			sptr[c] = src[c].ptr<float>(j + roi.y, roi.x);
		}
		for (int i = 0; i < roi.width; i += scale)
		{
			for (int c = 0; c < channels; c++)
			{
				ave[c] = 0.f;
			}
			for (int l = 0; l < scale; l++)
			{
				for (int k = 0; k < scale; k++)
				{
					for (int c = 0; c < channels; c++)
					{
						//ave[c] += sptr[c][l * w + i + k];
						ave[c] = sptr[c][l * w + i + k];
					}
				}
			}
			for (int c = 0; c < channels; c++)
			{
				//dptr[c][sample] = ave[c] * div;
				dptr[c][sample] = ave[c];
			}
			sample++;
		}
	}
}

//dest.create(Size(sample_num, channels), CV_32F);
template<int scale>
void gradientMaxSampling_(const vector<Mat>& src, Mat& dest, Rect roi)
{
	const int channels = src.size();
	const int sample_num = (roi.width * roi.height) / (scale * scale);
	dest.create(Size(sample_num, channels), CV_32F);

	const int h = src[0].rows;
	const int w = src[0].cols;
	Mat buff(Size(roi.width, scale), CV_32F);

	const int c = 0;
	vector<float*> dptr(channels);
	for (int c = 0; c < channels; c++)
	{
		dptr[c] = dest.ptr<float>(c);
	}

	int sample = 0;
	float* d = dest.ptr<float>();
	for (int j = roi.y; j < roi.y + roi.height; j += scale)
	{
		for (int l = 0; l < scale; l++)
		{
			const float* sc = src[c].ptr<float>(j + l);
			const float* sm = src[c].ptr<float>(max(j + l - 1, 0));
			const float* sp = src[c].ptr<float>(min(j + l + 1, h - 1));

			//const float* smm = src.ptr<float>(max(j - 2, 0));
			//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
			float* b = buff.ptr<float>(l);
			/*for (int i = 1; i < w - 1; i++)
			{
				//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
				//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
				if constexpr (false)
				{
					const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
					const float dy = 0.5f * (sm[i] + sp[i]);
					//d[i] += sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
					b[i] = sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
				}
				else
				{
					b[i] = max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
				}
				//d[i] += sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
				//d[i] += 0.5f * ((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])) + max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
			}*/
			for (int i = roi.x; i < roi.x + roi.width; i += 8)
			{
				const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
				const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
				const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
				const __m256 msp = _mm256_loadu_ps(sp + i + 0);
				const __m256 msm = _mm256_loadu_ps(sm + i + 0);
				_mm256_store_ps(b + i - roi.x, _mm256_max_ps(
					_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
					_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
				));
			}
		}

		for (int l = 0; l < scale; l++)
		{
			buff.at<float>(l, 0) = 0.f;
			buff.at<float>(l, roi.width - 1) = 0.f;
		}
		for (int i = 0; i < roi.width; i += scale)
		{
			int idx = 0;
			float gmax = 0.f;
			const float* b = buff.ptr<float>(0, i);
			for (int l = 0; l < scale; l++)
			{
				for (int k = 0; k < scale; k++)
				{
					if (gmax < b[k])
					{
						gmax = b[k];
						idx = l * w + k;
					}
				}
				b += roi.width;
			}

			idx += (j * w + i) + roi.x;
			for (int c = 0; c < channels; c++)
			{
				dptr[c][sample] = src[c].at<float>(idx);
			}
			sample++;
		}
	}
}

void gradientMaxSampling_(const vector<Mat>& src, Mat& dest, int scale, Rect roi)
{
	const int channels = src.size();
	const int sample_num = (roi.width * roi.height) / (scale * scale);
	dest.create(Size(sample_num, channels), CV_32F);

	const int h = src[0].rows;
	const int w = src[0].cols;
	Mat buff(Size(roi.width, scale), CV_32F);

	const int c = 0;
	vector<float*> dptr(channels);
	for (int c = 0; c < channels; c++)
	{
		dptr[c] = dest.ptr<float>(c);
	}

	int sample = 0;
	float* d = dest.ptr<float>();
	for (int j = roi.y; j < roi.y + roi.height; j += scale)
	{
		for (int l = 0; l < scale; l++)
		{
			const float* sc = src[c].ptr<float>(j + l);
			const float* sm = src[c].ptr<float>(max(j + l - 1, 0));
			const float* sp = src[c].ptr<float>(min(j + l + 1, h - 1));

			//const float* smm = src.ptr<float>(max(j - 2, 0));
			//const float* spp = src.ptr<float>(min(j + 2, src.rows - 1));
			float* b = buff.ptr<float>(l);
			/*for (int i = 1; i < w - 1; i++)
			{
				//const float dx = 0.25f * (sc[i - 1] + sc[i + 1] + sc[i - 2] + sc[i + 2]);
				//const float dy = 0.25f * (sm[i] + sp[i] + smm[i] + spp[i]);
				if constexpr (false)
				{
					const float dx = 0.5f * (sc[i - 1] + sc[i + 1]);
					const float dy = 0.5f * (sm[i] + sp[i]);
					//d[i] += sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
					b[i] = sqrt((sc[i] - dx) * (sc[i] - dx) + (sc[i] - dy) * (sc[i] - dy));
				}
				else
				{
					b[i] = max(max(abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])), max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
				}
				//d[i] += sqrt(max((sc[i] - dx) * (sc[i] - dx), (sc[i] - dy) * (sc[i] - dy)));
				//d[i] += 0.5f * ((abs(sc[i] - sc[i - 1]), abs(sc[i] - sc[i + 1])) + max(abs(sc[i] - sm[i]), abs(sc[i] - sp[i])));
			}*/
			for (int i = roi.x; i < roi.x + roi.width; i += 8)
			{
				const __m256 mscp = _mm256_loadu_ps(sc + i - 1);
				const __m256 mscc = _mm256_loadu_ps(sc + i - 0);
				const __m256 mscm = _mm256_loadu_ps(sc + i + 1);
				const __m256 msp = _mm256_loadu_ps(sp + i + 0);
				const __m256 msm = _mm256_loadu_ps(sm + i + 0);
				_mm256_store_ps(b + i - roi.x, _mm256_max_ps(
					_mm256_max_ps(_mm256_absdiff_ps(mscc, mscp), _mm256_absdiff_ps(mscc, mscm)),
					_mm256_max_ps(_mm256_absdiff_ps(mscc, msp), _mm256_absdiff_ps(mscc, msm))
				));
			}
		}

		for (int l = 0; l < scale; l++)
		{
			buff.at<float>(l, 0) = 0.f;
			buff.at<float>(l, roi.width - 1) = 0.f;
		}
		for (int i = 0; i < roi.width; i += scale)
		{
			int idx = 0;
			float gmax = 0.f;
			const float* b = buff.ptr<float>(0, i);
			for (int l = 0; l < scale; l++)
			{
				for (int k = 0; k < scale; k++)
				{
					if (gmax < b[k])
					{
						gmax = b[k];
						idx = l * w + k;
					}
				}
				b += roi.width;
			}

			idx += (j * w + i) + roi.x;
			for (int c = 0; c < channels; c++)
			{
				dptr[c][sample] = src[c].at<float>(idx);
			}
			sample++;
		}
	}
}

void gradientMaxSampling(const vector<Mat>& src, Mat& dest, int scale, Rect roi)
{
	if (scale == 2) gradientMaxSampling_<2>(src, dest, roi);
	else if (scale == 3) gradientMaxSampling_<3>(src, dest, roi);
	else if (scale == 4) gradientMaxSampling_<4>(src, dest, roi);
	else if (scale == 5) gradientMaxSampling_<5>(src, dest, roi);
	else if (scale == 6) gradientMaxSampling_<6>(src, dest, roi);
	else if (scale == 7) gradientMaxSampling_<7>(src, dest, roi);
	else if (scale == 8) gradientMaxSampling_<8>(src, dest, roi);
	else if (scale == 9) gradientMaxSampling_<9>(src, dest, roi);
	else gradientMaxSampling_(src, dest, scale, roi);
}