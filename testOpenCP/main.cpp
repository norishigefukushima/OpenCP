#include <opencp.hpp>
#include "test.hpp"
using namespace std;
using namespace cv;
using namespace cp;

void stream2(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	for (int i = 0; i < size; i += 8)
	{
		_mm256_stream_ps(bp + i, _mm256_stream_load_ps(ap + i));
	}
}

void stream3(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	for (int i = 0; i < size; i += 16)
	{
		_mm256_stream_ps(bp + i + 0, _mm256_stream_load_ps(ap + i + 0));
		_mm256_stream_ps(bp + i + 8, _mm256_stream_load_ps(ap + i + 8));
	}
}

void stream4(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	for (int i = 0; i < size; i += 32)
	{
		_mm256_stream_ps(bp + i + 0, _mm256_stream_load_ps(ap + i + 0));
		_mm256_stream_ps(bp + i + 8, _mm256_stream_load_ps(ap + i + 8));
		_mm256_stream_ps(bp + i + 16, _mm256_stream_load_ps(ap + i + 16));
		_mm256_stream_ps(bp + i + 24, _mm256_stream_load_ps(ap + i + 24));
	}
}

void stream5(Mat& a, Mat& b)
{
	const int size = a.size().area();
	const float* ap = a.ptr<float>();
	float* bp = b.ptr<float>();
	for (int i = 0; i < size; i += 64)
	{
		_mm256_stream_ps(bp + 0, _mm256_stream_load_ps(ap + 0));
		_mm256_stream_ps(bp + 8, _mm256_stream_load_ps(ap + 8));
		_mm256_stream_ps(bp + 16, _mm256_stream_load_ps(ap + 16));
		_mm256_stream_ps(bp + 24, _mm256_stream_load_ps(ap + 24));
		_mm256_stream_ps(bp + 32, _mm256_stream_load_ps(ap + 32));
		_mm256_stream_ps(bp + 40, _mm256_stream_load_ps(ap + 40));
		_mm256_stream_ps(bp + 48, _mm256_stream_load_ps(ap + 48));
		_mm256_stream_ps(bp + 56, _mm256_stream_load_ps(ap + 56));
		ap += 64;
		bp += 64;
	}
}

void stream51(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
#pragma omp parallel for 
	for (int i = 0; i < size; i += 32)
	{
		_mm256_stream_ps(bp + i + 0, _mm256_stream_load_ps(ap + i + 0));
		_mm256_stream_ps(bp + i + 8, _mm256_stream_load_ps(ap + i + 8));
		_mm256_stream_ps(bp + i + 16, _mm256_stream_load_ps(ap + i + 16));
		_mm256_stream_ps(bp + i + 24, _mm256_stream_load_ps(ap + i + 24));
	}
}

void stream52(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < size; i += 32)
	{
		_mm256_stream_ps(bp + i + 0, _mm256_stream_load_ps(ap + i + 0));
		_mm256_stream_ps(bp + i + 8, _mm256_stream_load_ps(ap + i + 8));
		_mm256_stream_ps(bp + i + 16, _mm256_stream_load_ps(ap + i + 16));
		_mm256_stream_ps(bp + i + 24, _mm256_stream_load_ps(ap + i + 24));
	}
}

void stream6(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
#pragma omp parallel for
	for (int i = 0; i < size; i += 32)
	{
		__m256 v0 = _mm256_stream_load_ps(ap + i + 0);
		__m256 v1 = _mm256_stream_load_ps(ap + i + 8);
		__m256 v2 = _mm256_stream_load_ps(ap + i + 16);
		__m256 v3 = _mm256_stream_load_ps(ap + i + 24);
		_mm256_stream_ps(bp + i + 0, v0);
		_mm256_stream_ps(bp + i + 8, v1);
		_mm256_stream_ps(bp + i + 16, v2);
		_mm256_stream_ps(bp + i + 24, v3);
	}
}

void stream7(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	const int thread_max = omp_get_max_threads();
	const int step = size / thread_max;
#pragma omp parallel for
	for (int t = 0; t < thread_max; t++)
	{
		//const int idx = omp_get_thread_num();
		const int st = (t + 0) * step;
		const int ed = (t + 1) * step;
		for (int i = st; i < ed; i += 32)
		{
			__m256 v0 = _mm256_stream_load_ps(ap + i + 0);
			__m256 v1 = _mm256_stream_load_ps(ap + i + 8);
			__m256 v2 = _mm256_stream_load_ps(ap + i + 16);
			__m256 v3 = _mm256_stream_load_ps(ap + i + 24);
			_mm256_stream_ps(bp + i + 0, v0);
			_mm256_stream_ps(bp + i + 8, v1);
			_mm256_stream_ps(bp + i + 16, v2);
			_mm256_stream_ps(bp + i + 24, v3);
		}
	}
}

void stream8(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	const int thread_max = omp_get_max_threads();
	const int step = size / thread_max;
#pragma omp parallel for
	for (int t = 0; t < thread_max; t++)
	{
		//const int idx = omp_get_thread_num();
		const int st = (t + 0) * step;
		const int ed = (t + 1) * step;
		__m256 v0 = _mm256_stream_load_ps(ap + 0);
		__m256 v1 = _mm256_stream_load_ps(ap + 8);
		__m256 v2 = _mm256_stream_load_ps(ap + 16);
		__m256 v3 = _mm256_stream_load_ps(ap + 24);
		for (int i = st; i < ed - 32; i += 32)
		{
			_mm256_stream_ps(bp + i + 0, v0);
			_mm256_stream_ps(bp + i + 8, v1);
			_mm256_stream_ps(bp + i + 16, v2);
			_mm256_stream_ps(bp + i + 24, v3);
			v0 = _mm256_stream_load_ps(ap + i + 32);
			v1 = _mm256_stream_load_ps(ap + i + 40);
			v2 = _mm256_stream_load_ps(ap + i + 48);
			v3 = _mm256_stream_load_ps(ap + i + 56);
		}
		_mm256_stream_ps(bp + ed - 32 + 0, v0);
		_mm256_stream_ps(bp + ed - 32 + 8, v1);
		_mm256_stream_ps(bp + ed - 32 + 16, v2);
		_mm256_stream_ps(bp + ed - 32 + 24, v3);
	}
}

void stream9(Mat& a, Mat& b)
{
	const int size = a.size().area();
	float* ap = a.ptr < float>();
	float* bp = b.ptr < float>();
	const int thread_max = omp_get_max_threads();
	const int step = size / thread_max;
#pragma omp parallel for
	for (int t = 0; t < thread_max; t++)
	{
		//const int idx = omp_get_thread_num();
		const int st = (t + 0) * step;
		const int ed = (t + 1) * step;
		__m256 v0 = _mm256_stream_load_ps(ap + 0);
		__m256 v1 = _mm256_stream_load_ps(ap + 8);
		__m256 v2 = _mm256_stream_load_ps(ap + 16);
		__m256 v3 = _mm256_stream_load_ps(ap + 24);
		__m256 v4 = _mm256_stream_load_ps(ap + 32);
		__m256 v5 = _mm256_stream_load_ps(ap + 40);
		__m256 v6 = _mm256_stream_load_ps(ap + 48);
		__m256 v7 = _mm256_stream_load_ps(ap + 56);
		for (int i = st; i < ed - 64; i += 64)
		{
			_mm256_stream_ps(bp + i + 0, v0);
			_mm256_stream_ps(bp + i + 8, v1);
			_mm256_stream_ps(bp + i + 16, v2);
			_mm256_stream_ps(bp + i + 24, v3);
			_mm256_stream_ps(bp + i + 32, v4);
			_mm256_stream_ps(bp + i + 40, v5);
			_mm256_stream_ps(bp + i + 48, v6);
			_mm256_stream_ps(bp + i + 56, v7);
			v0 = _mm256_stream_load_ps(ap + i + 32);
			v1 = _mm256_stream_load_ps(ap + i + 40);
			v2 = _mm256_stream_load_ps(ap + i + 48);
			v3 = _mm256_stream_load_ps(ap + i + 56);
			v4 = _mm256_stream_load_ps(ap + i + 32);
			v5 = _mm256_stream_load_ps(ap + i + 40);
			v6 = _mm256_stream_load_ps(ap + i + 48);
			v7 = _mm256_stream_load_ps(ap + i + 56);
		}
		_mm256_stream_ps(bp + ed - 32 + 0, v0);
		_mm256_stream_ps(bp + ed - 32 + 8, v1);
		_mm256_stream_ps(bp + ed - 32 + 16, v2);
		_mm256_stream_ps(bp + ed - 32 + 24, v3);
		_mm256_stream_ps(bp + ed - 32 + 32, v4);
		_mm256_stream_ps(bp + ed - 32 + 40, v5);
		_mm256_stream_ps(bp + ed - 32 + 48, v6);
		_mm256_stream_ps(bp + ed - 32 + 56, v7);
	}
}

void streamSet(Mat& a, float val, const bool isParallel, const int unroll)
{
	const __m256 mv = _mm256_set1_ps(val);
	const int size = a.size().area();
	float* ap = a.ptr<float>();
	if (isParallel)
	{
		if (unroll == 1)
		{
#pragma omp parallel for
			for (int i = 0; i < size; i += 8)
			{
				_mm256_stream_ps(ap + i + 0, mv);
			}
		}
		if (unroll == 2)
		{
#pragma omp parallel for
			for (int i = 0; i < size; i += 16)
			{
				_mm256_stream_ps(ap + i + 0, mv);
				_mm256_stream_ps(ap + i + 8, mv);
			}
		}
		if (unroll == 4)
		{
#pragma omp parallel for
			for (int i = 0; i < size; i += 32)
			{
				_mm256_stream_ps(ap + i + 0, mv);
				_mm256_stream_ps(ap + i + 8, mv);
				_mm256_stream_ps(ap + i + 16, mv);
				_mm256_stream_ps(ap + i + 24, mv);
			}
		}
		if (unroll == 8)
		{
			const int thread_max = omp_get_max_threads();
			const int step = size / thread_max;
#pragma omp parallel for schedule (static, 1)
			for (int t = 0; t < thread_max; t++)
			{
				float* aptr = ap + t * step;
				for (int i = 0; i < step; i += 32)
				{
					_mm256_stream_ps(aptr + 0, mv);
					_mm256_stream_ps(aptr + 8, mv);
					_mm256_stream_ps(aptr + 16, mv);
					_mm256_stream_ps(aptr + 24, mv);
					aptr += 32;
				}
			}
			/*
#pragma omp parallel for
			for (int i = 0; i < size; i += 64)
			{
				_mm256_stream_ps(ap + i + 0, mv);
				_mm256_stream_ps(ap + i + 8, mv);
				_mm256_stream_ps(ap + i + 16, mv);
				_mm256_stream_ps(ap + i + 24, mv);
				_mm256_stream_ps(ap + i + 32, mv);
				_mm256_stream_ps(ap + i + 40, mv);
				_mm256_stream_ps(ap + i + 48, mv);
				_mm256_stream_ps(ap + i + 56, mv);
			}
			*/
		}
	}
	else
	{
		if (unroll == 1)
		{
			for (int i = 0; i < size; i += 8)
			{
				_mm256_stream_ps(ap + 0, mv);
				ap += 8;
			}
		}
		if (unroll == 2)
		{
			for (int i = 0; i < size; i += 16)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				ap += 16;
			}
		}
		if (unroll == 4)
		{
			for (int i = 0; i < size; i += 32)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				_mm256_stream_ps(ap + 16, mv);
				_mm256_stream_ps(ap + 24, mv);
				ap += 32;
			}
		}
		if (unroll == 8)
		{
			for (int i = 0; i < size; i += 64)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				_mm256_stream_ps(ap + 16, mv);
				_mm256_stream_ps(ap + 24, mv);
				_mm256_stream_ps(ap + 32, mv);
				_mm256_stream_ps(ap + 40, mv);
				_mm256_stream_ps(ap + 48, mv);
				_mm256_stream_ps(ap + 56, mv);
				ap += 64;
			}
		}
	}
}


float streamSum(Mat& a, float val, const bool isParallel, const int unroll)
{
	float ret=0.f;
	const int thread_max = omp_get_max_threads();
	AutoBuffer<__m256> acc(thread_max);
	for (int i = 0; i < thread_max; i++) acc[i] = _mm256_setzero_ps();

	const __m256 mv = _mm256_set1_ps(val);
	const int size = a.size().area();
	const int step = size / thread_max;
	float* ap = a.ptr<float>();
	if (isParallel)
	{
		if (unroll == 1)
		{
			//_mm_clflush(ap);
#pragma omp parallel for // schedule  (static, 1)
			for (int t = 0; t < thread_max; t++)
			{
				float* aptr = ap + t * step;
				for (int i = 0; i < step; i += 8)
				{
					acc[t] = _mm256_add_ps(acc[t], _mm256_stream_load_ps(aptr));
					aptr += 8;
				}
			}
		}
		if (unroll == 2)
		{
#pragma omp parallel for // schedule  (static, 1)
			for (int t = 0; t < thread_max; t++)
			{
				float* aptr = ap + t * step;
				for (int i = 0; i < step; i += 16)
				{
					acc[t] = _mm256_add_ps(acc[t], _mm256_load_ps(aptr+0));
					acc[t] = _mm256_add_ps(acc[t], _mm256_load_ps(aptr+8));
					aptr += 16;
				}
			}
		}
		if (unroll == 4)
		{
#pragma omp parallel for  schedule  (static, 1)
			for (int t = 0; t < thread_max; t++)
			{
				float* aptr = ap + t * step;
				for (int i = 0; i < step; i += 32)
				{
					acc[t] = _mm256_add_ps(acc[t], _mm256_stream_load_ps(aptr + 0));
					acc[t] = _mm256_add_ps(acc[t], _mm256_stream_load_ps(aptr + 8));
					acc[t] = _mm256_add_ps(acc[t], _mm256_stream_load_ps(aptr +16));
					acc[t] = _mm256_add_ps(acc[t], _mm256_stream_load_ps(aptr +24));
					aptr += 32;
				}
			}
		}
		if (unroll == 8)
		{
			const int thread_max = omp_get_max_threads();
			const int step = size / thread_max;
#pragma omp parallel for schedule (static, 1)
			for (int t = 0; t < thread_max; t++)
			{
				float* aptr = ap + t * step;
				for (int i = 0; i < step; i += 32)
				{
					_mm256_stream_ps(aptr + 0, mv);
					_mm256_stream_ps(aptr + 8, mv);
					_mm256_stream_ps(aptr + 16, mv);
					_mm256_stream_ps(aptr + 24, mv);
					aptr += 32;
				}
			}
			/*
#pragma omp parallel for
			for (int i = 0; i < size; i += 64)
			{
				_mm256_stream_ps(ap + i + 0, mv);
				_mm256_stream_ps(ap + i + 8, mv);
				_mm256_stream_ps(ap + i + 16, mv);
				_mm256_stream_ps(ap + i + 24, mv);
				_mm256_stream_ps(ap + i + 32, mv);
				_mm256_stream_ps(ap + i + 40, mv);
				_mm256_stream_ps(ap + i + 48, mv);
				_mm256_stream_ps(ap + i + 56, mv);
			}
			*/
		}
	}
	else
	{
		if (unroll == 1)
		{
			for (int i = 0; i < size; i += 8)
			{
				_mm256_stream_ps(ap + 0, mv);
				ap += 8;
			}
		}
		if (unroll == 2)
		{
			for (int i = 0; i < size; i += 16)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				ap += 16;
			}
		}
		if (unroll == 4)
		{
			for (int i = 0; i < size; i += 32)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				_mm256_stream_ps(ap + 16, mv);
				_mm256_stream_ps(ap + 24, mv);
				ap += 32;
			}
		}
		if (unroll == 8)
		{
			for (int i = 0; i < size; i += 64)
			{
				_mm256_stream_ps(ap + 0, mv);
				_mm256_stream_ps(ap + 8, mv);
				_mm256_stream_ps(ap + 16, mv);
				_mm256_stream_ps(ap + 24, mv);
				_mm256_stream_ps(ap + 32, mv);
				_mm256_stream_ps(ap + 40, mv);
				_mm256_stream_ps(ap + 48, mv);
				_mm256_stream_ps(ap + 56, mv);
				ap += 64;
			}
		}
	}

	for (int t = 1; t < thread_max; t++)
	{
		acc[0], _mm256_add_ps(acc[0], acc[t]);
	}
	ret = _mm256_reduceadd_ps(acc[0]);
	return ret;
}
void testStreamCopy()
{
	string wname = "stream";
	cp::ConsoleImage ci(Size(700, 300));
	//namedWindow(wname);
	int iteration = 5;
	createTrackbar("iteration", "", &iteration, 1000);
	int size = 3;  createTrackbar("size", "", &size, 8);
	int th = 4;  createTrackbar("thread", "", &th, omp_get_max_threads());

	int key = 0;

	cp::Timer t("", TIME_MSEC, false);
	while (key != 'q')
	{
		const int w = 128 * (int)pow(2, size);
		ci("Size %d x %d", w, w);
		omp_set_num_threads(th);
		const int thread_max = omp_get_max_threads();
		ci("threads %d", thread_max);
		cv::Size imsize = Size(w, w);
		//float* data_a = (float*)_mm_malloc(imsize.area() * sizeof(float), AVX_ALIGN);
		//float* data_b = (float*)_mm_malloc(imsize.area() * sizeof(float), AVX_ALIGN);
		//AutoBuffer<float> data_a(imsize.area());
		//AutoBuffer<float> data_b(imsize.area());
		//Mat a(imsize, CV_32F, data_a);
		//Mat b(imsize, CV_32F, data_b);
		Mat a(imsize, CV_32F);

		auto benchSet = [&](int iteration, bool parallel, int unroll)
		{
			t.clearStat();
			for (int i = 0; i < iteration; i++)
			{
				t.start();
				//streamSet(a, 0.f, parallel, unroll);
				streamSum(a, 0.f, parallel, unroll);
				t.pushLapTime();
			}
		};

		benchSet(iteration, false, 1);
		ci("streamSet S1 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 2);
		ci("streamSet S2 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 4);
		ci("streamSet S4 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, false, 8);
		ci("streamSet S8 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 1);
		ci("streamSet P1 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 2);
		ci("streamSet P2 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 4);
		ci("streamSet P4 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		benchSet(iteration, true, 8);
		ci("streamSet P8 GB/s  %f", a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);



		/*
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			a.copyTo(b);
			t.pushLapTime();
		}
		ci("copy   Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			cp::streamCopy(a, b);
			t.pushLapTime();
		}
		ci("stream  Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream2(a, b);
			t.pushLapTime();
		}
		ci("stream2 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream3(a, b);
			t.pushLapTime();
		}
		ci("stream3 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream4(a, b);
			t.pushLapTime();
		}
		ci("stream4 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream5(a, b);
			t.pushLapTime();
		}
		ci("stream5 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream6(a, b);
			t.pushLapTime();
		}
		ci("stream6 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream7(a, b);
			t.pushLapTime();
		}
		ci("stream7 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream8(a, b);
			t.pushLapTime();
		}
		ci("stream8 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();

		for (int i = 0; i < iteration; i++)
		{
			t.start();
			stream9(a, b);
			t.pushLapTime();
		}
		ci("stream9 Time %f ms, GB/s  %f", t.getLapTimeMedian(), a.size().area() * sizeof(float) / t.getLapTimeMedian() / 1000000.0);
		t.clearStat();
		*/


		ci.show();
		key = waitKey(1);
	}

	//_mm_free(data_a);
	//_mm_free(data_b);
}
int main(int argc, char** argv)
{
	testStreamCopy(); return 0;

	const bool isShowInfo = true;
	if (isShowInfo)
	{
		cout << getInformation() << endl;
		cout << cv::getBuildInformation() << endl;
		cout << getOpenCLInformation() << endl;
		cv::cuda::printCudaDeviceInfo(0);
	}

	//cv::ipp::setUseIPP(false);
	//cv::setUseOptimized(false);


	//webPAnimationTest(); return 0;
	//testSpearmanRankOrderCorrelationCoefficient(); return 0;
	/*__m256i a = _mm256_set_step_epi32(0);
	__m256i b = _mm256_set_step_epi32(8);
	__m256i c = _mm256_set_step_epi32(16);
	__m256i d = _mm256_set_step_epi32(24);

	__m256i w = _mm256_cmpgt_epi32(a, _mm256_set1_epi32(2));
	print_int(w);
	print_int(_mm256_andnot_si256(w, b));

	//print_uchar(_mm256_packus_epi16(_mm256_packus_epi32(a, b), _mm256_packus_epi32(c, d)));
	return 0;*/

	//testUnnormalizedBilateralFilter(); return 0;
	//testMultiScaleFilter(); return 0;

	//testIsSame(); return 0;


	//detailTest(); return 0;
#pragma region setup
	//Mat img = imread("img/lenna.png");
	Mat img = imread("img/Kodak/kodim07.png");
	Mat gra = imread("img/Kodak/kodim07.png", 0);
	testSpatialFilter(gra);
	//rangeBlurFilterRef(aa, t0, 5, 3);
	//rangeBlurFilter(aa, t1, 5, 3);
	//guiAlphaBlend(convert(t0,CV_8U), convert(t1,CV_8U));
	//Mat img = imread("img/cameraman.png",0);
	//Mat img = imread("img/barbara.png", 0);
	//filter2DTest(img); return 0;

#pragma endregion


#pragma region core
	//guiPixelizationTest();
	//testStreamConvert8U(); return 0;
	//testKMeans(img); return 0;
	//testTiling(img); return 0;
	//copyMakeBorderTest(img); return 0;
	//testSplitMerge(img); return 0;
	//consoleImageTest(); return 0;
	//testConcat(); return 0;
	//testsimd(); return 0;

	//testHistogram(); return 0;
	//testPlot(); return 0;
	//testPlot2D(); return 0;

	//guiHazeRemoveTest();

	//testCropZoom(); return 0;
	//testAddNoise(img); return 0;
	//testLocalPSNR(img); return 0;
	//testPSNR(img); return 0;
	//resize(img, a, Size(513, 513));
	//testHistgram(img);
	//testRGBHistogram();
	//testRGBHistogram2();
	//testTimer(img);
	//testMatInfo(); return 0;
	testStat(); return 0;
	//testDestinationTimePrediction(img); return 0;
	//testAlphaBlend(left, right);
	//testAlphaBlendMask(left, right);
	//guiDissolveSlide(left, dmap);
	//guiLocalDiffHistogram(img);
	//guiContrast(img);
	//guiContrast(guiCropZoom(img));
	//testVideoSubtitle();
	//guiWindowFunction();
#pragma endregion

#pragma region imgproc
	//guiCvtColorPCATest(); return 0;
#pragma endregion

#pragma region stereo
	//testStereoBase(); return 0;
	//testCVStereoBM(); return 0;
	//testCVStereoSGBM(); return 0;
#pragma endregion

#pragma region filter
	//testGuidedImageFilter(Mat(), Mat()); return 0;
	highDimentionalGaussianFilterTest(img); return 0;
	//highDimentionalGaussianFilterHSITest(); return 0;
	//guiDenoiseTest(img);
	//testWeightedHistogramFilterDisparity(); return 0;
	//testWeightedHistogramFilter();return 0;
#pragma endregion 

	//guiUpsampleTest(img); return 0;
	guiDomainTransformFilterTest(img);
	//guiMedianFilterTest(img);
	//VisualizeDenormalKernel vdk;
	//vdk.run(img);
	//return 0;
	//VizKernel vk;
	//vk.run(img, 2);


	//guiShift(left,right); return 0;
	//
	//iirGuidedFilterTest2(img); return 0;
	//iirGuidedFilterTest1(dmap, left); return 0;
	//iirGuidedFilterTest(); return 0;
	//iirGuidedFilterTest(left); return 0;
	//fitPlaneTest(); return 0;
	//guiWeightMapTest(); return 0;


	//guiGeightedJointBilateralFilterTest();
	//guiHazeRemoveTest();
	//Mat ff3 = imread("img/pixelart/ff3.png");

	Mat src = imread("img/lenna.png");

	//Mat src = imread("img/Kodak/kodim07.png",0);
	//guiIterativeBackProjectionTest(src);
	//Mat src = imread("img/Kodak/kodim15.png",0);

	//Mat src = imread("img/cave-flash.png");
	//Mat src = imread("img/feathering/toy.png");
	//Mat src = imread("Clipboard01.png");

	//timeGuidedFilterTest(src);
	//Mat src = imread("img/flower.png");
	//Mat src = imread("img/teddy_disp1.png");
	//Mat src_ = imread("img/stereo/Art/view1.png",0);
	//	Mat src;
	//	copyMakeBorder(src_,src,0,1,0,1,BORDER_REPLICATE);

	//Mat src = imread("img/lenna.png", 0);

	//Mat src = imread("img/stereo/Dolls/view1.png");
	//guiDenoiseTest(src);
	guiBilateralFilterTest(src);
	Mat ref = imread("img/stereo/Dolls/view6.png");
	//guiColorCorrectionTest(src, ref); return 0;
	//Mat src = imread("img/flower.png");
	//guiAnalysisImage(src);
	Mat dst = src.clone();
	//paralleldenoise(src, dst, 5);
	//Mat disp = imread("img/stereo/Dolls/disp1.png", 0);
	//	Mat src;
	Mat dest;

	//guiCrossBasedLocalFilter(src); return 0;


	//eraseBoundary(src,10);
	//	resize(src,mega,Size(1024,1024));
	//resize(src,mega,Size(640,480));

	//guiDualBilateralFilterTest(src,disp);
	//guiGausianFilterTest(src); return 0;

	//guiCoherenceEnhancingShockFilter(src, dest);

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//guiDisparityPlaneFitSLICTest(src, ref, disp); return 0;
	//getPSNRRealtimeO1BilateralFilterKodak();
	//guiRealtimeO1BilateralFilterTest(src); return 0;

	Mat flashImg = imread("img/flash/cave-flash.png");
	Mat noflashImg = imread("img/flash/cave-noflash.png");
	Mat noflashImgGray; cvtColor(noflashImg, noflashImgGray, COLOR_BGR2GRAY);
	Mat flashImgGray; cvtColor(flashImg, flashImgGray, COLOR_BGR2GRAY);
	Mat fmega, nmega;
	resize(flashImgGray, fmega, Size(1024, 1024));
	resize(noflashImg, nmega, Size(1024, 1024));

	//guiEdgePresevingFilterOpenCV(src);
	//guiSLICTest(src);


	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImgGray); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImgGray, flashImg); return 0;
	//guiJointRealtimeO1BilateralFilterTest(noflashImg, flashImg); return 0;

	//guiWeightedHistogramFilterTest(noflashImgGray, flashImg); return 0;
	//guiRealtimeO1BilateralFilterTest(noflashImgGray); return 0;
	//guiRealtimeO1BilateralFilterTest(src); return 0;
	//guiDMFTest(nmega, nmega, fmega); return 0;
	//guiGausianFilterTest(src); return 0;


	//guiAlphaBlend(ff3,ff3);
	//guiJointNearestFilterTest(ff3);
	//guiViewSynthesis();

	//guiSeparableBilateralFilterTest(src);
	//guiBilateralFilterSPTest(mega);
	//guiRecursiveBilateralFilterTest(mega);
	//fftTest(src);

	//Mat feather = imread("img/feathering/toy-mask.png");
	//Mat guide = imread("img/feathering/toy.png");
	//timeBirateralTest(mega);
	//Mat disparity = imread("img/teddy_disp1.png", 0);
	//guiJointBirateralFilterTest(noflash,flash);
	//guiBinalyWeightedRangeFilterTest(disparity);
	//guiCodingDistortionRemoveTest(disparity);
	//guiJointBinalyWeightedRangeFilterTest(noflash,flash);

	//guiNonLocalMeansTest(src);

	//application 
	//guiDetailEnhancement(src);
	//guiDomainTransformFilterTest(mega);
	return 0;
}