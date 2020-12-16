#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void calcCovImageC3_OpenCV(const cv::Mat src, cv::Mat& cov, cv::Mat& mu);
void calcCovImage32FC3_scalar(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu);
void calcCovImage32FC3_scalar2(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu);
void calcCovImage32FC3_nonSplit(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu);
void calcCovImage32FC3(const cv::Mat* src, std::vector<float>& cov, std::vector<float>& mu);

void calcCovImageC3_OpenCV(const cv::Mat src, cv::Mat& cov, cv::Mat& mu)
{
	const Mat temp = src.reshape(1, src.rows*src.cols);

	calcCovarMatrix(temp, cov, mu, COVAR_NORMAL | COVAR_SCALE | COVAR_ROWS, CV_32F);
}

void calcCovImage32FC3_scalar(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu)
{
	const int ch = 3;
	const int norm = src.rows*src.cols;

	mu = { 0.f, 0.f, 0.f };
	cov = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

	//calc means
	float mu0 = 0.f;
	float mu1 = 0.f;
	float mu2 = 0.f;
#pragma omp parallel for reduction(+: mu0, mu1, mu2)
	for (int j = 0; j < src.rows; j++)
	{
		const float* sp = src.ptr<float>(j);
		for (int i = 0; i < src.cols; i++)
		{
			mu0 += *(sp + 0);
			mu1 += *(sp + 1);
			mu2 += *(sp + 2);
			sp += ch;
		}
	}
	mu[0] = mu0 / norm;
	mu[1] = mu1 / norm;
	mu[2] = mu2 / norm;

	//calc variance
	float cov0 = 0.f;
	float cov1 = 0.f;
	float cov2 = 0.f;
	float cov3 = 0.f;
	float cov4 = 0.f;
	float cov5 = 0.f;
#pragma omp parallel for reduction(+: cov0, cov1, cov2, cov3, cov4, cov5)
	for (int j = 0; j < src.rows; j++)
	{
		const float* sp = src.ptr<float>(j);
		for (int i = 0; i < src.cols; i++)
		{
			const float tempB = *(sp + 0) - mu[0];
			const float tempG = *(sp + 1) - mu[1];
			const float tempR = *(sp + 2) - mu[2];
			cov0 += tempB * tempB;
			cov1 += tempB * tempG;
			cov2 += tempB * tempR;
			cov3 += tempG * tempG;
			cov4 += tempG * tempR;
			cov5 += tempR * tempR;
			sp += ch;
		}
	}
	cov[0] = cov0 / norm;
	cov[1] = cov1 / norm;
	cov[2] = cov2 / norm;
	cov[3] = cov3 / norm;
	cov[4] = cov4 / norm;
	cov[5] = cov5 / norm;
}

void calcCovImage32FC3_scalar2(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu)
{
	//TODO: ?U??
	const int ch = 3;
	const int norm = src.rows*src.cols;

	mu = { 0.f, 0.f, 0.f };
	cov = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

	float mu0 = 0.f;
	float mu1 = 0.f;
	float mu2 = 0.f;

	float cov0 = 0.f;
	float cov1 = 0.f;
	float cov2 = 0.f;
	float cov3 = 0.f;
	float cov4 = 0.f;
	float cov5 = 0.f;
#pragma omp parallel for reduction(+: mu0, mu1, mu2, cov0, cov1, cov2, cov3, cov4, cov5)
	for (int j = 0; j < src.rows; j++)
	{
		const float* sp = src.ptr<float>(j);
		for (int i = 0; i < src.cols; i++)
		{
			mu0 += *(sp + 0);
			mu1 += *(sp + 1);
			mu2 += *(sp + 2);
			cov0 += *(sp + 0) * *(sp + 0);
			cov1 += *(sp + 0) * *(sp + 1);
			cov2 += *(sp + 0) * *(sp + 2);
			cov3 += *(sp + 1) * *(sp + 1);
			cov4 += *(sp + 1) * *(sp + 2);
			cov5 += *(sp + 2) * *(sp + 2);
			sp += ch;
		}
	}
	cov[0] = (cov0 - mu[0] * mu[0]) / (norm * norm);
	cov[1] = (cov1 - mu[0] * mu[1]) / (norm * norm);
	cov[2] = (cov2 - mu[0] * mu[2]) / (norm * norm);
	cov[3] = (cov3 - mu[1] * mu[1]) / (norm * norm);
	cov[4] = (cov4 - mu[1] * mu[2]) / (norm * norm);
	cov[5] = (cov5 - mu[2] * mu[2]) / (norm * norm);

	mu[0] = mu0 / norm;
	mu[1] = mu1 / norm;
	mu[2] = mu2 / norm;
}

class CalcMeanImpl_nonSplit : public ParallelLoopBody
{
private:
	const int ch = 3;
	const int vecNum = 8;
	Mat src;
	__m256* bSum = nullptr;
	__m256* gSum = nullptr;
	__m256* rSum = nullptr;
	int nthreads = 1;
	vector<float>* mu;
	int norm = 1;

public:
	CalcMeanImpl_nonSplit(const Mat src, vector<float>* mu)
	{
		this->src = src;
		nthreads = getNumThreads() - 1;
		this->mu = mu;
		norm = src.rows * src.cols;

		bSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		gSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		rSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
	}

	~CalcMeanImpl_nonSplit()
	{
		const float* sp = src.ptr<float>(0);
		//residual
		for (int j = norm - norm % vecNum; j < norm; j++)
		{
			const int jj = j * ch;
			bSum[0].m256_f32[0] += *(sp + jj + 0);
			gSum[0].m256_f32[0] += *(sp + jj + 1);
			rSum[0].m256_f32[0] += *(sp + jj + 2);
		}

		if (nthreads > 1)
		{
			for (int i = 1; i < nthreads; i++)
			{
				bSum[0] = _mm256_add_ps(bSum[0], bSum[i]);
				gSum[0] = _mm256_add_ps(gSum[0], gSum[i]);
				rSum[0] = _mm256_add_ps(rSum[0], rSum[i]);
			}
		}

		bSum[0] = _mm256_hadd_ps(bSum[0], bSum[0]);
		bSum[0] = _mm256_hadd_ps(bSum[0], bSum[0]);
		(*mu)[0] = (bSum[0].m256_f32[0] + bSum[0].m256_f32[4]) / norm;

		gSum[0] = _mm256_hadd_ps(gSum[0], gSum[0]);
		gSum[0] = _mm256_hadd_ps(gSum[0], gSum[0]);
		(*mu)[1] = (gSum[0].m256_f32[0] + gSum[0].m256_f32[4]) / norm;

		rSum[0] = _mm256_hadd_ps(rSum[0], rSum[0]);
		rSum[0] = _mm256_hadd_ps(rSum[0], rSum[0]);
		(*mu)[2] = (rSum[0].m256_f32[0] + rSum[0].m256_f32[4]) / norm;

		_mm_free(bSum);
		_mm_free(gSum);
		_mm_free(rSum);
	}

	void operator()(const cv::Range& range) const override
	{
		const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
		bSum[range.start] = _mm256_setzero_ps();
		gSum[range.start] = _mm256_setzero_ps();
		rSum[range.start] = _mm256_setzero_ps();

		const float* sp = src.ptr<float>(0);
		const int step = ((norm / nthreads) / vecNum) * vecNum;
		const int end = (range.end != nthreads) ? step * range.end : norm;
		for (int j = step * range.start; j < end; j += vecNum)
		{
			const int jj = j * ch;
			const __m256 sb = _mm256_i32gather_ps(sp + jj + 0, index, 4);
			bSum[range.start] = _mm256_add_ps(bSum[range.start], sb);
			const __m256 sg = _mm256_i32gather_ps(sp + jj + 1, index, 4);
			gSum[range.start] = _mm256_add_ps(gSum[range.start], sg);
			const __m256 sr = _mm256_i32gather_ps(sp + jj + 2, index, 4);
			rSum[range.start] = _mm256_add_ps(rSum[range.start], sr);
		}
	}
};

class CalcCovImpl_nonSplit : public ParallelLoopBody
{
private:
	const int ch = 3;
	const int vecNum = 8;
	int norm = 1;
	int nthreads = 1;
	Mat src;
	__m256* mcov0 = nullptr;
	__m256* mcov1 = nullptr;
	__m256* mcov2 = nullptr;
	__m256* mcov3 = nullptr;
	__m256* mcov4 = nullptr;
	__m256* mcov5 = nullptr;
	const vector<float>* mu;
	vector<float>* cov;

public:
	CalcCovImpl_nonSplit(const Mat src, const vector<float>* mu, vector<float>* cov)
	{
		this->src = src;
		this->mu = mu;
		this->cov = cov;
		nthreads = getNumThreads() - 1;
		norm = src.rows * src.cols;

		mcov0 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov1 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov2 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov3 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov4 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov5 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
	}

	~CalcCovImpl_nonSplit()
	{
		const float* sp = src.ptr<float>(0);
		//residual
		for (int j = norm - norm % vecNum; j < norm; j++)
		{
			const int jj = j * ch;
			const float tempB = *(sp + jj + 0) - (*mu)[0];
			mcov0[0].m256_f32[0] += tempB * tempB;
			const float tempG = *(sp + jj + 1) - (*mu)[1];
			mcov1[0].m256_f32[0] += tempB * tempG;
			mcov3[0].m256_f32[0] += tempG * tempG;
			const float tempR = *(sp + jj + 2) - (*mu)[2];
			mcov2[0].m256_f32[0] += tempB * tempR;
			mcov4[0].m256_f32[0] += tempG * tempR;
			mcov5[0].m256_f32[0] += tempR * tempR;
		}

		if (nthreads > 1)
		{
			for (int i = 1; i < nthreads; i++)
			{
				mcov0[0] = _mm256_add_ps(mcov0[0], mcov0[i]);
				mcov1[0] = _mm256_add_ps(mcov1[0], mcov1[i]);
				mcov2[0] = _mm256_add_ps(mcov2[0], mcov2[i]);
				mcov3[0] = _mm256_add_ps(mcov3[0], mcov3[i]);
				mcov4[0] = _mm256_add_ps(mcov4[0], mcov4[i]);
				mcov5[0] = _mm256_add_ps(mcov5[0], mcov5[i]);
			}
		}

		mcov0[0] = _mm256_hadd_ps(mcov0[0], mcov0[0]);
		mcov0[0] = _mm256_hadd_ps(mcov0[0], mcov0[0]);
		(*cov)[0] = (mcov0[0].m256_f32[0] + mcov0[0].m256_f32[4]) / norm;

		mcov1[0] = _mm256_hadd_ps(mcov1[0], mcov1[0]);
		mcov1[0] = _mm256_hadd_ps(mcov1[0], mcov1[0]);
		(*cov)[1] = (mcov1[0].m256_f32[0] + mcov1[0].m256_f32[4]) / norm;

		mcov2[0] = _mm256_hadd_ps(mcov2[0], mcov2[0]);
		mcov2[0] = _mm256_hadd_ps(mcov2[0], mcov2[0]);
		(*cov)[2] = (mcov2[0].m256_f32[0] + mcov2[0].m256_f32[4]) / norm;

		mcov3[0] = _mm256_hadd_ps(mcov3[0], mcov3[0]);
		mcov3[0] = _mm256_hadd_ps(mcov3[0], mcov3[0]);
		(*cov)[3] = (mcov3[0].m256_f32[0] + mcov3[0].m256_f32[4]) / norm;

		mcov4[0] = _mm256_hadd_ps(mcov4[0], mcov4[0]);
		mcov4[0] = _mm256_hadd_ps(mcov4[0], mcov4[0]);
		(*cov)[4] = (mcov4[0].m256_f32[0] + mcov4[0].m256_f32[4]) / norm;

		mcov5[0] = _mm256_hadd_ps(mcov5[0], mcov5[0]);
		mcov5[0] = _mm256_hadd_ps(mcov5[0], mcov5[0]);
		(*cov)[5] = (mcov5[0].m256_f32[0] + mcov5[0].m256_f32[4]) / norm;

		_mm_free(mcov0);
		_mm_free(mcov1);
		_mm_free(mcov2);
		_mm_free(mcov3);
		_mm_free(mcov4);
		_mm_free(mcov5);
	}

	void operator()(const cv::Range& range) const override
	{
		const __m256 mmu0 = _mm256_set1_ps((*mu)[0]);
		const __m256 mmu1 = _mm256_set1_ps((*mu)[1]);
		const __m256 mmu2 = _mm256_set1_ps((*mu)[2]);
		const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
		mcov0[range.start] = _mm256_setzero_ps();
		mcov1[range.start] = _mm256_setzero_ps();
		mcov2[range.start] = _mm256_setzero_ps();
		mcov3[range.start] = _mm256_setzero_ps();
		mcov4[range.start] = _mm256_setzero_ps();
		mcov5[range.start] = _mm256_setzero_ps();

		const float* sp = src.ptr<float>(0);
		const int step = ((norm / nthreads) / vecNum) * vecNum;
		const int end = (range.end != nthreads) ? step * range.end : norm;
		for (int j = step * range.start; j < end; j += vecNum)
		{
			const int jj = j * ch;
			const __m256 sb = _mm256_i32gather_ps(sp + jj + 0, index, 4);
			const __m256 tmpB = _mm256_sub_ps(sb, mmu0);
			mcov0[range.start] = _mm256_fmadd_ps(tmpB, tmpB, mcov0[range.start]);

			const __m256 sg = _mm256_i32gather_ps(sp + jj + 1, index, 4);
			const __m256 tmpG = _mm256_sub_ps(sg, mmu1);
			mcov1[range.start] = _mm256_fmadd_ps(tmpB, tmpG, mcov1[range.start]);
			mcov3[range.start] = _mm256_fmadd_ps(tmpG, tmpG, mcov3[range.start]);

			const __m256 sr = _mm256_i32gather_ps(sp + jj + 2, index, 4);
			const __m256 tmpR = _mm256_sub_ps(sr, mmu2);
			mcov2[range.start] = _mm256_fmadd_ps(tmpB, tmpR, mcov2[range.start]);
			mcov4[range.start] = _mm256_fmadd_ps(tmpG, tmpR, mcov4[range.start]);
			mcov5[range.start] = _mm256_fmadd_ps(tmpR, tmpR, mcov5[range.start]);
		}
	}
};

void calcCovImage32FC3_nonSplit(const cv::Mat src, std::vector<float>& cov, std::vector<float>& mu)
{
	mu = { 0.f, 0.f, 0.f };
	cov = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

	{
		CalcMeanImpl_nonSplit cm(src, &mu);
		parallel_for_(Range(0, getNumThreads() - 1), cm, getNumThreads() - 1);
	}
	{
		CalcCovImpl_nonSplit cc(src, &mu, &cov);
		parallel_for_(Range(0, getNumThreads() - 1), cc, getNumThreads() - 1);
	}
}

class CalcMeanImpl : public ParallelLoopBody
{
private:
	const int vecNum = 8;
	const Mat* src;
	__m256* bSum = nullptr;
	__m256* gSum = nullptr;
	__m256* rSum = nullptr;
	int nthreads = 1;
	int step = 0;
	vector<float>* mu;
	int norm = 1;

public:
	CalcMeanImpl(const Mat* src, vector<float>* mu)
	{
		this->src = src;
		nthreads = getNumThreads() - 1;
		this->mu = mu;
		norm = src[0].rows * src[0].cols;
		step = ((norm / nthreads) / vecNum) * vecNum;

		bSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		gSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		rSum = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
	}

	~CalcMeanImpl()
	{
		const float* spb = src[0].ptr<float>(0);
		const float* spg = src[1].ptr<float>(0);
		const float* spr = src[2].ptr<float>(0);
		//residual
		for (int j = norm - norm % vecNum; j < norm; j++)
		{
			bSum[0].m256_f32[0] += *(spb + j);
			gSum[0].m256_f32[0] += *(spg + j);
			rSum[0].m256_f32[0] += *(spr + j);
		}

		if (nthreads > 1)
		{
			for (int i = 1; i < nthreads; i++)
			{
				bSum[0] = _mm256_add_ps(bSum[0], bSum[i]);
				gSum[0] = _mm256_add_ps(gSum[0], gSum[i]);
				rSum[0] = _mm256_add_ps(rSum[0], rSum[i]);
			}
		}

		bSum[0] = _mm256_hadd_ps(bSum[0], bSum[0]);
		bSum[0] = _mm256_hadd_ps(bSum[0], bSum[0]);
		(*mu)[0] = (bSum[0].m256_f32[0] + bSum[0].m256_f32[4]) / norm;

		gSum[0] = _mm256_hadd_ps(gSum[0], gSum[0]);
		gSum[0] = _mm256_hadd_ps(gSum[0], gSum[0]);
		(*mu)[1] = (gSum[0].m256_f32[0] + gSum[0].m256_f32[4]) / norm;

		rSum[0] = _mm256_hadd_ps(rSum[0], rSum[0]);
		rSum[0] = _mm256_hadd_ps(rSum[0], rSum[0]);
		(*mu)[2] = (rSum[0].m256_f32[0] + rSum[0].m256_f32[4]) / norm;

		_mm_free(bSum);
		_mm_free(gSum);
		_mm_free(rSum);
	}

	void operator()(const cv::Range& range) const override
	{
		const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
		bSum[range.start] = _mm256_setzero_ps();
		gSum[range.start] = _mm256_setzero_ps();
		rSum[range.start] = _mm256_setzero_ps();

		const float* spb = src[0].ptr<float>(0);
		const float* spg = src[1].ptr<float>(0);
		const float* spr = src[2].ptr<float>(0);
		const int end = (range.end != nthreads) ? step * range.end : norm;
		for (int j = step * range.start; j < end; j += vecNum)
		{
			const __m256 sb = _mm256_load_ps(spb + j);
			bSum[range.start] = _mm256_add_ps(bSum[range.start], sb);
			const __m256 sg = _mm256_load_ps(spg + j);
			gSum[range.start] = _mm256_add_ps(gSum[range.start], sg);
			const __m256 sr = _mm256_load_ps(spr + j);
			rSum[range.start] = _mm256_add_ps(rSum[range.start], sr);
		}
	}
};

class CalcCovImpl : public ParallelLoopBody
{
private:
	const int ch = 3;
	const int vecNum = 8;
	int norm = 1;
	int nthreads = 1;
	int step = 0;
	const Mat* src;
	__m256* mcov0 = nullptr;
	__m256* mcov1 = nullptr;
	__m256* mcov2 = nullptr;
	__m256* mcov3 = nullptr;
	__m256* mcov4 = nullptr;
	__m256* mcov5 = nullptr;
	const vector<float>* mu;
	vector<float>* cov;

public:
	CalcCovImpl(const Mat* src, const vector<float>* mu, vector<float>* cov)
	{
		this->src = src;
		this->mu = mu;
		this->cov = cov;
		nthreads = getNumThreads() - 1;
		norm = src[0].rows * src[0].cols;
		step = ((norm / nthreads) / vecNum) * vecNum;

		mcov0 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov1 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov2 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov3 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov4 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
		mcov5 = (__m256*)_mm_malloc(sizeof(__m256)*nthreads, 32);
	}

	~CalcCovImpl()
	{
		const float* spb = src[0].ptr<float>(0);
		const float* spg = src[1].ptr<float>(0);
		const float* spr = src[2].ptr<float>(0);
		//residual
		for (int j = norm - norm % vecNum; j < norm; j++)
		{
			const float tempB = *(spb + j) - (*mu)[0];
			mcov0[0].m256_f32[0] += tempB * tempB;
			const float tempG = *(spg + j) - (*mu)[1];
			mcov1[0].m256_f32[0] += tempB * tempG;
			mcov3[0].m256_f32[0] += tempG * tempG;
			const float tempR = *(spr + j) - (*mu)[2];
			mcov2[0].m256_f32[0] += tempB * tempR;
			mcov4[0].m256_f32[0] += tempG * tempR;
			mcov5[0].m256_f32[0] += tempR * tempR;
		}

		if (nthreads > 1)
		{
			for (int i = 1; i < nthreads; i++)
			{
				mcov0[0] = _mm256_add_ps(mcov0[0], mcov0[i]);
				mcov1[0] = _mm256_add_ps(mcov1[0], mcov1[i]);
				mcov2[0] = _mm256_add_ps(mcov2[0], mcov2[i]);
				mcov3[0] = _mm256_add_ps(mcov3[0], mcov3[i]);
				mcov4[0] = _mm256_add_ps(mcov4[0], mcov4[i]);
				mcov5[0] = _mm256_add_ps(mcov5[0], mcov5[i]);
			}
		}

		mcov0[0] = _mm256_hadd_ps(mcov0[0], mcov0[0]);
		mcov0[0] = _mm256_hadd_ps(mcov0[0], mcov0[0]);
		(*cov)[0] = (mcov0[0].m256_f32[0] + mcov0[0].m256_f32[4]) / norm;

		mcov1[0] = _mm256_hadd_ps(mcov1[0], mcov1[0]);
		mcov1[0] = _mm256_hadd_ps(mcov1[0], mcov1[0]);
		(*cov)[1] = (mcov1[0].m256_f32[0] + mcov1[0].m256_f32[4]) / norm;

		mcov2[0] = _mm256_hadd_ps(mcov2[0], mcov2[0]);
		mcov2[0] = _mm256_hadd_ps(mcov2[0], mcov2[0]);
		(*cov)[2] = (mcov2[0].m256_f32[0] + mcov2[0].m256_f32[4]) / norm;

		mcov3[0] = _mm256_hadd_ps(mcov3[0], mcov3[0]);
		mcov3[0] = _mm256_hadd_ps(mcov3[0], mcov3[0]);
		(*cov)[3] = (mcov3[0].m256_f32[0] + mcov3[0].m256_f32[4]) / norm;

		mcov4[0] = _mm256_hadd_ps(mcov4[0], mcov4[0]);
		mcov4[0] = _mm256_hadd_ps(mcov4[0], mcov4[0]);
		(*cov)[4] = (mcov4[0].m256_f32[0] + mcov4[0].m256_f32[4]) / norm;

		mcov5[0] = _mm256_hadd_ps(mcov5[0], mcov5[0]);
		mcov5[0] = _mm256_hadd_ps(mcov5[0], mcov5[0]);
		(*cov)[5] = (mcov5[0].m256_f32[0] + mcov5[0].m256_f32[4]) / norm;

		_mm_free(mcov0);
		_mm_free(mcov1);
		_mm_free(mcov2);
		_mm_free(mcov3);
		_mm_free(mcov4);
		_mm_free(mcov5);
	}

	void operator()(const cv::Range& range) const override
	{
		const __m256 mmu0 = _mm256_set1_ps((*mu)[0]);
		const __m256 mmu1 = _mm256_set1_ps((*mu)[1]);
		const __m256 mmu2 = _mm256_set1_ps((*mu)[2]);
		const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
		mcov0[range.start] = _mm256_setzero_ps();
		mcov1[range.start] = _mm256_setzero_ps();
		mcov2[range.start] = _mm256_setzero_ps();
		mcov3[range.start] = _mm256_setzero_ps();
		mcov4[range.start] = _mm256_setzero_ps();
		mcov5[range.start] = _mm256_setzero_ps();

		const float* spb = src[0].ptr<float>(0);
		const float* spg = src[1].ptr<float>(0);
		const float* spr = src[2].ptr<float>(0);
		const int end = (range.end != nthreads) ? step * range.end : norm;
		for (int j = step * range.start; j < end; j += vecNum)
		{
			const __m256 sb = _mm256_load_ps(spb + j);
			const __m256 tmpB = _mm256_sub_ps(sb, mmu0);
			mcov0[range.start] = _mm256_fmadd_ps(tmpB, tmpB, mcov0[range.start]);

			const __m256 sg = _mm256_load_ps(spg + j);
			const __m256 tmpG = _mm256_sub_ps(sg, mmu1);
			mcov1[range.start] = _mm256_fmadd_ps(tmpB, tmpG, mcov1[range.start]);
			mcov3[range.start] = _mm256_fmadd_ps(tmpG, tmpG, mcov3[range.start]);

			const __m256 sr = _mm256_load_ps(spr + j);
			const __m256 tmpR = _mm256_sub_ps(sr, mmu2);
			mcov2[range.start] = _mm256_fmadd_ps(tmpB, tmpR, mcov2[range.start]);
			mcov4[range.start] = _mm256_fmadd_ps(tmpG, tmpR, mcov4[range.start]);
			mcov5[range.start] = _mm256_fmadd_ps(tmpR, tmpR, mcov5[range.start]);
		}
	}
};

void calcCovImage32FC3(const cv::Mat* src, std::vector<float>& cov, std::vector<float>& mu)
{
	mu = { 0.f, 0.f, 0.f };
	cov = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	{
		CalcMeanImpl cm(src, &mu);
		parallel_for_(Range(0, getNumThreads() - 1), cm, getNumThreads() - 1);
	}
	{
		CalcCovImpl cc(src, &mu, &cov);
		parallel_for_(Range(0, getNumThreads() - 1), cc, getNumThreads() - 1);
	}
}

void testRGBHistogram2()
{
	Mat img = imread("img/lenna.png");
	//Mat img = imread("img/Kodak/kodim06.png");	

	Mat filtered;
	RGBHistogram h3d;
	namedWindow("kmeans");
	int samples = 30;  createTrackbar("ss", "BF", &samples, 200);
	//int sr = 30;  createTrackbar("sr", "BF", &sr, 255);
	int key = 0;
	cv::Mat clusters;
	Mat sample;

	Mat imgf;
	img.convertTo(imgf, CV_32FC3);
	Mat cov;
	Mat mu;
	Mat projImg;
	calcCovImageC3_OpenCV(imgf, cov, mu);
	Vec3f m = Vec3f(mu.at<float>(0), mu.at<float>(1), mu.at<float>(2));
	for (int i = 0; i < imgf.size().area(); i++)
	{
		imgf.at<Vec3f>(i) -= m;
	}
	Mat eigenVal;
	Mat eigenVec;
	eigen(cov, eigenVal, eigenVec);
	transform(imgf, projImg, eigenVec);

	Mat center = Mat::zeros(3, 1, CV_32F);
	h3d.setCenter(center);

	while (key != 'q')
	{
		Mat srcf = projImg.reshape(1, projImg.size().area());
		cv::kmeans(srcf, samples, clusters, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, FLT_EPSILON), 1, cv::KMEANS_PP_CENTERS, sample);
		h3d.push_back(sample);

		Mat src; projImg.convertTo(src, CV_8U);
		h3d.plot(projImg, false);
		h3d.clear();

		key = waitKey(1);
	}
	return;
}

void testRGBHistogram()
{
	Mat img = imread("img/lenna.png");
	//Mat img = imread("img/Kodak/kodim06.png");	

	RGBHistogram h3d;
	namedWindow("BF");
	int ss = 5;  createTrackbar("ss", "BF", &ss, 200);
	int sr = 30;  createTrackbar("sr", "BF", &sr, 255);

	Mat filtered;
	int key = 0;
	while (key != 'q')
	{
		int d = (int)ceil(ss*3.0) * 2 + 1;
		bilateralFilter(img, filtered, d, sr, ss);
		//GaussianBlur(img, filtered, Size(d, d), ss);

		h3d.plot(filtered, false);
		h3d.clear();
		imshow("BF", filtered);
		key = waitKey(1);
	}
	return;
}