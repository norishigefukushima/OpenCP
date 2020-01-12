#pragma once

#include "guidedFilter_Merge_Base.h"

namespace tiling
{
	class guidedFilter_Merge_nonVec
	{
	protected:
		cv::Mat src;
		cv::Mat guide;
		cv::Mat& dest;
		int r;
		float eps;
		int parallelType;

		cv::Mat a;
		cv::Mat b;

		std::vector<cv::Mat> vI;
		std::vector<cv::Mat> va;

		std::vector<cv::Mat> temp;

		virtual void filter_Guide1(cv::Mat& input, cv::Mat& output);
		virtual void filter_Guide3(cv::Mat& input, cv::Mat& output);

	public:
		guidedFilter_Merge_nonVec(cv::Mat _src, cv::Mat _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType);
		~guidedFilter_Merge_nonVec() {};
		virtual void init();
		virtual void filter();
	};

	//class guidedFilter_Merge_SSE : public guidedFilter_Merge_nonVec
	//{
	//private:
	//	void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
	//	void filter_Guide3(cv::Mat& input, cv::Mat& output) override;
	//public:
	//	guidedFilter_Merge_SSE(cv::Mat _src, cv::Mat _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
	//		: guidedFilter_Merge_nonVec(_src, _guide, _dest, _r, _eps, _parallelType) {};
	//};

	class guidedFilter_Merge_AVX : public guidedFilter_Merge_nonVec
	{
	private:
		void filter_Guide1(cv::Mat& input, cv::Mat& output) override;
		void filter_Guide3(cv::Mat& input, cv::Mat& output) override;
	public:
		guidedFilter_Merge_AVX(cv::Mat _src, cv::Mat _guide, cv::Mat& _dest, int _r, float _eps, int _parallelType)
			: guidedFilter_Merge_nonVec(_src, _guide, _dest, _r, _eps, _parallelType) {};
	};



	/* --- Guide1 --- */
	struct RowSumFilter_Ip2ab_Guide1 : public RowSumFilter_base
	{
	protected:
		cv::Mat& p;
		cv::Mat& I;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		RowSumFilter_Ip2ab_Guide1(cv::Mat& _p, cv::Mat& _I, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
			: p(_p), I(_I), RowSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = p.rows;
			img_col = p.cols;
		}
	};

	struct ColumnSumFilter_Ip2ab_Guide1_nonVec : public ColumnSumFilter_base
	{
	protected:
		cv::Mat& a;
		cv::Mat& b;
		float eps;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_Ip2ab_Guide1_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
			: a(_a), b(_b), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = a.rows;
			img_col = a.cols;
			step = a.cols;
		}

		void filter() override
		{
			if (parallelType == ParallelTypes::NAIVE)
			{
				filter_naive_impl();
			}
			else if (parallelType == ParallelTypes::OMP)
			{
				filter_omp_impl();
			}
			else if (parallelType == ParallelTypes::PARALLEL_FOR_)
			{
				cv::parallel_for_(cv::Range(r, img_col - r), *this, cv::getNumThreads() - 1);
			}
			else
			{

			}
		}
	};

	//struct ColumnSumFilter_Ip2ab_Guide1_SSE : public ColumnSumFilter_Ip2ab_Guide1_nonVec
	//{
	//private:
	//	const __m128 mDiv = _mm_set1_ps(div);
	//	const __m128 mEps = _mm_set1_ps(eps);
	//	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	//	void filter_naive_impl() override;
	//	void filter_omp_impl() override;
	//	void operator()(const cv::Range& range) const override;
	//public:
	//	ColumnSumFilter_Ip2ab_Guide1_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
	//		: ColumnSumFilter_Ip2ab_Guide1_nonVec(_tempVec, _a, _b, _r, _eps, _parallelType)
	//	{
	//		img_col = a.cols / 4;
	//	}
	//};

	struct ColumnSumFilter_Ip2ab_Guide1_AVX : public ColumnSumFilter_Ip2ab_Guide1_nonVec
	{
	private:
		const __m256 mDiv = _mm256_set1_ps(div);
		const __m256 mEps = _mm256_set1_ps(eps);
		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_Ip2ab_Guide1_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _a, cv::Mat& _b, int _r, float _eps, int _parallelType)
			: ColumnSumFilter_Ip2ab_Guide1_nonVec(_tempVec, _a, _b, _r, _eps, _parallelType)
		{
			img_col = a.cols / 8;
		}

		void filter() override
		{
			if (parallelType == ParallelTypes::NAIVE)
			{
				filter_naive_impl();
			}
			else if (parallelType == ParallelTypes::OMP)
			{
				filter_omp_impl();
			}
			else if (parallelType == ParallelTypes::PARALLEL_FOR_)
			{
				cv::parallel_for_(cv::Range(r, img_col - r), *this, cv::getNumThreads() - 1);
			}
			else
			{

			}
		}
	};



	struct RowSumFilter_ab2q_Guide1 : public RowSumFilter_base
	{
	protected:
		cv::Mat& a;
		cv::Mat& b;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		RowSumFilter_ab2q_Guide1(cv::Mat& _a, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
			: a(_a), b(_b), RowSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = a.rows;
			img_col = a.cols;
		}
			
		void filter() override
		{
			if (parallelType == ParallelTypes::NAIVE)
			{
				filter_naive_impl();
			}
			else if (parallelType == ParallelTypes::OMP)
			{
				filter_omp_impl();
			}
			else if (parallelType == ParallelTypes::PARALLEL_FOR_)
			{
				cv::parallel_for_(cv::Range(r, img_row - r), *this, cv::getNumThreads() - 1);
			}
			else
			{

			}
		}
	};

	struct ColumnSumFilter_ab2q_Guide1_nonVec : public ColumnSumFilter_base
	{
	protected:
		cv::Mat& I;
		cv::Mat& q;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_ab2q_Guide1_nonVec(std::vector<cv::Mat>& _tempVec, cv::Mat& _I, cv::Mat& _q, int _r, int _parallelType)
			: I(_I), q(_q), ColumnSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = I.rows;
			img_col = I.cols;
			step = I.cols;
		}
	
		void filter() override
		{
			if (parallelType == ParallelTypes::NAIVE)
			{
				filter_naive_impl();
			}
			else if (parallelType == ParallelTypes::OMP)
			{
				filter_omp_impl();
			}
			else if (parallelType == ParallelTypes::PARALLEL_FOR_)
			{
				cv::parallel_for_(cv::Range(r, img_col - r), *this, cv::getNumThreads() - 1);
			}
			else
			{

			}
		}
	};

	//struct ColumnSumFilter_ab2q_Guide1_SSE : public ColumnSumFilter_ab2q_Guide1_nonVec
	//{
	//private:
	//	const __m128 mDiv = _mm_set1_ps(div);
	//	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	//	void filter_naive_impl() override;
	//	void filter_omp_impl() override;
	//	void operator()(const cv::Range& range) const override;
	//public:
	//	ColumnSumFilter_ab2q_Guide1_SSE(std::vector<cv::Mat>& _tempVec, cv::Mat& _I, cv::Mat& _q, int _r, int _parallelType)
	//		: ColumnSumFilter_ab2q_Guide1_nonVec(_tempVec, _I, _q, _r, _parallelType)
	//	{
	//		img_col = I.cols / 4;
	//	}
	//};

	struct ColumnSumFilter_ab2q_Guide1_AVX : public ColumnSumFilter_ab2q_Guide1_nonVec
	{
	private:
		const __m256 mDiv = _mm256_set1_ps(div);
		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_ab2q_Guide1_AVX(std::vector<cv::Mat>& _tempVec, cv::Mat& _I, cv::Mat& _q, int _r, int _parallelType)
			: ColumnSumFilter_ab2q_Guide1_nonVec(_tempVec, _I, _q, _r, _parallelType)
		{
			img_col = I.cols / 8;
		}

		void filter() override
		{
			if (parallelType == ParallelTypes::NAIVE)
			{
				filter_naive_impl();
			}
			else if (parallelType == ParallelTypes::OMP)
			{
				filter_omp_impl();
			}
			else if (parallelType == ParallelTypes::PARALLEL_FOR_)
			{
				cv::parallel_for_(cv::Range(r, img_col - r), *this, cv::getNumThreads() - 1);
			}
			else
			{

			}
		}
	};



	///* --- Guide3 --- */
	struct RowSumFilter_Ip2ab_Guide3 : public RowSumFilter_base
	{
	protected:
		cv::Mat& p;
		std::vector<cv::Mat>& vI;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		RowSumFilter_Ip2ab_Guide3(cv::Mat& _p, std::vector<cv::Mat>& _vI, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
			: p(_p), vI(_vI), RowSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = p.rows;
			img_col = p.cols;
		}
	};

	struct ColumnSumFilter_Ip2ab_Guide3_nonVec : public ColumnSumFilter_base
	{
	protected:
		std::vector<cv::Mat>& va;
		cv::Mat& b;
		float eps;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_Ip2ab_Guide3_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
			: va(_va), b(_b), eps(_eps), ColumnSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = va[0].rows;
			img_col = va[0].cols;
			step = va[0].cols;
		}
	};

	//struct ColumnSumFilter_Ip2ab_Guide3_SSE : public ColumnSumFilter_Ip2ab_Guide3_nonVec
	//{
	//private:
	//	const __m128 mDiv = _mm_set1_ps(div);
	//	const __m128 mEps = _mm_set1_ps(eps);
	//	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	//	void filter_naive_impl() override;
	//	void filter_omp_impl() override;
	//	void operator()(const cv::Range& range) const override;
	//public:
	//	ColumnSumFilter_Ip2ab_Guide3_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
	//		: ColumnSumFilter_Ip2ab_Guide3_nonVec(_tempVec, _va, _b, _r, _eps, _parallelType)
	//	{
	//		img_col = va[0].cols / 4;
	//	}
	//};

	struct ColumnSumFilter_Ip2ab_Guide3_AVX : public ColumnSumFilter_Ip2ab_Guide3_nonVec
	{
	private:
		const __m256 mDiv = _mm256_set1_ps(div);
		const __m256 mEps = _mm256_set1_ps(eps);
		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_Ip2ab_Guide3_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _va, cv::Mat& _b, int _r, float _eps, int _parallelType)
			: ColumnSumFilter_Ip2ab_Guide3_nonVec(_tempVec, _va, _b, _r, _eps, _parallelType)
		{
			img_col = va[0].cols / 8;
		}
	};



	struct RowSumFilter_ab2q_Guide3 : public RowSumFilter_base
	{
	protected:
		std::vector<cv::Mat>& va;
		cv::Mat& b;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		RowSumFilter_ab2q_Guide3(std::vector<cv::Mat>& _va, cv::Mat& _b, std::vector<cv::Mat>& _tempVec, int _r, int _parallelType)
			: va(_va), b(_b), RowSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = va[0].rows;
			img_col = va[0].cols;
		}
	};

	struct ColumnSumFilter_ab2q_Guide3_nonVec : public ColumnSumFilter_base
	{
	protected:
		std::vector<cv::Mat>& vI;
		cv::Mat& q;

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_ab2q_Guide3_nonVec(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
			: vI(_vI), q(_q), ColumnSumFilter_base(_tempVec, _r, _parallelType)
		{
			img_row = tempVec[0].rows;
			img_col = tempVec[0].cols;
			step = tempVec[0].cols;
		}
	};

	//struct ColumnSumFilter_ab2q_Guide3_SSE : public ColumnSumFilter_ab2q_Guide3_nonVec
	//{
	//private:
	//	const __m128 mDiv = _mm_set1_ps(div);
	//	const __m128 mBorder = _mm_set1_ps((float)(r + 1));

	//	void filter_naive_impl() override;
	//	void filter_omp_impl() override;
	//	void operator()(const cv::Range& range) const override;
	//public:
	//	ColumnSumFilter_ab2q_Guide3_SSE(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
	//		: ColumnSumFilter_ab2q_Guide3_nonVec(_tempVec, _vI, _q, _r, _parallelType)
	//	{
	//		img_col = tempVec[0].cols / 4;
	//	}
	//};

	struct ColumnSumFilter_ab2q_Guide3_AVX : public ColumnSumFilter_ab2q_Guide3_nonVec
	{
	private:
		const __m256 mDiv = _mm256_set1_ps(div);
		const __m256 mBorder = _mm256_set1_ps((float)(r + 1));

		void filter_naive_impl() override;
		void filter_omp_impl() override;
		void operator()(const cv::Range& range) const override;
	public:
		ColumnSumFilter_ab2q_Guide3_AVX(std::vector<cv::Mat>& _tempVec, std::vector<cv::Mat>& _vI, cv::Mat& _q, int _r, int _parallelType)
			: ColumnSumFilter_ab2q_Guide3_nonVec(_tempVec, _vI, _q, _r, _parallelType)
		{
			img_col = tempVec[0].cols / 8;
		}
	};
}
