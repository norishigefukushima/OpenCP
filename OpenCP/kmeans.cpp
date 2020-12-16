#include "kmeans.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	//static int CV_KMEANS_PARALLEL_GRANULARITY = (int)utils::getConfigurationParameterSizeT("OPENCV_KMEANS_PARALLEL_GRANULARITY", 1000);
	static int	CV_KMEANS_PARALLEL_GRANULARITY = 1000;

	//(a-b)^2
	inline __m256 normL2Sqr(__m256 a, __m256 b)
	{
		__m256 temp = _mm256_sub_ps(a, b);
		return _mm256_mul_ps(temp, temp);
	}

	//(a-b)^2 + c=fma(a-b, a-b, c);
	inline __m256 normL2SqrAdd(__m256 a, __m256 b, __m256 c)
	{
		__m256 temp = _mm256_sub_ps(a, b);
		return _mm256_fmadd_ps(temp, temp, c);
	}

#pragma region initialCentroid
	void KMeans::generateKmeansRandomInitialCentroid(cv::Mat& data_points, cv::Mat& centroids, const int K, cv::RNG& rng)
	{
		const int N = data_points.cols;
		const int dims = data_points.rows;
		cv::AutoBuffer<Vec2f, 64> box(dims);//min-max value for each dimension

		{
			int i = 0;
			for (int j = 0; j < dims; j++)
			{
				const float* sample = data_points.ptr<float>(j);
				box[j] = Vec2f(sample[i], sample[i]);
			}
		}
		for (int d = 0; d < dims; d++)
		{
			for (int i = 1; i < N; i++)
			{
				const float* sample = data_points.ptr<float>(d);
				float v = sample[i];
				box[d][0] = std::min(box[d][0], v);
				box[d][1] = std::max(box[d][1], v);
			}
		}

		for (int k = 0; k < K; k++)
		{
			for (int d = 0; d < dims; d++)
			{
				centroids.ptr<float>(k)[d] = rng.uniform(box[d][0], box[d][1]);
			}
		}
	}

	class KMeansPPDistanceComputer_AVX : public ParallelLoopBody
	{
	private:
		const __m256* src_distance;
		__m256* dest_distance;
		const Mat& data_points;
		const int centroid_index;
	public:
		KMeansPPDistanceComputer_AVX(__m256* dest_dist, const Mat& data_points, const __m256* src_distance, int centroid_index) :
			dest_distance(dest_dist), data_points(data_points), src_distance(src_distance), centroid_index(centroid_index)
		{ }

		void operator()(const cv::Range& range) const CV_OVERRIDE
		{
			//CV_TRACE_FUNCTION();
			const int begin = range.start;
			const int end = range.end;
			const int dims = data_points.rows;

			const int simd_width = data_points.cols / 8;

			std::vector<__m256*> dim(dims);
			{
				int d = 0;
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

				for (int i = 0; i < simd_width; i++)
				{
					dest_distance[i] = normL2Sqr(dim[d][i], centers_value);
				}
			}
			for (int d = 1; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

				for (int i = 0; i < simd_width; i++)
				{
					dest_distance[i] = normL2SqrAdd(dim[d][i], centers_value, dest_distance[i]);
				}
			}

			for (int i = 0; i < simd_width; i++)
			{
				dest_distance[i] = _mm256_min_ps(dest_distance[i], src_distance[i]);
			}
		}
	};

	//k - means center initialization using the following algorithm :
	//Arthur & Vassilvitskii(2007) k-means++ : The Advantages of Careful Seeding
	void KMeans::generateKmeansPPInitialCentroid_AVX(const Mat& data_points, Mat& dest_centroids,
		int K, RNG& rng, int trials)
	{
		//CV_TRACE_FUNCTION();
		const int dims = data_points.rows, N = data_points.cols;
		cv::AutoBuffer<int, 64> _centers(K);
		int* centers = &_centers[0];

		//3 buffers; dist, tdist, tdist2.
		if (_distance.size() != N * 3) _distance.allocate(N * 3);

		__m256* dist = (__m256*) & _distance[0];
		__m256* tdist = dist + N / 8;
		__m256* tdist2 = tdist + N / 8;

		const int simd_sizeN = N / 8;

		//randomize the first centroid
		centers[0] = (unsigned)rng % N;

		//determin the first centroid by mean (not effective)
		if (false)
		{
			Scalar v0 = mean(data_points.row(0));
			Scalar v1 = mean(data_points.row(1));
			Scalar v2 = mean(data_points.row(2));
			const float* d0 = data_points.ptr<float>(0);
			const float* d1 = data_points.ptr<float>(1);
			const float* d2 = data_points.ptr<float>(2);
			float diff_max = FLT_MAX;
			int argindex = 0;
			for (int i = 0; i < N; i++)
			{
				float diff = (d0[i] - v0.val[0]) * (d0[i] - v0.val[0])
					+ (d1[i] - v1.val[0]) * (d1[i] - v1.val[0])
					+ (d2[i] - v2.val[0]) * (d2[i] - v2.val[0]);
				if (diff < diff_max)
				{
					diff_max = diff;
					argindex = i;
				}
			}
			centers[0] = argindex;
		}

		for (int i = 0; i < simd_sizeN; i++)
		{
			dist[i] = _mm256_setzero_ps();
		}

		float distance_sum = 0.f;
		for (int d = 0; d < dims; d++)
		{
			const float* p = data_points.ptr<float>(d);
			__m256* mp = (__m256*)p;
			__m256 centers_value = _mm256_set1_ps(p[centers[0]]);
			__m256 dist_value_acc = _mm256_setzero_ps();
			for (int i = 0; i < simd_sizeN; i++)
			{
				//	dist[i]を求めるための処理
				__m256 dist_value = cp::normL2Sqr(mp[i], centers_value);
				dist[i] = _mm256_add_ps(dist[i], dist_value);

				//	sum0を求めるための処理
				dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
			}
			distance_sum += _mm256_reduceadd_ps(dist_value_acc);
		}

		for (int k = 1; k < K; k++)
		{
			float bestSum = FLT_MAX;
			int bestCenter = -1;

			for (int j = 0; j < trials; j++)
			{
				float p = (float)rng * distance_sum;//original
				//float p = (float)rng * distance_sum / dims;//better? case by case
				int ci = 0;

				for (; ci < N - 1; ci++)
				{
					p -= _distance[ci];
					if (p <= 0)
					{
						break;
					}
				}

				//	Range : start=0,end=N
				//	KMeansPPDistanceComputer : tdist2=tdist2, data=data, dist=dist, ci=ci
				//	divUp : (dims*N + CV_KMEANS_PARALLEL_GRANULIARITY - 1) / CV_KMEANS_PARALLEL_GRANULIARITY　
				parallel_for_(Range(0, N),
					KMeansPPDistanceComputer_AVX(tdist2, data_points, dist, ci),
					cv::getNumThreads());

				float distance_sum_local = 0.f;
				__m256 tdist2_acc = _mm256_setzero_ps();
				for (int i = 0; i < simd_sizeN; i++)
				{
					tdist2_acc = _mm256_add_ps(tdist2_acc, tdist2[i]);
				}
				distance_sum_local += _mm256_reduceadd_ps(tdist2_acc);

				if (distance_sum_local < bestSum)
				{
					bestSum = distance_sum_local;
					bestCenter = ci;

					std::swap(tdist, tdist2);
				}
			}

			if (bestCenter < 0)
				CV_Error(Error::StsNoConv, "kmeans: can't update cluster center (check input for huge or NaN values)");

			centers[k] = bestCenter;//in intensity index, where have minimum distance
			distance_sum = bestSum;
			std::swap(dist, tdist);
		}

		for (int k = 0; k < K; k++)
		{
			float* dst = dest_centroids.ptr<float>(k);
			for (int d = 0; d < dims; d++)
			{
				const float* src = data_points.ptr<float>(d);
				dst[d] = src[centers[k]];
			}
		}
	}
#pragma endregion

#pragma region updateCentroid
	void KMeans::getOuterSample(cv::Mat& src_centroids, cv::Mat& dest_centroids, const cv::Mat& data_points, const cv::Mat& labels)
	{
		const int N = data_points.cols;
		const int dims = data_points.rows;
		const int K = src_centroids.rows;
		cv::AutoBuffer<float, 64> Hcounters(K);

		for (int k = 0; k < K; k++) Hcounters[k] = 0.f;

		const int* l = labels.ptr<int>();
		for (int i = 0; i < N; i++)
		{
			const int arg_k = l[i];

			float dist = 0.f;
			for (int d = 0; d < dims; d++)
			{
				const float* dataPtr = data_points.ptr<float>(d);
				float diff = (src_centroids.ptr<float>(arg_k)[d] - dataPtr[i]);
				dist += diff * diff;
			}
			if (dist > Hcounters[arg_k])
			{
				Hcounters[arg_k] = dist;
				for (int d = 0; d < dims; d++)
				{
					const float* dataPtr = data_points.ptr<float>(d);
					dest_centroids.ptr<float>(arg_k)[d] = dataPtr[i];
				}
			}
		}
	}

	//Nxdims
	void KMeans::boxMeanCentroid(Mat& data_points, const int* labels, Mat& dest_centroid, int* counters)
	{
		//cannot vectorize it without scatter
		const int dims = data_points.rows;
		const int N = data_points.cols;

		{
			int d = 0;
			float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				int arg_k = labels[i];
				dest_centroid.ptr<float>(arg_k)[d] += dataPtr[i];
				counters[arg_k]++;
			}
		}
		for (int d = 1; d < dims; d++)
		{
			float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				int arg_k = labels[i];
				dest_centroid.ptr<float>(arg_k)[d] += dataPtr[i];
			}
		}
	}

	//N*dims
	void KMeans::weightedMeanCentroid(Mat& data_points, const int* labels, const Mat& src_centroid, float* Table, Mat& dest_centroid, float* centroid_weight, int* counters)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int K = src_centroid.rows;

		for (int k = 0; k < K; k++) centroid_weight[k] = 0.f;

		cv::AutoBuffer<float*, 64> dataTop(dims);
		for (int d = 0; d < dims; d++)
		{
			dataTop[d] = data_points.ptr<float>(d);
		}

#if 0
		//scalar
		cv::AutoBuffer<const float*, 64> centroidTop(K);
		for (int k = 0; k < K; k++)
		{
			centroidTop[k] = src_centroid.ptr<float>(k);
		}
		for (int i = 0; i < N; i++)
		{
			const int arg_k = labels[i];

			float dist = 0.f;
			for (int d = 0; d < dims; d++)
			{
				float diff = (centroidTop[arg_k][d] - dataTop[d][i]);
				dist += diff * diff;
			}
			const float wi = Table[int(sqrt(dist))];
			centroid_weight[arg_k] += wi;
			counters[arg_k]++;
			for (int d = 0; d < dims; d++)
			{
				dest_centroid.ptr<float>(arg_k)[d] += wi * dataTop[d][i];
			}
		}
#else
		const float* centroidPtr = src_centroid.ptr<float>();//dim*K
		for (int i = 0; i < N; i += 8)
		{
			const __m256i marg_k = _mm256_load_si256((__m256i*)(labels + i));
			const __m256i midx = _mm256_mullo_epi32(marg_k, _mm256_set1_epi32(dims));
			__m256 mdist = _mm256_setzero_ps();

			for (int d = 0; d < dims; d++)
			{
				__m256 mc = _mm256_i32gather_ps(centroidPtr, _mm256_add_epi32(midx, _mm256_set1_epi32(d)), 4);
				mc = _mm256_sub_ps(mc, _mm256_load_ps(&dataTop[d][i]));
				mdist = _mm256_fmadd_ps(mc, mc, mdist);
			}

			__m256 mwi = _mm256_i32gather_ps(Table, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist)), 4);
			for (int v = 0; v < 8; v++)
			{
				const int arg_k = marg_k.m256i_i32[v];
				const float wi = mwi.m256_f32[v];
				centroid_weight[arg_k] += wi;
				counters[arg_k]++;
				float* dstCentroidPtr = dest_centroid.ptr<float>(arg_k);
				for (int d = 0; d < dims; d++)
				{
					dstCentroidPtr[d] += wi * dataTop[d][i + v];
				}
			}
		}
#endif 
	}

	//N*dims
	void KMeans::harmonicMeanCentroid(Mat& data_points, const int* labels, const Mat& src_centroid, Mat& dest_centroid, float* centroid_weight, int* counters)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int K = src_centroid.rows;
		for (int k = 0; k < K; k++) centroid_weight[k] = 0.f;

		for (int i = 0; i < N; i++)
		{
			float w = 0.f;
			const float p = 3.5f;
			const int arg_k = labels[i];

			float w0 = 0.f;
			float w1 = 0.f;
			for (int k = 0; k < K; k++)
			{
				float w0_ = 0.f;
				float w1_ = 0.f;
				for (int d = 0; d < dims; d++)
				{
					float* dataPtr = data_points.ptr<float>(d);
					float diff = abs(src_centroid.ptr<float>(k)[d] - dataPtr[i]);
					if (diff == 0.f)diff += FLT_EPSILON;
					w0_ += pow(diff, -p - 2.f);
					w1_ += pow(diff, -p);
				}
				w0 += pow(w0_, 1.f / (-p - 2.f));
				w1 += pow(w1_, 1.f / (-p));
			}
			w = w0 / (w1 * w1);
			//std::cout <<i<<":"<< w <<", "<<w0<<","<<w1<< std::endl;

			centroid_weight[arg_k] += w;
			counters[arg_k]++;
			for (int d = 0; d < dims; d++)
			{
				float* dataPtr = data_points.ptr<float>(d);
				dest_centroid.ptr<float>(arg_k)[d] += w * dataPtr[i];
			}
		}
	}
#pragma endregion

#pragma region assignCentroid
	template<bool onlyDistance>
	class KMeansDistanceComputer_AVX : public ParallelLoopBody
	{
	private:
		KMeansDistanceComputer_AVX& operator=(const KMeansDistanceComputer_AVX&); // = delete

		float* distances;
		int* labels;
		const Mat& dataPoints;
		const Mat& centroids;

	public:
		KMeansDistanceComputer_AVX(float* dest_distance,
			int* dest_labels,
			const Mat& dataPoints,
			const Mat& centroids)
			: distances(dest_distance),
			labels(dest_labels),
			dataPoints(dataPoints),
			centroids(centroids)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			//CV_TRACE_FUNCTION();
			const int K = centroids.rows;
			const int dims = centroids.cols;//when color case, dim= 3
			const int BEGIN = range.start / 8;
			const int END = range.end / 8;
			__m256i* mlabel_dest = (__m256i*) & labels[0];
			__m256* mdist_dest = (__m256*) & distances[0];
			if constexpr (onlyDistance)
			{
				const float* center = centroids.ptr<float>();
				for (int n = BEGIN; n < END; n++)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(*(__m256*)dataPoints.ptr<float>(d, 8 * n), _mm256_set1_ps(center[d]), mdist);
					}
					mdist_dest[n] = mdist;
				}
			}
			else
			{
				enum LOOP
				{
					KDN,
					NKD
				};
				const int loop = LOOP::KDN;
				//const int loop = LOOP::NKD;
				if (loop == LOOP::KDN)
				{
					//loop k-d-n
#if 0
					__m256* mcenter = (__m256*) _mm_malloc(sizeof(__m256) * dims, AVX_ALIGN);
					{
						//k=0
						const float* center = centroids.ptr<float>(0);
						for (int d = 0; d < dims; d++)
						{
							mcenter[d] = _mm256_set1_ps(center[d]);
						}
						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(*(__m256*)dataPoints.ptr<float>(0, 8 * n), mcenter[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(*(__m256*)dataPoints.ptr<float>(d, 8 * n), mcenter[d], mdist);
							}
							mdist_dest[n] = mdist;
							mlabel_dest[n] = _mm256_setzero_si256();//set K=0;
						}
					}
					for (int k = 1; k < K; k++)
					{
						const float* center = centroids.ptr<float>(k);
						for (int d = 0; d < dims; d++)
						{
							mcenter[d] = _mm256_set1_ps(center[d]);
						}
						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(*(__m256*)dataPoints.ptr<float>(0, 8 * n), mcenter[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(*(__m256*)dataPoints.ptr<float>(d, 8 * n), mcenter[d], mdist);
							}

							__m256 mask = _mm256_cmp_ps(mdist, mdist_dest[n], _CMP_GT_OQ);
							mdist_dest[n] = _mm256_blendv_ps(mdist, mdist_dest[n], mask);
							__m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							mlabel_dest[n] = _mm256_blendv_epi8(mlabel_dest[n], _mm256_set1_epi32(k), label_mask);
						}
					}
					_mm_free(mcenter);
#else
					{
						//k=0
						const float* center = centroids.ptr<float>(0);
						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(*(__m256*)dataPoints.ptr<float>(0, 8 * n), _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(*(__m256*)dataPoints.ptr<float>(d, 8 * n), _mm256_set1_ps(center[d]), mdist);
							}
							mdist_dest[n] = mdist;
							mlabel_dest[n] = _mm256_setzero_si256();//set K=0;
						}
					}
					for (int k = 1; k < K; k++)
					{
						const float* center = centroids.ptr<float>(k);
						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(*(__m256*)dataPoints.ptr<float>(0, 8 * n), _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(*(__m256*)dataPoints.ptr<float>(d, 8 * n), _mm256_set1_ps(center[d]), mdist);
							}
							__m256 mask = _mm256_cmp_ps(mdist, mdist_dest[n], _CMP_GT_OQ);
							mdist_dest[n] = _mm256_blendv_ps(mdist, mdist_dest[n], mask);
							__m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							mlabel_dest[n] = _mm256_blendv_epi8(mlabel_dest[n], _mm256_set1_epi32(k), label_mask);
						}
					}
#endif
				}
				else
				{
					//loop n-k-d
					__m256* mdp = (__m256*)_mm_malloc(sizeof(__m256) * dims, AVX_ALIGN);
					for (int n = BEGIN; n < END; n++)
					{
						mlabel_dest[n] = _mm256_setzero_si256();
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = *((__m256*)(dataPoints.ptr<float>(d, 8 * n)));
						}
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							mdist_dest[n] = mdist;
							mlabel_dest[n] = _mm256_setzero_si256();//set K=0;
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));//d=0
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							__m256 mask = _mm256_cmp_ps(mdist, mdist_dest[n], _CMP_GT_OQ);
							mdist_dest[n] = _mm256_blendv_ps(mdist, mdist_dest[n], mask);
							__m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							mlabel_dest[n] = _mm256_blendv_epi8(mlabel_dest[n], _mm256_set1_epi32(k), label_mask);
						}
					}
					_mm_free(mdp);
				}
			}
		}
	};
#pragma endregion

	double KMeans::clustering(cv::InputArray _data, int K,
		cv::InputOutputArray _bestLabels,
		cv::TermCriteria criteria, int attempts,
		int flags, OutputArray dest_centroids, MeanFunction function)
	{
		//std::cout << "sigma" << sigma << std::endl;
		float* weight_table = (float*)_mm_malloc(sizeof(float) * 443, AVX_ALIGN);
		if (function == MeanFunction::GaussInv)
		{
			//for 3chanel 255 max case
			//443=sqrt(3*255^2)
			for (int i = 0; i < 443; i++)
			{
				weight_table[i] = 1.f - exp(i * i / (-2.f * sigma * sigma)) + 0.02f;
				//weight_table[i] = pow(i,sigma*0.1)+0.01;
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
				//w = exp(w / (-2.0 * sigma * sigma));
				//w = 1.0 - exp(sqrt(w) / (-1.0 * sigma));
			}
		}
		if (function == MeanFunction::Gauss)
		{
			for (int i = 0; i < 443; i++)
			{
				weight_table[i] = exp(i * i / (-2.f * sigma * sigma));
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
			}
		}

		const int SPP_TRIALS = 3;
		Mat src = _data.getMat();
		const bool isrow = src.rows == 1;
		const int N = max(src.cols, src.rows);//input data size
		const int dims = min(src.cols, src.rows) * src.channels();//input dimensions
		const int type = src.depth();

		//AoS to SoA by transpose
		Mat	src_t = (src.cols < src.rows) ? src.t() : src;

		attempts = std::max(attempts, 1);
		CV_Assert(src.dims <= 2 && type == CV_32F && K > 0);
		CV_CheckGE(N, K, "Number of clusters should be more than number of elements");

		//	data format
		//	Mat::Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP)
		//	data0.step = data0 byte(3*4byte(size_of_float)=12byte)
		//	Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));
		Mat data_points(dims, N, CV_32F, src_t.ptr(), isrow ? N * sizeof(float) : static_cast<size_t>(src_t.step));

		_bestLabels.create(N, 1, CV_32S, -1, true);//8U is better for small label cases
		Mat best_labels = _bestLabels.getMat();

		if (flags & cv::KMEANS_USE_INITIAL_LABELS)// for KMEANS_USE_INITIAL_LABELS
		{
			CV_Assert((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous());

			best_labels.reshape(1, N).copyTo(labels_internal);
			for (int i = 0; i < N; i++)
			{
				CV_Assert((unsigned)labels_internal.at<int>(i) < (unsigned)K);
			}
		}
		else //alloc buffer
		{
			if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous()))
			{
				_bestLabels.create(N, 1, CV_32S);
				best_labels = _bestLabels.getMat();
			}
			labels_internal.create(best_labels.size(), best_labels.type());
		}
		int* labels = labels_internal.ptr<int>();

		Mat centroids(K, dims, type);
		Mat old_centroids(K, dims, type);
		Mat temp(1, dims, type);

		cv::AutoBuffer<float, 64> centroid_weight(K);
		cv::AutoBuffer<int, 64> counters(K);
		cv::AutoBuffer<float, 64> dists(N);//double->float
		RNG& rng = theRNG();

		if (criteria.type & TermCriteria::EPS)criteria.epsilon = std::max(criteria.epsilon, 0.);
		else criteria.epsilon = FLT_EPSILON;

		criteria.epsilon *= criteria.epsilon;

		if (criteria.type & TermCriteria::COUNT)criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
		else criteria.maxCount = 100;

		if (K == 1)
		{
			attempts = 1;
			criteria.maxCount = 2;
		}

		float best_compactness = FLT_MAX;
		for (int attempt_index = 0; attempt_index < attempts; attempt_index++)
		{
			float compactness = 0.f;

			//main loop
			for (int iter = 0; ;)
			{
				float max_center_shift = (iter == 0) ? FLT_MAX : 0.f;

				swap(centroids, old_centroids);

				bool isInit = ((iter == 0) && (attempt_index > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)));//initial attemp && KMEANS_USE_INITIAL_LABELS is true
				if (isInit)//initialization for first loop
				{
					//cp::Timer t("generate sample"); //<1ns
					if (flags & KMEANS_PP_CENTERS)//kmean++
					{
						generateKmeansPPInitialCentroid_AVX(data_points, centroids, K, rng, SPP_TRIALS);
					}
					else //random initialization
					{
						//if (!(flags & KMEANS_PP_CENTERS))//!kmeans++ processing
						generateKmeansRandomInitialCentroid(data_points, centroids, K, rng);
					}
				}
				else
				{
					//cp::Timer t("compute centroid"); //<1ms．
					//update centroid 

					centroids.setTo(0.f);
					for (int k = 0; k < K; k++) counters[k] = 0;

					if (data_points.cols == dims) data_points = data_points.t();

					//compute centroid without normalization; loop: N x d 
					if (function == MeanFunction::Harmonic)
					{
						harmonicMeanCentroid(data_points, labels, old_centroids, centroids, centroid_weight, counters);
					}
					else if (function == MeanFunction::Gauss || function == MeanFunction::GaussInv)
					{
						weightedMeanCentroid(data_points, labels, old_centroids, weight_table, centroids, centroid_weight, counters);
					}
					else if (function == MeanFunction::Mean)
					{
						boxMeanCentroid(data_points, labels, centroids, counters);
					}

					//processing for empty cluster
					//loop: N x K loop; but the most parts are skipped
					//if some cluster appeared to be empty then:
					//   1. find the biggest cluster
					//   2. find the farthest from the center point in the biggest cluster
					//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
					for (int k = 0; k < K; k++)
					{
						if (counters[k] != 0) continue;

						//std::cout << "empty: " << k << std::endl;

						int k_count_max = 0;
						for (int k1 = 1; k1 < K; k1++)
						{
							if (counters[k_count_max] < counters[k1])
								k_count_max = k1;
						}

						float max_dist = 0.f;
						int farthest_i = -1;
						float* base_center = centroids.ptr<float>(k_count_max);
						float* _base_center = temp.ptr<float>(); // normalized
						const float count_normalize = 1.f / counters[k_count_max];

						for (int j = 0; j < dims; j++)
							_base_center[j] = base_center[j] * count_normalize;

						for (int i = 0; i < N; i++)
						{
							if (labels[i] != k_count_max) continue;

							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += (data_points.ptr<float>(d)[i] - _base_center[d]) * (data_points.ptr<float>(d)[i] - _base_center[d]);
							}

							if (max_dist <= dist)
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[k_count_max]--;
						counters[k]++;
						labels[farthest_i] = k;

						float* cur_center = centroids.ptr<float>(k);

						for (int d = 0; d < dims; d++)
						{
							base_center[d] -= data_points.ptr<float>(d)[farthest_i];
							cur_center[d] += data_points.ptr<float>(d)[farthest_i];
						}
					}

					//normalization and compute max shift distance between old centroid and new centroid
					//small loop: K x d
					for (int k = 0; k < K; k++)
					{
						float* centroidsPtr = centroids.ptr<float>(k);
						CV_Assert(counters[k] != 0);

						float count_normalize = 0.f;
						if (function == MeanFunction::Mean)
							count_normalize = 1.f / counters[k];
						else
							count_normalize = 1.f / centroid_weight[k];//weighted mean


						for (int d = 0; d < dims; d++) centroidsPtr[d] *= count_normalize;

						if (iter > 0)
						{
							float dist = 0.f;
							const float* old_center = old_centroids.ptr<float>(k);
							for (int d = 0; d < dims; d++)
							{
								float t = centroidsPtr[d] - old_center[d];
								dist += t * t;
							}
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				//compute distance and relabel
				//image size x dimensions x K (the most large loop)
				bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);
				{
					//cp::Timer t(format("%d: distant computing", iter)); //15msここが一番重たい．ラストは速い．
					if (isLastIter)
					{
						// compute distance only
						parallel_for_(Range(0, N), KMeansDistanceComputer_AVX<true>(dists.data(), labels, data_points, centroids), cv::getNumThreads());
						compactness = sum(Mat(Size(N, 1), CV_32F, &dists[0]))[0];
						//getOuterSample(centroids, old_centroids, data_points, labels_internal);
						//swap(centroids, old_centroids);		
						break;
					}
					else
					{
						// assign labels
						parallel_for_(Range(0, N), KMeansDistanceComputer_AVX<false>(dists.data(), labels, data_points, centroids), cv::getNumThreads());
					}
				}
			}

			//reshape data structure for output
			if (compactness < best_compactness)
			{
				best_compactness = compactness;
				if (dest_centroids.needed())
				{
					if (dest_centroids.fixedType() && dest_centroids.channels() == dims)
						centroids.reshape(dims).copyTo(dest_centroids);
					else
						centroids.copyTo(dest_centroids);
				}
				labels_internal.copyTo(best_labels);
			}
		}

		_mm_free(weight_table);
		return best_compactness;
	}

	double kmeans(InputArray _data, int K,
		InputOutputArray _bestLabels,
		TermCriteria criteria, int attempts,
		int flags, OutputArray _centers)
	{
		KMeans km;
		return km.clustering(_data, K, _bestLabels, criteria, attempts, flags, _centers);
	}
}