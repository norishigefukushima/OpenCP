#include "kmeans.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	//static int CV_KMEANS_PARALLEL_GRANULARITY = (int)utils::getConfigurationParameterSizeT("OPENCV_KMEANS_PARALLEL_GRANULARITY", 1000);
	static int	CV_KMEANS_PARALLEL_GRANULARITY = 1000;
	enum KMeansDistanceLoop
	{
		KND,
		NKD
	};

	double KMeans::clustering(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, Schedule schedule)
	{
		double ret = 0.0;
		int channels = min(_data.size().width, _data.size().height);
		switch (schedule)
		{
		case cp::KMeans::Schedule::AoS_NKD:
			ret = clusteringAoS(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoA_KND:
			ret = clusteringSoA(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;
		case cp::KMeans::Schedule::AoS_KND:
			ret = clusteringAoS(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;
		case cp::KMeans::Schedule::SoA_NKD:
			ret = clusteringSoA(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoAoS_NKD:
			ret = clusteringSoAoS(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoAoS_KND:
			ret = clusteringSoAoS(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;

		case cp::KMeans::Schedule::Auto:
		default:
		{
			if (channels < 7)
			{
				ret = clusteringSoA(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND);
			}
			else
			{
				ret = clusteringAoS(_data, K, _bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD);
			}
		}
		break;
		}

		return ret;
	}

	double kmeans(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers)
	{
		KMeans km;
		return km.clustering(_data, K, _bestLabels, criteria, attempts, flags, _centers);
	}

#pragma region SoA
	inline float normL2Sqr(float a, float b)
	{
		float temp = a - b;
		return temp * temp;
	}

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
	void KMeans::generateKmeansRandomInitialCentroidSoA(const cv::Mat& data_points, cv::Mat& dest_centroids, const int K, cv::RNG& rng)
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
				dest_centroids.ptr<float>(k)[d] = rng.uniform(box[d][0], box[d][1]);
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
	void KMeans::generateKmeansPPInitialCentroidSoA(const Mat& data_points, Mat& dest_centroids,
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
				//	dist[i]‚ð‹‚ß‚é‚½‚ß‚Ìˆ—
				__m256 dist_value = cp::normL2Sqr(mp[i], centers_value);
				dist[i] = _mm256_add_ps(dist[i], dist_value);

				//	sum0‚ð‹‚ß‚é‚½‚ß‚Ìˆ—
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
				//	divUp : (dims*N + CV_KMEANS_PARALLEL_GRANULIARITY - 1) / CV_KMEANS_PARALLEL_GRANULIARITY@
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
	void KMeans::boxMeanCentroidSoA(Mat& data_points, const int* labels, Mat& dest_centroid, int* counters)
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
	void KMeans::weightedMeanCentroid(Mat& data_points, const int* labels, const Mat& src_centroid, const float* Table, const int tableSize, Mat& dest_centroid, float* dest_centroid_weight, int* dest_counters)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int K = src_centroid.rows;

		for (int k = 0; k < K; k++) dest_centroid_weight[k] = 0.f;

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
		const __m256i mtsize = _mm256_set1_epi32(tableSize - 1);
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

			//__m256 mwi = _mm256_i32gather_ps(Table, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist)), 4);
			__m256 mwi = _mm256_i32gather_ps(Table, _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist))), 4);
			for (int v = 0; v < 8; v++)
			{
				const int arg_k = ((int*)&marg_k)[v];
				const float wi = ((float*)&mwi)[v];
				dest_centroid_weight[arg_k] += wi;
				dest_counters[arg_k]++;
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
	template<bool onlyDistance, int loop>
	class KMeansDistanceComputer_SoADim : public ParallelLoopBody
	{
	private:
		KMeansDistanceComputer_SoADim& operator=(const KMeansDistanceComputer_SoADim&); // = delete

		float* distances;
		int* labels;
		const Mat& dataPoints;
		const Mat& centroids;

	public:
		KMeansDistanceComputer_SoADim(float* dest_distance,
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
			const int dims = centroids.cols;//when color case, dim= 3

			const int K = centroids.rows;
			const int BEGIN = range.start / 8;
			const int END = (range.end % 8 == 0) ? range.end / 8 : (range.end / 8) - 1;

			__m256i* mlabel_dest = (__m256i*) & labels[0];
			__m256* mdist_dest = (__m256*) & distances[0];
			if constexpr (onlyDistance)
			{

				AutoBuffer<__m256*> dptr(dims);
				AutoBuffer<__m256> mc(dims);

				const float* center = centroids.ptr<float>();
				for (int d = 0; d < dims; d++)
				{
					dptr[d] = (__m256*)dataPoints.ptr<float>(d);
					mc[d] = _mm256_set1_ps(center[d]);
				}

				for (int n = BEGIN; n < END; n++)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
					}
					mdist_dest[n] = mdist;
				}

				for (int n = END * 8; n < range.end; n++)
				{
					float dist = 0.f;
					for (int d = 0; d < dims; d++)
					{
						dist += normL2Sqr(dataPoints.at<float>(d, n), center[d]);
					}
					distances[n] = dist;
				}
			}
			else
			{
				//std::cout << "SoA: KND" << std::endl;
				if (loop == KMeansDistanceLoop::KND)//loop k-n-d
				{
					AutoBuffer<__m256*> dptr(dims);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = (__m256*)dataPoints.ptr<float>(d);
					}

					AutoBuffer<__m256> mc(dims);
					{
						//k=0
						const float* center = centroids.ptr<float>(0);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(dptr[0][n], mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
							}
							mdist_dest[n] = mdist;
							mlabel_dest[n] = _mm256_setzero_si256();//set K=0;
						}

						for (int n = END * 8; n < range.end; n++)
						{
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += normL2Sqr(dataPoints.at<float>(d, n), center[d]);
							}
							distances[n] = dist;
							labels[n] = 0;
						}
					}
					for (int k = 1; k < K; k++)
					{
						const float* center = centroids.ptr<float>(k);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(dptr[0][n], mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
							}
							__m256 mask = _mm256_cmp_ps(mdist, mdist_dest[n], _CMP_GT_OQ);
							mdist_dest[n] = _mm256_blendv_ps(mdist, mdist_dest[n], mask);
							__m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							mlabel_dest[n] = _mm256_blendv_epi8(mlabel_dest[n], _mm256_set1_epi32(k), label_mask);
						}

						for (int n = END * 8; n < range.end; n++)
						{
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += normL2Sqr(dataPoints.at<float>(d, n), center[d]);
							}
							if (dist < distances[n])
							{
								distances[n] = dist;
								labels[n] = k;
							}
						}
					}
				}
				else //loop n-k-d
				{
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

					for (int n = END * 8; n < range.end; n++)
					{
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += normL2Sqr(dataPoints.at<float>(d, n), center[d]);
							}
							distances[n] = dist;
							labels[n] = 0;
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += normL2Sqr(dataPoints.at<float>(d, n), center[d]);
							}
							if (dist < distances[n])
							{
								distances[n] = dist;
								labels[n] = k;
							}
						}
					}

					_mm_free(mdp);
				}
			}
		}
	};

	//copy from KMeansDistanceComputer_SoADim
	template<bool onlyDistance, int loop, int dims>
	class KMeansDistanceComputer_SoA : public ParallelLoopBody
	{
	private:
		KMeansDistanceComputer_SoA& operator=(const KMeansDistanceComputer_SoA&); // = delete

		float* distances;
		int* labels;
		const Mat& dataPoints;
		const Mat& centroids;

	public:
		KMeansDistanceComputer_SoA(float* dest_distance,
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
			const int BEGIN = range.start / 8;
			const int END = (range.end % 8 == 0) ? range.end / 8 : (range.end / 8) - 1;
			//const int END = range.end / 8;

			__m256i* mlabel_dest = (__m256i*) & labels[0];
			__m256* mdist_dest = (__m256*) & distances[0];
			if constexpr (onlyDistance)
			{
				AutoBuffer<__m256*> dptr(dims);
				AutoBuffer<__m256> mc(dims);
				{
					const float* center = centroids.ptr<float>(0);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = (__m256*)dataPoints.ptr<float>(d);
						mc[d] = _mm256_set1_ps(center[d]);
					}
				}
				for (int n = BEGIN; n < END; n++)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
					}
					mdist_dest[n] = mdist;
				}
			}
			else
			{
				if (loop == KMeansDistanceLoop::KND)//loop k-n-d
				{
					AutoBuffer<__m256*> dptr(dims);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = (__m256*)dataPoints.ptr<float>(d);
					}

					AutoBuffer<__m256> mc(dims);
					{
						//k=0
						const float* center = centroids.ptr<float>(0);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(dptr[0][n], mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
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
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n++)
						{
							__m256 mdist = normL2Sqr(dptr[0][n], mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(dptr[d][n], mc[d], mdist);
							}

							__m256 mask = _mm256_cmp_ps(mdist, mdist_dest[n], _CMP_GT_OQ);
							mdist_dest[n] = _mm256_blendv_ps(mdist, mdist_dest[n], mask);
							__m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							mlabel_dest[n] = _mm256_blendv_epi8(mlabel_dest[n], _mm256_set1_epi32(k), label_mask);
						}
					}
				}
				else //loop n-k-d
				{
					__m256* mdp = (__m256*)_mm_malloc(sizeof(__m256) * dims, AVX_ALIGN);
					for (int n = BEGIN; n < END; n++)
					{
						const int N = 8 * n;
						mlabel_dest[n] = _mm256_setzero_si256();
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = *((__m256*)(dataPoints.ptr<float>(d, N)));
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
	double KMeans::clusteringSoA(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, OutputArray dest_centroids, MeanFunction function, int loop)
	{
		const int SPP_TRIALS = 3;
		Mat src = _data.getMat();
		const bool isrow = (src.rows == 1);
		const int N = max(src.cols, src.rows);//input data size
		const int dims = min(src.cols, src.rows) * src.channels();//input dimensions
		const int type = src.depth();

		//std::cout << "KMeans::clustering" << std::endl;
		//std::cout << "sigma" << sigma << std::endl;
		weightTableSize = (int)ceil(sqrt(signal_max * signal_max * dims));//for 3channel 255 max case 442=ceil(sqrt(3*255^2))
		//std::cout << "tableSize" << tableSize << std::endl;
		float* weight_table = (float*)_mm_malloc(sizeof(float) * weightTableSize, AVX_ALIGN);
		if (function == MeanFunction::GaussInv)
		{
			//cout << "MeanFunction::GaussInv sigma " << sigma << endl;

			for (int i = 0; i < weightTableSize; i++)
			{
				//weight_table[i] = 1.f;
				weight_table[i] = 1.f - exp(i * i / (-2.f * sigma * sigma)) + 0.001f;
				//weight_table[i] =Huber(i, sigma) + 0.001f;
				//weight_table[i] = i< sigma ? 0.001f: 1.f;
				//float n = 2.2f;
				//weight_table[i] = 1.f - exp(pow(i,n) / (-n * pow(sigma,n))) + 0.02f;
				//weight_table[i] = pow(i,sigma*0.1)+0.01;
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
				//w = exp(w / (-2.0 * sigma * sigma));
				//w = 1.0 - exp(sqrt(w) / (-1.0 * sigma));
			}
		}
		if (function == MeanFunction::LnNorm)
		{
			for (int i = 0; i < weightTableSize; i++)
			{
				weight_table[i] = pow(i, min(sigma, 10.f)) + FLT_EPSILON;
			}
		}

		if (function == MeanFunction::Gauss)
		{
			for (int i = 0; i < weightTableSize; i++)
			{
				weight_table[i] = exp(i * i / (-2.f * sigma * sigma));
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
			}
		}

		//AoS to SoA by using transpose
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

			best_labels.copyTo(labels_internal);
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
		if ((flags & KMEANS_USE_INITIAL_LABELS) && (function == MeanFunction::Gauss || function == MeanFunction::GaussInv || function == MeanFunction::LnNorm))
		{
			dest_centroids.copyTo(centroids);
		}
		Mat old_centroids(K, dims, type);
		Mat temp(1, dims, type);

		cv::AutoBuffer<float, 64> centroid_weight(K);
		cv::AutoBuffer<int, 64> label_count(K);
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

				const bool isInit = ((iter == 0) && (attempt_index > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)));//initial attemp && KMEANS_USE_INITIAL_LABELS is true
				if (isInit)//initialization for first loop
				{
					//cp::Timer t("generate sample"); //<1ns
					if (flags & KMEANS_PP_CENTERS)//kmean++
					{
						generateKmeansPPInitialCentroidSoA(data_points, centroids, K, rng, SPP_TRIALS);
					}
					else //random initialization
					{
						generateKmeansRandomInitialCentroidSoA(data_points, centroids, K, rng);
					}
				}
				else
				{
					//cp::Timer t("compute centroid"); //<1msD
					//update centroid 
					centroids.setTo(0.f);
					for (int k = 0; k < K; k++) label_count[k] = 0;

					//compute centroid without normalization; loop: N x d 
					if (function == MeanFunction::Harmonic)
					{
						harmonicMeanCentroid(data_points, labels, old_centroids, centroids, centroid_weight, label_count);
					}
					else if (function == MeanFunction::Gauss || function == MeanFunction::GaussInv || function == MeanFunction::LnNorm)
					{
						weightedMeanCentroid(data_points, labels, old_centroids, weight_table, weightTableSize, centroids, centroid_weight, label_count);
					}
					else if (function == MeanFunction::Mean)
					{
						boxMeanCentroidSoA(data_points, labels, centroids, label_count);
					}

					//processing for empty cluster
					//loop: N x K loop; but the most parts are skipped
					//if some cluster appeared to be empty then:
					//   1. find the biggest cluster
					//   2. find the farthest from the center point in the biggest cluster
					//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
	//#define DEBUG_SHOW_SKIP 
#ifdef DEBUG_SHOW_SKIP
					int count = 0; //for cout
#endif
					for (int k = 0; k < K; k++)
					{
						if (label_count[k] != 0) continue;

						//std::cout << "empty: " << k << std::endl;

						int k_count_max = 0;
						for (int k1 = 1; k1 < K; k1++)
						{
							if (label_count[k_count_max] < label_count[k1])
								k_count_max = k1;
						}

						float max_dist = 0.f;
						int farthest_i = -1;
						float* base_centroids = centroids.ptr<float>(k_count_max);
						float* normalized_centroids = temp.ptr<float>(); // normalized
						const float count_normalize = 1.f / label_count[k_count_max];
						for (int j = 0; j < dims; j++)
						{
							normalized_centroids[j] = base_centroids[j] * count_normalize;
						}
						for (int i = 0; i < N; i++)
						{
							if (labels[i] != k_count_max) continue;

#ifdef DEBUG_SHOW_SKIP
							count++; //for cout 
#endif
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += (data_points.ptr<float>(d)[i] - normalized_centroids[d]) * (data_points.ptr<float>(d)[i] - normalized_centroids[d]);
							}

							if (max_dist <= dist)
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						label_count[k_count_max]--;
						label_count[k]++;
						labels[farthest_i] = k;

						float* cur_center = centroids.ptr<float>(k);

						for (int d = 0; d < dims; d++)
						{
							base_centroids[d] -= data_points.ptr<float>(d)[farthest_i];
							cur_center[d] += data_points.ptr<float>(d)[farthest_i];
						}
					}
#ifdef DEBUG_SHOW_SKIP
					cout << iter << ": compute " << count / ((float)N * K) * 100.f << " %" << endl;
#endif

					//normalization and compute max shift distance between old centroid and new centroid
					//small loop: K x d
					for (int k = 0; k < K; k++)
					{
						float* centroidsPtr = centroids.ptr<float>(k);
						CV_Assert(label_count[k] != 0);

						float count_normalize = 0.f;
						if (function == MeanFunction::Mean)
							count_normalize = 1.f / label_count[k];
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
				//if (max_center_shift <= criteria.epsilon)					cout << "exit (max_center_shift <= criteria.epsilon), iteration" << iter - 1 << endl;
				{
					//cp::Timer t(format("%d: distant computing", iter)); //last loop is fast
					if (isLastIter)
					{
						// compute distance only
						parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<true, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), cv::getNumThreads());
						compactness = sum(Mat(Size(N, 1), CV_32F, &dists[0]))[0];
						//getOuterSample(centroids, old_centroids, data_points, labels_internal);
						//swap(centroids, old_centroids);		
						break;
					}
					else
					{
						// assign labels
						//int parallel = CV_KMEANS_PARALLEL_GRANULARITY;
						int parallel = cv::getNumThreads();

						if (loop == KMeansDistanceLoop::KND)
						{
							switch (dims)
							{
							case 1:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 1>(dists.data(), labels, data_points, centroids), parallel); break;
							case 2:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 2>(dists.data(), labels, data_points, centroids), parallel); break;
							case 3:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 3>(dists.data(), labels, data_points, centroids), parallel); break;
							case 64:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 64>(dists.data(), labels, data_points, centroids), parallel); break;
							default:parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
							}
							//parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel);
						}
						else if (loop == KMeansDistanceLoop::NKD)
						{
							switch (dims)
							{
							case 1:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 1>(dists.data(), labels, data_points, centroids), parallel); break;
							case 2:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 2>(dists.data(), labels, data_points, centroids), parallel); break;
							case 3:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 3>(dists.data(), labels, data_points, centroids), parallel); break;
							case 64:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 64>(dists.data(), labels, data_points, centroids), parallel); break;
							default:parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
							}
							//parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel);
							//parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel);
						}
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
#pragma endregion

#pragma region AoS
	//static int CV_KMEANS_PARALLEL_GRANULARITY = (int)utils::getConfigurationParameterSizeT("OPENCV_KMEANS_PARALLEL_GRANULARITY", 1000);

#pragma region normL2Sqr
	inline float normL2Sqr_(const float* a, const float* b, const int avxend, const int issse, const int rem)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		__m256 msum = _mm256_mul_ps(v, v);
		for (int j = 0; j < avxend; j++)
		{
			__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
			msum = _mm256_fmadd_ps(v, v, msum);
			a += 8;
			b += 8;
		}
		float d = _mm256_reduceadd_ps(msum);
		if (issse)
		{
			__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
			v = _mm_mul_ps(v, v);
			d += _mm_reduceadd_ps(v);
			a += 4;
			b += 4;
		}
		for (int j = 0; j < rem; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}

		return d;
	}

	inline float normL2Sqr_(const float* a, const float* b, int n)
	{
		float d = 0.f;
		for (int j = 0; j < n; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}
		return d;
	}

	template<int n>
	inline float normL2Sqr_(const float* a, const float* b)
	{
		float d = 0.f;
		for (int j = 0; j < n; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}
		return d;
	}

	template<>
	inline float normL2Sqr_<3>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);

		return v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2];
		//return _mm_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<4>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		return _mm_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<5>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t = a[4] - b[4];
		return _mm_reduceadd_ps(v) + t * t;
	}

	template<>
	inline float normL2Sqr_<6>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t1 = a[4] - b[4];
		const float t2 = a[5] - b[5];
		return _mm_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<7>(const float* a, const float* b)
	{
		/*__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		v = *(__m256*) &_mm256_insert_epi32(*(__m256i*) & v, 0, 7);
		return _mm256_reduceadd_ps(v);*/

		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t1 = a[4] - b[4];
		const float t2 = a[5] - b[5];
		const float t3 = a[6] - b[6];
		return _mm_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<8>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<9>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		return _mm256_reduceadd_ps(v) + t1 * t1;
	}

	template<>
	inline float normL2Sqr_<10>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		const float t2 = a[9] - b[9];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<11>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		const float t2 = a[9] - b[9];
		const float t3 = a[10] - b[10];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<12>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m128 v2 = _mm_sub_ps(_mm_loadu_ps(a + 8), _mm_loadu_ps(b + 8));
		v2 = _mm_mul_ps(v2, v2);
		v = _mm256_add_ps(v, _mm256_castps128_ps256(v2));
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<16>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<24>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<32>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<40>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<41>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		return _mm256_reduceadd_ps(v) + t1 * t1;
	}

	template<>
	inline float normL2Sqr_<42>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		const float t2 = a[41] - b[41];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<43>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		const float t2 = a[41] - b[41];
		const float t3 = a[42] - b[42];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<44>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		__m128 v3 = _mm_sub_ps(_mm_loadu_ps(a + 40), _mm_loadu_ps(b + 40));
		v3 = _mm_mul_ps(v3, v3);
		v = _mm256_add_ps(v, _mm256_castps128_ps256(v3));
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<48>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<56>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<64>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

#pragma endregion

	void KMeans::generateKmeansRandomInitialCentroidAoS(const cv::Mat& data_points, Mat& dest_centroids, const int K, RNG& rng)
	{
		const int dims = data_points.cols;
		const int N = data_points.rows;
		cv::AutoBuffer<Vec2f, 64> box(dims);
		{
			const float* sample = data_points.ptr<float>(0);
			for (int j = 0; j < dims; j++)
				box[j] = Vec2f(sample[j], sample[j]);
		}
		for (int i = 1; i < N; i++)
		{
			const float* sample = data_points.ptr<float>(i);
			for (int j = 0; j < dims; j++)
			{
				float v = sample[j];
				box[j][0] = std::min(box[j][0], v);
				box[j][1] = std::max(box[j][1], v);
			}
		}

		const bool isUseMargin = false;//using margin is OpenCV's implementation
		if (isUseMargin)
		{
			const float margin = 1.f / dims;
			for (int k = 0; k < K; k++)
			{
				float* dptr = dest_centroids.ptr<float>(k);
				for (int d = 0; d < dims; d++)
				{
					dptr[d] = ((float)rng * (1.f + margin * 2.f) - margin) * (box[d][1] - box[d][0]) + box[d][0];
				}
			}
		}
		else
		{
			for (int k = 0; k < K; k++)
			{
				float* dptr = dest_centroids.ptr<float>(k);
				for (int d = 0; d < dims; d++)
				{
					dptr[d] = rng.uniform(box[d][0], box[d][1]);
				}
			}
		}

	}

	class KMeansPPDistanceComputerAoS : public ParallelLoopBody
	{
	public:
		KMeansPPDistanceComputerAoS(float* tdist2_, const Mat& data_, const float* dist_, int ci_) :
			tdist2(tdist2_), data(data_), dist(dist_), ci(ci_)
		{ }

		void operator()(const cv::Range& range) const CV_OVERRIDE
		{
			//CV_TRACE_FUNCTION();
			const int begin = range.start;
			const int end = range.end;
			const int dims = data.cols;

			for (int i = begin; i < end; i++)
			{
				tdist2[i] = std::min(normL2Sqr_(data.ptr<float>(i), data.ptr<float>(ci), dims), dist[i]);
			}
		}

	private:
		KMeansPPDistanceComputerAoS& operator=(const KMeansPPDistanceComputerAoS&); // = delete

		float* tdist2;
		const Mat& data;
		const float* dist;
		const int ci;
	};

	void KMeans::generateKmeansPPInitialCentroidAoS(const Mat& data, Mat& _out_centers, int K, RNG& rng, int trials)
	{
		//CV_TRACE_FUNCTION();
		const int dims = data.cols;
		const int N = data.rows;
		cv::AutoBuffer<int, 64> _centers(K);
		int* centers = &_centers[0];
		cv::AutoBuffer<float, 0> _dist(N * 3);
		float* dist = &_dist[0], * tdist = dist + N, * tdist2 = tdist + N;
		double sum0 = 0;

		centers[0] = (unsigned)rng % N;

		for (int i = 0; i < N; i++)
		{
			dist[i] = normL2Sqr_(data.ptr<float>(i), data.ptr<float>(centers[0]), dims);
			sum0 += dist[i];
		}

		for (int k = 1; k < K; k++)
		{
			double bestSum = DBL_MAX;
			int bestCenter = -1;

			for (int j = 0; j < trials; j++)
			{
				double p = (double)rng * sum0;
				int ci = 0;
				for (; ci < N - 1; ci++)
				{
					p -= dist[ci];
					if (p <= 0)
						break;
				}

				parallel_for_(Range(0, N),
					KMeansPPDistanceComputerAoS(tdist2, data, dist, ci),
					(double)divUp((size_t)(dims * N), CV_KMEANS_PARALLEL_GRANULARITY));
				double s = 0;
				for (int i = 0; i < N; i++)
				{
					s += tdist2[i];
				}

				if (s < bestSum)
				{
					bestSum = s;
					bestCenter = ci;
					std::swap(tdist, tdist2);
				}
			}
			if (bestCenter < 0)
				CV_Error(Error::StsNoConv, "kmeans: can't update cluster center (check input for huge or NaN values)");
			centers[k] = bestCenter;
			sum0 = bestSum;
			std::swap(dist, tdist);
		}

		for (int k = 0; k < K; k++)
		{
			const float* src = data.ptr<float>(centers[k]);
			float* dst = _out_centers.ptr<float>(k);
			for (int j = 0; j < dims; j++)
				dst[j] = src[j];
		}
	}


	template<bool onlyDistance, int loop>
	class KMeansDistanceComputerAoSDim : public ParallelLoopBody
	{
	public:
		KMeansDistanceComputerAoSDim(float* distances_,
			int* labels_,
			const Mat& data_,
			const Mat& centers_)
			: distances(distances_),
			labels(labels_),
			data(data_),
			centers(centers_)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			const int begin = range.start;
			const int end = range.end;
			const int K = centers.rows;
			const int dims = centers.cols;

			const int avxend = dims / 8;
			const int issse = (dims - avxend * 8) / 4;
			const int rem = dims - avxend * 8 - issse * 4;

			for (int i = begin; i < end; ++i)
			{
				const float* sample = data.ptr<float>(i);
				if (onlyDistance)
				{
					const float* center = centers.ptr<float>(labels[i]);
					distances[i] = normL2Sqr_(sample, center, dims);
					continue;
				}
				else
				{
					int k_best = 0;
					float min_dist = FLT_MAX;

					for (int k = 0; k < K; k++)
					{
						const float* center = centers.ptr<float>(k);
						const float dist = normL2Sqr_(sample, center, dims);
						//const float dist = normL2Sqr_(sample, center, avxend, issse, rem);

						if (min_dist > dist)
						{
							min_dist = dist;
							k_best = k;
						}
					}

					distances[i] = min_dist;
					labels[i] = k_best;
				}
			}
		}

	private:
		KMeansDistanceComputerAoSDim& operator=(const KMeansDistanceComputerAoSDim&); // = delete

		float* distances;
		int* labels;
		const Mat& data;
		const Mat& centers;
	};

	template<bool onlyDistance, int dims, int loop>
	class KMeansDistanceComputerAoS : public ParallelLoopBody
	{
	public:
		KMeansDistanceComputerAoS(float* distances_, int* labels_, const Mat& data_, const Mat& centers_)
			: distances(distances_), labels(labels_), data(data_), centers(centers_)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			const int begin = range.start;
			const int end = range.end;
			const int K = centers.rows;
			//n-k-d
			if (onlyDistance)
			{
				for (int n = begin; n < end; ++n)
				{
					const float* sample = data.ptr<float>(n);
					{
						const float* center = centers.ptr<float>(labels[n]);
						distances[n] = normL2Sqr_<dims>(sample, center);
						continue;
					}
				}
			}
			else
			{
				if (loop == KMeansDistanceLoop::NKD)
				{
					for (int n = begin; n < end; ++n)
					{
						const float* sample = data.ptr<float>(n);
						int k_best = 0;
						float min_dist = FLT_MAX;

						for (int k = 0; k < K; ++k)
						{
							const float* center = centers.ptr<float>(k);
							const float dist = normL2Sqr_<dims>(sample, center);

							if (min_dist > dist)
							{
								min_dist = dist;
								k_best = k;
							}
						}

						distances[n] = min_dist;
						labels[n] = k_best;
					}
				}
				else //k-n-d
				{
					{
						//int k = 0;
						const float* center = centers.ptr<float>(0);
						for (int n = begin; n < end; ++n)
						{
							const float* sample = data.ptr<float>(n);
							distances[n] = normL2Sqr_<dims>(sample, center);
							labels[n] = 0;
						}
					}
					for (int k = 1; k < K; ++k)
					{
						const float* center = centers.ptr<float>(k);
						for (int n = begin; n < end; ++n)
						{
							const float* sample = data.ptr<float>(n);
							const float dist = normL2Sqr_<dims>(sample, center);

							if (distances[n] > dist)
							{
								distances[n] = dist;
								labels[n] = k;
							}
						}
					}
				}
			}
		}

	private:
		KMeansDistanceComputerAoS& operator=(const KMeansDistanceComputerAoS&); // = delete

		float* distances;
		int* labels;
		const Mat& data;
		const Mat& centers;
	};

	void KMeans::boxMeanCentroidAoS(Mat& data_points, const int* labels, Mat& centroids, int* counters)
	{
		const int N = data_points.rows;
		const int dims = data_points.cols;

		for (int i = 0; i < N; i++)
		{
			const float* sample = data_points.ptr<float>(i);
			int k = labels[i];
			float* center = centroids.ptr<float>(k);
			for (int j = 0; j < dims; j++)
			{
				center[j] += sample[j];
			}
			counters[k]++;
		}
	}

	double KMeans::clusteringAoS(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, int loop)
	{
		const int SPP_TRIALS = 3;
		Mat src = _data.getMat();
		const bool isrow = (src.rows == 1);
		const int N = isrow ? src.cols : src.rows;
		const int dims = (isrow ? 1 : src.cols) * src.channels();
		const int type = src.depth();

		attempts = std::max(attempts, 1);
		CV_Assert(src.dims <= 2 && type == CV_32F && K > 0);
		CV_CheckGE(N, K, "Number of clusters should be more than number of elements");

		Mat data_points(N, dims, CV_32F, src.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(src.step));

		_bestLabels.create(N, 1, CV_32S, -1, true);
		Mat best_labels = _bestLabels.getMat();

		if (flags & cv::KMEANS_USE_INITIAL_LABELS)
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
		else
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
		Mat old_centroids(K, dims, type), temp(1, dims, type);
		cv::AutoBuffer<int, 64> counters(K);
		cv::AutoBuffer<float, 64> dists(N);
		//dists.resize(N);
		RNG& rng = theRNG();

		if (criteria.type & TermCriteria::EPS) criteria.epsilon = std::max(criteria.epsilon, 0.);
		else criteria.epsilon = FLT_EPSILON;

		criteria.epsilon *= criteria.epsilon;

		if (criteria.type & TermCriteria::COUNT) criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
		else criteria.maxCount = 100;

		if (K == 1)
		{
			attempts = 1;
			criteria.maxCount = 2;
		}

		double best_compactness = DBL_MAX;
		for (int attempt_index = 0; attempt_index < attempts; attempt_index++)
		{
			double compactness = 0.0;

			//main loop
			for (int iter = 0; ;)
			{
				float max_center_shift = (iter == 0) ? FLT_MAX : 0.f;

				swap(centroids, old_centroids);

				const bool isInit = ((iter == 0) && (attempt_index > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)));//initial attemp && KMEANS_USE_INITIAL_LABELS is true
				if (isInit)
				{
					if (flags & KMEANS_PP_CENTERS)
					{
						generateKmeansPPInitialCentroidAoS(data_points, centroids, K, rng, SPP_TRIALS);
					}
					else
					{
						generateKmeansRandomInitialCentroidAoS(data_points, centroids, K, rng);
					}
				}
				else
				{
					//update centroid 
					centroids = Scalar(0.f);
					for (int k = 0; k < K; k++) counters[k] = 0;
					if (function == MeanFunction::Harmonic)
					{
						cout << "MeanFunction::Harmonic not support" << endl;
					}
					else if (function == MeanFunction::Gauss || function == MeanFunction::GaussInv)
					{
						cout << "MeanFunction::Gauss/MeanFunction::GaussInv not support" << endl;
					}
					else if (function == MeanFunction::Mean)
					{
						boxMeanCentroidAoS(data_points, labels, centroids, counters);
					}

					for (int k = 0; k < K; k++)
					{
						if (counters[k] != 0)
							continue;

						// if some cluster appeared to be empty then:
						//   1. find the biggest cluster
						//   2. find the farthest from the center point in the biggest cluster
						//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
						int k_count_max = 0;
						for (int k1 = 1; k1 < K; k1++)
						{
							if (counters[k_count_max] < counters[k1])
								k_count_max = k1;
						}

						double max_dist = 0;
						int farthest_i = -1;
						float* base_center = centroids.ptr<float>(k_count_max);
						float* _base_center = temp.ptr<float>(); // normalized
						float scale = 1.f / counters[k_count_max];
						for (int j = 0; j < dims; j++)
							_base_center[j] = base_center[j] * scale;

						for (int i = 0; i < N; i++)
						{
							if (labels[i] != k_count_max)
								continue;
							const float* sample = data_points.ptr<float>(i);
							double dist = normL2Sqr_(sample, _base_center, dims);

							if (max_dist <= dist)
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[k_count_max]--;
						counters[k]++;
						labels[farthest_i] = k;

						const float* sample = data_points.ptr<float>(farthest_i);
						float* cur_center = centroids.ptr<float>(k);
						for (int j = 0; j < dims; j++)
						{
							base_center[j] -= sample[j];
							cur_center[j] += sample[j];
						}
					}

					for (int k = 0; k < K; k++)
					{
						float* center = centroids.ptr<float>(k);
						CV_Assert(counters[k] != 0);

						float scale = 1.f / counters[k];
						for (int j = 0; j < dims; j++)
							center[j] *= scale;

						if (iter > 0)
						{
							float dist = 0.f;
							const float* old_center = old_centroids.ptr<float>(k);
							for (int j = 0; j < dims; j++)
							{
								float t = center[j] - old_center[j];
								dist += t * t;
							}
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);

				if (isLastIter)
				{
					//int parallel = (double)divUp((size_t)(dims * N * K), CV_KMEANS_PARALLEL_GRANULARITY);
					int parallel = cv::getNumThreads();
					// don't re-assign labels to avoid creation of empty clusters
					parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<true, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel);
					compactness = sum(Mat(Size(N, 1), CV_32F, &dists[0]))[0];
					break;
				}
				else
				{
					//int parallel = (double)divUp((size_t)(dims * N * K), CV_KMEANS_PARALLEL_GRANULARITY);
					int parallel = cv::getNumThreads();
					// assign labels
					if (loop == KMeansDistanceLoop::NKD)
					{
						switch (dims)
						{
						case 1:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 1, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 2:  parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 2, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 3:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 3, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 4:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 4, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 5:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 5, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 6:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 6, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 7:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 7, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 8:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 8, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 9:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 9, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 10: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 10, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 11: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 11, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 12: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 12, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 16: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 16, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 24: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 24, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 32: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 32, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 40: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 40, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 41: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 41, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 42: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 42, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 43: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 43, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 44: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 44, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 48: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 48, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 56: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 56, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 64: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 64, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						default: parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						}
					}
					else
					{
						switch (dims)
						{
						case 1:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 1, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 2:  parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 2, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 3:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 3, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 4:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 4, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 5:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 5, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 6:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 6, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 7:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 7, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 8:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 8, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 9:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 9, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 10: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 10, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 11: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 11, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 12: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 12, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 16: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 16, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 24: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 24, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 32: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 32, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 40: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 40, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 41: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 41, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 42: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 42, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 43: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 43, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 44: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 44, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 48: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 48, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 56: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 56, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 64: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 64, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						default: parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<false, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						}
					}
				}
			}

			if (compactness < best_compactness)
			{
				best_compactness = compactness;
				if (_centers.needed())
				{
					if (_centers.fixedType() && _centers.channels() == dims)
						centroids.reshape(dims).copyTo(_centers);
					else
						centroids.copyTo(_centers);
				}
				labels_internal.copyTo(best_labels);
			}
		}

		return best_compactness;
	}
#pragma endregion

#pragma region SoAoS
	double KMeans::clusteringSoAoS(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, int loop)
	{
		cout << "not implemented clusteringSoAoS" << endl;
		return 0.0;
	}
#pragma endregion
#
}