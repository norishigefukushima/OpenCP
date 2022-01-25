#include "GaussianKDTree.hpp"
#include "inlineSIMDFunctions.hpp"
#include "tiling.hpp"
#include "debugcp.hpp"
#include "count.hpp"
using namespace std;
using namespace cv;

namespace cp
{
#define INF (std::numeric_limits<float>::infinity())

	RNG rng;
	float rand_float()
	{
		return rng.uniform(0.f, 1.f);
		//return rand() / (RAND_MAX + 1.0f);
	}

	//piecewise cubic approximation of Gaussian
	static inline float gCDF(float x)
	{
		x *= 0.81649658092772592f;
		if (x < -2.f) return 0.f;
		if (x < -1.f)
		{
			x += 2.f;
			x *= x;
			x *= x;
			return x;
		}
		if (x < 0.f) return 12.f + x * (16.f - x * x * (8.f + 3.f * x));
		if (x < 1.f) return 12.f + x * (16.f - x * x * (8.f - 3.f * x));
		if (x < 2.f)
		{
			x = x - 2.f;
			x *= x;
			x *= x;
			return -x + 24.f;
		}
		return 24.f;
	}

	class GKDTree
	{
	private:

		class Node
		{
		public:
			virtual ~Node() {}

			// Returns a list of samples from the kdtree distributed
			// around value with std-dev sigma in all dimensions. Some
			// samples may be repeated. Returns how many entries in the
			// ids and weights arrays were used.
			virtual int GaussianLookup(const float* value, int** ids, float** weights, const int nSamples, const float p) = 0;

			// special case optimization of the above where nsamples = 1
			virtual int singleGaussianLookup(const float* value, int** ids, float** weights, const float p) = 0;
			virtual void computeBounds(float* mins, float* maxs) = 0;
		};

		class Split : public Node
		{
		public:
			virtual ~Split()
			{
				delete left;
				delete right;
			}

			// for a given gaussian and a given value, the probability of splitting left at this node
			inline float pLeft(const float value)
			{
				// Coarsely approximate the cumulative normal distribution
				const float val = gCDF(cut_val - value);
				const float minBound = gCDF(min_val - value);
				const float maxBound = gCDF(max_val - value);
				return (val - minBound) / (maxBound - minBound + FLT_EPSILON);
			}

			int GaussianLookup(const float* value, int** ids, float** weights, const int nSamples, const float p)
			{
				// Calculate how much of a gaussian ball of radius sigma,
				// that has been trimmed by all the cuts so far, lies on
				// each side of the split

				// compute the probability of a sample splitting left
				const float val = pLeft(value[cut_dim]);

				// Send some samples to the left of the split
				int leftSamples = (int)(val * nSamples);

				// Send some samples to the right of the split
				int rightSamples = (int)((1.f - val) * nSamples);

				// There's probably one sample left over by the rounding
				if (leftSamples + rightSamples != nSamples)
				{
					const float fval = val * nSamples - leftSamples;
					// if val is high we send it left, if val is low we send it right
					if (rand_float() < fval)
					{
						leftSamples++;
					}
					else
					{
						rightSamples++;
					}
				}

				int samplesFound = 0;
				// Get the left samples
				if (leftSamples > 0)
				{
					if (leftSamples > 1)
					{
						samplesFound += left->GaussianLookup(value, ids, weights, leftSamples, p * val);
					}
					else
					{
						samplesFound += left->singleGaussianLookup(value, ids, weights, p * val);
					}
				}

				// Get the right samples
				if (rightSamples > 0)
				{
					if (rightSamples > 1)
					{
						samplesFound += right->GaussianLookup(value, ids, weights, rightSamples, p * (1 - val));
					}
					else
					{
						samplesFound += right->singleGaussianLookup(value, ids, weights, p * (1 - val));
					}
				}

				return samplesFound;
			}

			// a special case optimization of the above for when nSamples is 1
			int singleGaussianLookup(const float* value, int** ids, float** weights, const float p)
			{
				const float val = pLeft(value[cut_dim]);
				if (rand_float() < val)
				{
					return left->singleGaussianLookup(value, ids, weights, p * val);
				}
				else
				{
					return right->singleGaussianLookup(value, ids, weights, p * (1.f - val));
				}
			}

			void computeBounds(float* mins, float* maxs)
			{
				min_val = mins[cut_dim];
				max_val = maxs[cut_dim];

				maxs[cut_dim] = cut_val;
				left->computeBounds(mins, maxs);
				maxs[cut_dim] = max_val;

				mins[cut_dim] = cut_val;
				right->computeBounds(mins, maxs);
				mins[cut_dim] = min_val;
			}

			int cut_dim;
			float cut_val, min_val, max_val;
			Node* left, * right;
		};

		class Leaf : public Node
		{
		private:
			int id;
			int dimensions;
			float* position;
		public:
			Leaf(int id_, const float** data, int nData, int dimensions_)
				: id(id_), dimensions(dimensions_)
			{
				position = new float[dimensions];
				for (int i = 0; i < dimensions; i++)
				{
					position[i] = 0.f;
					for (int j = 0; j < nData; j++)
					{
						position[i] += data[j][i];
					}
					position[i] /= nData;
				}
			}
			~Leaf()
			{
				delete[] position;
			}

			int GaussianLookup(const float* query, int** ids, float** weights, const int nSamples, const float p)
			{
				// p is the probability with which one sample arrived here
				// calculate the correct probability, q

				float q = 0.f;
				for (int i = 0; i < dimensions; i++)
				{
					const float diff = query[i] - position[i];
					q += diff * diff;
				}

				// Gaussian of variance 1/2
				q = exp(-q);

				*(*ids)++ = id;
				*(*weights)++ = nSamples * q / p;

				return 1;
			}

			int singleGaussianLookup(const float* query, int** ids, float** weights, const float p)
			{
				return GaussianLookup(query, ids, weights, 1, p);
			}

			void computeBounds(float* mins, float* maxs)
			{
			}
		};

		Node* root;
		int dimensions;
		float sizeBound;
		int leaves;

		Node* build(const float** data, const int numData)
		{
			if (numData == 1)
			{
				return new Leaf(leaves++, data, numData, dimensions);
			}
			else
			{
				AutoBuffer<float> mins(dimensions), maxs(dimensions);

				// calculate the data bounds in every dimension
				//j=0
				for (int i = 0; i < dimensions; i++)
				{
					mins[i] = maxs[i] = data[0][i];
				}
				for (int j = 1; j < numData; j++)
				{
					for (int i = 0; i < dimensions; i++)
					{
						mins[i] = min(mins[i], data[j][i]);
						maxs[i] = max(maxs[i], data[j][i]);
					}
				}

				// find the longest dimension
				int longest = 0;
				for (int i = 1; i < dimensions; i++)
				{
					const float delta = maxs[i] - mins[i];
					if (delta > maxs[longest] - mins[longest])
						longest = i;
				}

				// if it's large enough, cut in that dimension
				const float maxl = maxs[longest];
				const float minl = mins[longest];
				if (maxl - minl > sizeBound)
				{
					Split* n = new Split;
					n->cut_dim = longest;
					n->cut_val = (maxl + minl) * 0.5f;

					// these get computed later
					n->min_val = -INF;
					n->max_val = INF;

					// resort the input over the split
					int pivot = 0;
					for (int i = 0; i < numData; i++)
					{
						// The next value is larger than the pivot
						if (data[i][longest] >= n->cut_val) continue;

						// We haven't seen anything larger than the pivot yet
						if (i == pivot)
						{
							pivot++;
							continue;
						}

						// The current value is smaller than the pivot
						swap(data[i], data[pivot]);
						pivot++;
					}

					// Build the two subtrees
					n->left = build(data, pivot);
					n->right = build(data + pivot, numData - pivot);
					return n;
				}
				else
				{
					return new Leaf(leaves++, data, numData, dimensions);
				}
			}
		};
	public:
		// Build a gkdtree using the supplied array of points to control
		// the sampling.  sizeBound specifies the maximum allowable side
		// length of a kdtree leaf.  At least one point from data lies in
		// any given leaf.
		GKDTree(int dims, const float** data, int numData, float sBound) :
			dimensions(dims), sizeBound(sBound), leaves(0)
		{
			root = build(data, numData);
		}

		~GKDTree()
		{
			delete root;
		}

		void finalize()
		{
			AutoBuffer<float> kdtreeMins(dimensions);
			AutoBuffer<float> kdtreeMaxs(dimensions);

			for (int i = 0; i < dimensions; i++)
			{
				kdtreeMins[i] = -INF;
				kdtreeMaxs[i] = +INF;
			}

			root->computeBounds(kdtreeMins, kdtreeMaxs);
		}

		int getLeaves()
		{
			return leaves;
		}

		// Compute a gaussian spread of kdtree leaves around the given
		// point. This is the general case sampling strategy.
		int GaussianLookup(const float* value, int* ids, float* weights, int nSamples)
		{
			return root->GaussianLookup(value, &ids, &weights, nSamples, 1.f);
		}

		static void filter(const Mat& src, const Mat& ref, Mat& dest,
			const int SPLAT_ACCURACY = 4,
			const int SLICE_ACCURACY = 64,
			const float th_rho = sqrt(2.f) * 0.5f)
		{
			dest.create(src.size(), src.type());
			const int chs = src.channels();
			const int chr = ref.channels();

			AutoBuffer<const float*> points(ref.size().area());
			int i = 0;
			for (int y = 0; y < ref.rows; y++)
			{
				for (int x = 0; x < ref.cols; x++)
				{
					points[i++] = ref.ptr<float>(y, x);
				}
			}

			GKDTree tree(ref.channels(), points, (int)points.size(), th_rho);
			tree.finalize();

			AutoBuffer<int> indices(SLICE_ACCURACY);
			AutoBuffer<float> weights(SLICE_ACCURACY);

			Mat leafValues = Mat::zeros(tree.getLeaves(), 1, CV_MAKETYPE(CV_32F, chs + 1));//leaves x 1 x (ch+1)

			const float* imPtr = src.ptr<float>();
			const float* refPtr = ref.ptr<float>();
			for (int y = 0; y < src.rows; y++)
			{
				for (int x = 0; x < src.cols; x++)
				{
					const int results = tree.GaussianLookup(refPtr, indices, weights, SPLAT_ACCURACY);
					for (int i = 0; i < results; i++)
					{
						const float w = weights[i];
						float* vPtr = leafValues.ptr<float>(indices[i]);
						for (int c = 0; c < chs; c++)
						{
							vPtr[c] += imPtr[c] * w;
						}
						vPtr[chs] += w;
					}
					refPtr += chr;
					imPtr += chs;
				}
			}

			const float* slicePtr = ref.ptr<float>();
			float* outPtr = dest.ptr<float>();
			for (int y = 0; y < dest.rows; y++)
			{
				for (int x = 0; x < dest.cols; x++)
				{
					const int results = tree.GaussianLookup(slicePtr, indices, weights, SLICE_ACCURACY);
					float outW = 0.f;
					for (int i = 0; i < results; i++)
					{
						const float w = weights[i];
						float* vPtr = leafValues.ptr<float>(indices[i]);

						for (int c = 0; c < chs; c++)
						{
							outPtr[c] += vPtr[c] * w;
						}
						outW += w * vPtr[chs];
					}

					if (abs(outW) < 0.00001f || cvIsNaN(outW) || cvIsInf(outW))
					{
						for (int c = 0; c < chs; c++)
						{
							outPtr[c] = src.ptr<float>(y, x)[c];
						}
					}
					else
					{
						const float invOutW = 1.f / outW;
						for (int c = 0; c < chs; c++)
						{
							const float v = outPtr[c] * invOutW;
							if (0.f <= v && v<= 255.f) outPtr[c] = v;
							else outPtr[c] = src.ptr<float>(y, x)[c];
						}
					}
					slicePtr += chr;
					outPtr += chs;
				}
			}
		}
	};

	void highDimensionalGaussianFilterGaussianKDTree(const Mat& src, const Mat& guide, Mat& dest, const float sigma_color, const float sigma_space)
	{
		dest.create(src.size(), src.type());

		const float invSpatialStdev = 1.0f / sigma_space;
		const float invColorStdev = 1.0f / (sigma_color);

		Mat ref(src.size(), CV_MAKETYPE(CV_32F, guide.channels() + 2));
		if (src.depth() == CV_8U)
		{
			for (int y = 0; y < src.rows; y++)
			{
				for (int x = 0; x < src.cols; x++)
				{
					ref.ptr<float>(y, x)[0] = invSpatialStdev * x;
					ref.ptr<float>(y, x)[1] = invSpatialStdev * y;
					for (int c = 0; c < guide.channels(); c++)
					{
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * (float)guide.at<uchar>(y, guide.channels() * x + c);
					}
				}
			}
		}
		else if (src.depth() == CV_32F)
		{
			for (int y = 0; y < src.rows; y++)
			{
				for (int x = 0; x < src.cols; x++)
				{
					ref.ptr<float>(y, x)[0] = invSpatialStdev * x;
					ref.ptr<float>(y, x)[1] = invSpatialStdev * y;
					for (int c = 0; c < guide.channels(); c++)
					{
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * guide.at<float>(y, guide.channels() * x + c);
					}
				}
			}
		}

		// Filter the input with respect to the position vectors. 
		if (src.depth() == CV_8U)
		{
			Mat src32f; src.convertTo(src32f, CV_32F);
			Mat dst32f(src.size(), src32f.type());
			GKDTree::filter(src32f, ref, dst32f);
			dst32f.convertTo(dest, CV_8U);
		}
		else
		{
			GKDTree::filter(src, ref, dest);
		}
	}

	void highDimensionalGaussianFilterGaussianKDTree(const Mat& src, Mat& dest, const float sigma_color, const float sigma_space)
	{
		highDimensionalGaussianFilterGaussianKDTree(src, src, dest, sigma_color, sigma_space);
	}

	void highDimensionalGaussianFilterGaussianKDTree(const vector<Mat>& vsrc, const vector<Mat>& vguide, Mat& dest, const float sigma_color, const float sigma_space)
	{
		Mat src; merge(vsrc, src);
		Mat ref; merge(vguide, ref);
		highDimensionalGaussianFilterGaussianKDTree(src, ref, dest, sigma_color, sigma_space);
	}

	void highDimensionalGaussianFilterGaussianKDTreeTile(const Mat& src, const Mat& guide, Mat& dest, const float sigma_color, const float sigma_space, const Size div, const float truncateBoundary)
	{
		const int channels = src.channels();
		const int guide_channels = guide.channels();

		dest.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int borderType = cv::BORDER_REFLECT;
		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			highDimensionalGaussianFilterGaussianKDTree(src, guide, dest, sigma_color, sigma_space);
		}
		else
		{
			int r = (int)ceil(truncateBoundary * sigma_space);
			const int R = get_simd_ceil(r, 8);
			Size tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
			Size divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

			vector<Mat> split_dst(channels);

			for (int c = 0; c < channels; c++)
			{
				split_dst[c].create(tileSize, CV_32FC1);
			}

			const int thread_max = omp_get_max_threads();
			vector<vector<Mat>>	subImageInput(thread_max);
			vector<vector<Mat>>	subImageGuide(thread_max);
			vector<Mat>	subImageOutput(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageGuide[n].resize(guide_channels);
				subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
			}

			std::vector<cv::Mat> srcSplit;
			std::vector<cv::Mat> guideSplit;
			if (src.channels() != 3)split(src, srcSplit);
			if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
			for (int n = 0; n < div.area(); n++)
			{
				const int thread_num = omp_get_thread_num();
				const cv::Point idx = cv::Point(n % div.width, n / div.width);


				if (src.channels() == 3)
				{
					cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < srcSplit.size(); c++)
					{
						cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}
				if (guide.channels() == 3)
				{
					cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
				else
				{
					for (int c = 0; c < guideSplit.size(); c++)
					{
						cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
					}
				}

				highDimensionalGaussianFilterGaussianKDTree(subImageInput[thread_num], subImageGuide[thread_num], subImageOutput[thread_num], sigma_color, sigma_space);

				cp::pasteTileAlign(subImageOutput[thread_num], dest, div, idx, r, 8, 8);
			}
		}
	}
}