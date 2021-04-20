#include "PermutohedralLattice.hpp"
#include "inlineSIMDFunctions.hpp"
#include "tiling.hpp"
#include "timer.hpp"
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;

namespace cp
{
	/***************************************************************/
	/* Hash table implementation for permutohedral lattice
	*
	* The lattice points are stored sparsely using a hash table.
	* The key for each point is its spatial location in the (d+1)-
	* dimensional space.
	*/
	/***************************************************************/
	class HashTablePermutohedral
	{
	public:
		/* Constructor
		*  kd_: the dimensionality of the position vectors on the hyperplane.
		*  vd_: the dimensionality of the value vectors
		*/
		HashTablePermutohedral(int kd_, int vd_) : kd(kd_), vd(vd_)
		{
			capacity = 1 << 15;
			filled = 0;
			entries = new Entry[capacity];
			keys = new short[kd * capacity / 2];
			values = new float[vd * capacity / 2];
			memset(values, 0, sizeof(float) * vd * capacity / 2);
		}
		~HashTablePermutohedral()
		{
			delete[] entries;
			delete[] keys;
			delete[] values;
		}

		// Returns the number of vectors stored.
		int size() { return (int)filled; }

		// Returns a pointer to the keys array.
		short* getKeys() { return keys; }

		// Returns a pointer to the values array.
		float* getValues() { return values; }

		/* Returns the index into the hash table for a given key.
		*     key: a pointer to the position vector.
		*       h: hash of the position vector.
		*  create: a flag specifying whether an entry should be created,
		*          should an entry with the given key not found.
		*/
		int lookupOffset(short* key, size_t h, bool create = true)
		{

			// Double hash table size if necessary
			if (filled >= (capacity / 2) - 1) { grow(); }

			// Find the entry with the given key
			while (1)
			{
				Entry e = entries[h];
				// check if the cell is empty
				if (e.keyIdx == -1)
				{
					if (!create) return -1; // Return not found.
					// need to create an entry. Store the given key.
					for (int i = 0; i < kd; i++)
						keys[filled * kd + i] = key[i];

					e.keyIdx = (int)filled * kd;
					e.valueIdx = (int)filled * vd;
					entries[h] = e;
					filled++;
					return e.valueIdx;
				}

				// check if the cell has a matching key
				bool match = true;
				for (int i = 0; i < kd && match; i++)
					match = keys[e.keyIdx + i] == key[i];
				if (match)
					return e.valueIdx;

				// increment the bucket with wraparound
				h++;
				if (h == capacity) h = 0;
			}
		}

		/* Looks up the value vector associated with a given key vector.
		*        k : pointer to the key vector to be looked up.
		*   create : true if a non-existing key should be created.
		*/
		float* lookup(short* k, bool create = true)
		{
			size_t h = hash(k) % capacity;
			int offset = lookupOffset(k, h, create);
			if (offset < 0) return NULL;
			else return values + offset;
		};

		/* Hash function used in this implementation. A simple base conversion. */
		size_t hash(const short* key)
		{
			size_t k = 0;
			for (int i = 0; i < kd; i++)
			{
				k += key[i];
				k *= 2531011;
			}
			return k;
		}

	private:
		/* Grows the size of the hash table */
		void grow()
		{
			//printf("Resizing hash table\n");

			size_t oldCapacity = capacity;
			capacity *= 2;

			// Migrate the value vectors.
			float* newValues = new float[vd * capacity / 2];
			memset(newValues, 0, sizeof(float) * vd * capacity / 2);
			memcpy(newValues, values, sizeof(float) * vd * filled);
			delete[] values;
			values = newValues;

			// Migrate the key vectors.
			short* newKeys = new short[kd * capacity / 2];
			memcpy(newKeys, keys, sizeof(short) * kd * filled);
			delete[] keys;
			keys = newKeys;

			Entry* newEntries = new Entry[capacity];

			// Migrate the table of indices.
			for (size_t i = 0; i < oldCapacity; i++)
			{
				if (entries[i].keyIdx == -1) continue;
				size_t h = hash(keys + entries[i].keyIdx) % capacity;
				while (newEntries[h].keyIdx != -1)
				{
					h++;
					if (h == capacity) h = 0;
				}
				newEntries[h] = entries[i];
			}
			delete[] entries;
			entries = newEntries;
		}

		// Private struct for the hash table entries.
		struct Entry {
			Entry() : keyIdx(-1), valueIdx(-1) {}
			int keyIdx;
			int valueIdx;
		};

		short* keys;
		float* values;
		Entry* entries;
		size_t capacity, filled;
		int kd, vd;
	};

	/***************************************************************/
	/* The algorithm class that performs the filter
	*
	* PermutohedralLattice::filter(...) does all the work.
	*
	*/
	/***************************************************************/
	class PermutohedralLattice
	{
	public:
		/* Filters given image against a reference image.
		*   src : image to be filtered.
		*  ref : reference image whose edges are to be respected.
		*/
		static void filter(const Mat& src, const Mat& ref, Mat& dest)
		{
			const int src_channels = src.channels();
			const int ref_channels = ref.channels();
			// Create lattice

			PermutohedralLattice lattice(ref.channels(), src.channels() + 1, src.cols * src.rows);

			// Splat into the lattice
			AutoBuffer<float> col(src_channels + 1);
			col[src_channels] = 1.f; // homogeneous coordinate

			{
				//Timer t("Splatting");
				const float* srcPtr = src.ptr<float>();
				const float* refPtr = ref.ptr<float>();
				for (int y = 0; y < src.rows; y++)
				{
					for (int x = 0; x < src.cols; x++)
					{
						memcpy(col, srcPtr, sizeof(float) * src_channels);
						//lattice.splat_<5>(refPtr, col);
						lattice.splat(refPtr, col);
						srcPtr += src_channels;
						refPtr += ref_channels;
					}
				}
			}

			// Blur the lattice
			{
				//Timer t("Blurring");
				lattice.blur();
			}


			// Slice from the lattice
			{
				//Timer t("Slicing");
				lattice.beginSlice();
				float* dst = dest.ptr<float>();
				for (int y = 0; y < src.rows; y++)
				{
					for (int x = 0; x < src.cols; x++)
					{
						lattice.slice(col);
						const float scale = 1.0f / col[src_channels];
						for (int c = 0; c < src_channels; c++)
						{
							*dst++ = col[c] * scale;
						}
					}
				}
			}
		}

		/* Constructor
		*     d_ : dimensionality of key vectors
		*    vd_ : dimensionality of value vectors
		* nData_ : number of points in the input
		*/
		PermutohedralLattice(int d_, int vd_, int nData_) :
			dim(d_), sdim(vd_), nData(nData_), hashTable(d_, vd_)
		{

			CV_Assert(dim < 127);
			CV_Assert(sdim < 127);
			// Allocate storage for various arrays
			elevated = new float[dim + 1];
			scaleFactor = new float[dim];

			greedy = new short[dim + 1];
			rank = new char[dim + 1];
			barycentric = new float[dim + 2];
			replay = new ReplayEntry[nData * (dim + 1)];
			nReplay = 0;
			canonical = new short[(dim + 1) * (dim + 1)];
			key = new short[dim + 1];

			// compute the coordinates of the canonical simplex, in which
			// the difference between a contained point and the zero
			// remainder vertex is always in ascending order. (See pg.4 of paper.)
			for (int i = 0; i <= dim; i++)
			{
				for (int j = 0; j <= dim - i; j++)
				{
					canonical[i * (dim + 1) + j] = i;
				}
				for (int j = dim - i + 1; j <= dim; j++)
				{
					canonical[i * (dim + 1) + j] = i - (dim + 1);
				}
			}

			// Compute parts of the rotation matrix E. (See pg.4-5 of paper.)      
			for (int i = 0; i < dim; i++)
			{
				// the diagonal entries for normalization
				scaleFactor[i] = 1.0f / (sqrtf((float)(i + 1) * (i + 2)));

				/* We presume that the user would like to do a Gaussian blur of standard deviation
				* 1 in each dimension (or a total variance of d, summed over dimensions.)
				* Because the total variance of the blur performed by this algorithm is not d,
				* we must scale the space to offset this.
				*
				* The total variance of the algorithm is (See pg.6 and 10 of paper):
				*  [variance of splatting] + [variance of blurring] + [variance of splatting]
				*   = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
				*   = 2d(d+1)(d+1)/3.
				*
				* So we need to scale the space by (d+1)sqrt(2/3).
				*/
				scaleFactor[i] *= (dim + 1) * sqrtf(2.0f / 3.f);
			}
		}

		~PermutohedralLattice()
		{
			delete[] elevated;
			delete[] scaleFactor;
			delete[] greedy;
			delete[] rank;
			delete[] barycentric;
			delete[] replay;
			delete[] canonical;
			delete[] key;
		}

		/* Performs splatting with given position and value vectors */
		void splat(const float* position, const float* value)
		{
			const int dim1 = dim + 1;//dim + 1
			// first rotate position into the (d+1)-dimensional hyperplane
			elevated[dim] = -dim * position[dim - 1] * scaleFactor[dim - 1];

			for (int i = dim - 1; i > 0; i--)
			{
				elevated[i] = (
					elevated[i + 1]
					- i * position[i - 1] * scaleFactor[i - 1]
					+ (i + 2) * position[i] * scaleFactor[i]
					);
			}
			elevated[0] = elevated[1] + 2.f * position[0] * scaleFactor[0];

			// prepare to find the closest lattice points
			const float scale = 1.f / (dim1);
			char* myrank = rank;
			short* mygreedy = greedy;

			// greedily search for the closest zero-colored lattice point
			int sum = 0;

			for (int i = 0; i <= dim; i++)
			{
				float v = elevated[i] * scale;
				float up = ceilf(v) * (dim1);
				float down = floorf(v) * (dim1);

				mygreedy[i] = (up - elevated[i] < elevated[i] - down) ? (short)up : (short)down;
				sum += mygreedy[i];
			}
			sum /= dim1;

			// rank differential to find the permutation between this simplex and the canonical one.
			// (See pg. 3-4 in paper.)
			memset(myrank, 0, sizeof(char) * (dim1));
			for (int i = 0; i < dim; i++)
			{
				for (int j = i + 1; j <= dim; j++)
				{
					if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j])
					{
						myrank[i]++;
					}
					else
					{
						myrank[j]++;
					}
				}
			}

			if (sum > 0)
			{
				// sum too large - the point is off the hyperplane.
				// need to bring down the ones with the smallest differential
				for (int i = 0; i <= dim; i++)
				{
					if (myrank[i] >= dim1 - sum)
					{
						mygreedy[i] -= dim1;
						myrank[i] += sum - (dim1);
					}
					else
					{
						myrank[i] += sum;
					}
				}
			}
			else if (sum < 0)
			{
				// sum too small - the point is off the hyperplane
				// need to bring up the ones with largest differential
				for (int i = 0; i <= dim; i++)
				{
					if (myrank[i] < -sum)
					{
						mygreedy[i] += dim1;
						myrank[i] += (dim1)+sum;
					}
					else
					{
						myrank[i] += sum;
					}
				}
			}

			// Compute barycentric coordinates (See pg.10 of paper.)
			memset(barycentric, 0, sizeof(float) * (dim + 2));
			for (int i = 0; i <= dim; i++)
			{
				barycentric[dim - myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
				barycentric[dim + 1 - myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
			}
			barycentric[0] += 1.f + barycentric[dim + 1];

			// Splat the value into each vertex of the simplex, with barycentric weights.
			for (int remainder = 0; remainder <= dim; remainder++)
			{
				// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
				for (int i = 0; i < dim; i++)
				{
					key[i] = mygreedy[i] + canonical[remainder * (dim1)+myrank[i]];
				}

				// Retrieve pointer to the value at this vertex.
				float* val = hashTable.lookup(key, true);

				// Accumulate values with barycentric weight.
				for (int i = 0; i < sdim; i++)
				{
					val[i] += barycentric[remainder] * value[i];
				}

				// Record this interaction to use later when slicing
				replay[nReplay].offset = (int)(val - hashTable.getValues());
				replay[nReplay].weight = barycentric[remainder];
				nReplay++;
			}
		}

		template<int DIM>
		void splat_(const float* position, const float* value)
		{
			// first rotate position into the (d+1)-dimensional hyperplane
			elevated[DIM] = -DIM * position[DIM - 1] * scaleFactor[DIM - 1];

			for (int i = DIM - 1; i > 0; i--)
			{
				elevated[i] = (elevated[i + 1] -
					i * position[i - 1] * scaleFactor[i - 1] +
					(i + 2) * position[i] * scaleFactor[i]);
			}
			elevated[0] = elevated[1] + 2.f * position[0] * scaleFactor[0];

			// prepare to find the closest lattice points
			const float scale = 1.f / (DIM + 1);
			char* myrank = rank;
			short* mygreedy = greedy;

			// greedily search for the closest zero-colored lattice point
			int sum = 0;

			for (int i = 0; i <= DIM; i++)
			{
				float v = elevated[i] * scale;
				float up = ceilf(v) * (DIM + 1);
				float down = floorf(v) * (DIM + 1);

				mygreedy[i] = (up - elevated[i] < elevated[i] - down) ? (short)up : (short)down;
				sum += mygreedy[i];
			}
			sum /= DIM + 1;

			// rank differential to find the permutation between this simplex and the canonical one.
			// (See pg. 3-4 in paper.)
			memset(myrank, 0, sizeof(char) * (DIM + 1));
			for (int i = 0; i < DIM; i++)
			{
				for (int j = i + 1; j <= DIM; j++)
				{
					if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j])
					{
						myrank[i]++;
					}
					else
					{
						myrank[j]++;
					}
				}
			}

			if (sum > 0)
			{
				// sum too large - the point is off the hyperplane.
				// need to bring down the ones with the smallest differential
				for (int i = 0; i <= DIM; i++)
				{
					if (myrank[i] >= DIM + 1 - sum)
					{
						mygreedy[i] -= DIM + 1;
						myrank[i] += sum - (DIM + 1);
					}
					else
					{
						myrank[i] += sum;
					}
				}
			}
			else if (sum < 0)
			{
				// sum too small - the point is off the hyperplane
				// need to bring up the ones with largest differential
				for (int i = 0; i <= DIM; i++)
				{
					if (myrank[i] < -sum)
					{
						mygreedy[i] += DIM + 1;
						myrank[i] += (DIM + 1) + sum;
					}
					else
					{
						myrank[i] += sum;
					}
				}
			}

			// Compute barycentric coordinates (See pg.10 of paper.)
			memset(barycentric, 0, sizeof(float) * (DIM + 2));
			for (int i = 0; i <= DIM; i++)
			{
				barycentric[DIM - myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
				barycentric[DIM + 1 - myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
			}
			barycentric[0] += 1.f + barycentric[DIM + 1];

			// Splat the value into each vertex of the simplex, with barycentric weights.
			for (int remainder = 0; remainder <= DIM; remainder++)
			{
				// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
				for (int i = 0; i < DIM; i++)
				{
					key[i] = mygreedy[i] + canonical[remainder * (DIM + 1) + myrank[i]];
				}

				// Retrieve pointer to the value at this vertex.
				float* val = hashTable.lookup(key, true);

				// Accumulate values with barycentric weight.
				for (int i = 0; i < sdim; i++)
				{
					val[i] += barycentric[remainder] * value[i];
				}

				// Record this interaction to use later when slicing
				replay[nReplay].offset = (int)(val - hashTable.getValues());
				replay[nReplay].weight = barycentric[remainder];
				nReplay++;
			}
		}

		// Prepare for slicing
		void beginSlice()
		{
			nReplay = 0;
		}

		/* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
		* containing each position vector were calculated and stored in the splatting step.
		* We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
		*/
		void slice(float* col)
		{
			float* base = hashTable.getValues();
			for (int j = 0; j < sdim; j++) col[j] = 0;

			for (int i = 0; i <= dim; i++)
			{
				ReplayEntry r = replay[nReplay++];
				for (int j = 0; j < sdim; j++)
				{
					col[j] += r.weight * base[r.offset + j];
				}
			}
		}

		// Performs a Gaussian blur along each projected axis in the hyperplane.
		void blur()
		{
			// Prepare arrays
			AutoBuffer<short> neighbor1(dim + 1);
			AutoBuffer<short> neighbor2(dim + 1);
			AutoBuffer<float> zero(sdim);
			for (int k = 0; k < sdim; k++) zero[k] = 0.f;

			float* newValue = new float[sdim * hashTable.size()];
			float* oldValue = hashTable.getValues();
			float* hashTableBase = oldValue;

			// For each of d+1 axes,
			for (int j = 0; j <= dim; j++)
			{
				//printf(" %d", j);fflush(stdout);

				// For each vertex in the lattice,
				for (int i = 0; i < hashTable.size(); i++)
				{
					// blur point i in dimension j
					short* key = hashTable.getKeys() + i * dim; // keys to current vertex
					for (int k = 0; k < dim; k++)
					{
						neighbor1[k] = key[k] + 1;
						neighbor2[k] = key[k] - 1;
					}
					neighbor1[j] = key[j] - dim;
					neighbor2[j] = key[j] + dim; // keys to the neighbors along the given axis.

					float* oldVal = oldValue + i * sdim;
					float* newVal = newValue + i * sdim;

					float* vm1, * vp1;

					vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
					if (vm1) vm1 = vm1 - hashTableBase + oldValue;
					else vm1 = zero;

					vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor	  
					if (vp1) vp1 = vp1 - hashTableBase + oldValue;
					else vp1 = zero;

					// Mix values of the three vertices
					for (int k = 0; k < sdim; k++)
					{
						newVal[k] = (0.25f * vm1[k] + 0.5f * oldVal[k] + 0.25f * vp1[k]);
					}
				}
				swap(newValue, oldValue);

				// the freshest data is now in oldValue, and newValue is ready to be written over
			}

			// depending where we ended up, we may have to copy data
			if (oldValue != hashTableBase)
			{
				memcpy(hashTableBase, oldValue, hashTable.size() * sdim * sizeof(float));
				delete oldValue;
			}
			else
			{
				delete newValue;
			}
			//printf("\n");
		}

	private:
		const int dim;// guide image dimention
		const int sdim;//src image homogeneous dimension 
		const int nData;
		float* elevated, * scaleFactor, * barycentric;
		short* canonical;
		short* key;

		// slicing is done by replaying splatting (ie storing the sparse matrix)
		struct ReplayEntry
		{
			int offset;
			float weight;
		} *replay;
		int nReplay, nReplaySub;

	public:
		char* rank;
		short* greedy;
		HashTablePermutohedral hashTable;
	};

	void highDimensionalGaussianFilterPermutohedralLattice(const Mat& src, const Mat& guide, Mat& dest, const float sigma_color, const float sigma_space)
	{
		dest.create(src.size(), src.type());

		const float invSpatialStdev = 1.0f / sigma_space;
		const float invColorStdev = 1.0f / (sigma_color / 255.f);
		const float ColorStdev = 255.f / invColorStdev;

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
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * (float)guide.at<uchar>(y, guide.channels() * x + c) / 255.f;
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
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * guide.at<float>(y, guide.channels() * x + c) / 255.f;
					}
				}
			}
		}

		// Filter the input with respect to the position vectors. 
		PermutohedralLattice::filter(src, ref, dest);
	}

	void highDimensionalGaussianFilterPermutohedralLattice(const vector<Mat>& vsrc, const vector<Mat>& vguide, Mat& dest, const float sigma_color, const float sigma_space)
	{
		Mat src; merge(vsrc, src);
		Mat ref; merge(vguide, ref);
		highDimensionalGaussianFilterPermutohedralLattice(src, ref, dest, sigma_color, sigma_space);
	}


	void highDimensionalGaussianFilterPermutohedralLattice(const Mat& src, Mat& dest, const float sigma_color, const float sigma_space)
	{
		highDimensionalGaussianFilterPermutohedralLattice(src, src, dest, sigma_color, sigma_space);
	}


	void highDimensionalGaussianFilterPermutohedralLatticeTile(const Mat& src, const Mat& guide, Mat& dest, const float sigma_color, const float sigma_space, const Size div, const float truncateBoundary)
	{
		const int channels = src.channels();
		const int guide_channels = guide.channels();

		dest.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

		const int borderType = cv::BORDER_REFLECT;
		const int vecsize = sizeof(__m256) / sizeof(float);//8

		if (div.area() == 1)
		{
			highDimensionalGaussianFilterPermutohedralLattice(src, guide, dest, sigma_color, sigma_space);
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

				highDimensionalGaussianFilterPermutohedralLattice(subImageInput[thread_num], subImageGuide[thread_num], subImageOutput[thread_num], sigma_color, sigma_space);

				cp::pasteTileAlign(subImageOutput[thread_num], dest, div, idx, r, 8, 8);
			}
		}
	}
}