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
			keys = (short*)_mm_malloc(sizeof(short) * kd * capacity / 2, SSE_ALIGN);
			values = (float*)_mm_malloc(sizeof(float) * vd * capacity / 2, SSE_ALIGN);
			memset(values, 0, sizeof(float) * vd * capacity / 2);
		}

		~HashTablePermutohedral()
		{
			delete[] entries;
			_mm_free(keys);
			_mm_free(values);
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
			if (offset < 0) return nullptr;
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
			float* newValues = (float*)_mm_malloc(sizeof(float) * vd * capacity / 2, SSE_ALIGN);
			memset(newValues, 0, sizeof(float) * vd * capacity / 2);
			memcpy(newValues, values, sizeof(float) * vd * filled);
			_mm_free(values);
			values = newValues;

			// Migrate the key vectors.
			short* newKeys = (short*)_mm_malloc(sizeof(short) * kd * capacity / 2, SSE_ALIGN);
			memcpy(newKeys, keys, sizeof(short) * kd * filled);
			_mm_free(keys);
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
		struct Entry
		{
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

			const bool useOpt = true;
			const bool isOptColorBF = (src.channels() == 3 && ref.channels() == 5) && useOpt;
			if (isOptColorBF)
			{
				//cout << "color bilateral filtering" << endl;
				{
					//Timer t("Splatting");
					const float* srcPtr = src.ptr<float>();
					const float* refPtr = ref.ptr<float>();
					for (int y = 0; y < src.rows; y++)
					{
						for (int x = 0; x < src.cols; x++)
						{
							memcpy(col, srcPtr, 12);//sizeof(float)*3
							lattice.splatColor(refPtr, col);
							srcPtr += src_channels;
							refPtr += ref_channels;
						}
					}
				}

				// Blur the lattice
				{
					//Timer t("Blurring");
					lattice.blurColor();
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
							lattice.sliceColor(col);
							_mm_storeu_ps(dst, _mm_mul_ps(_mm_load_ps(col), _mm_set1_ps(1.0f / col[3])));
							dst += 3;
						}
					}
				}
			}
			else
			{
				{
					const int copySize = sizeof(float) * src_channels;
					//Timer t("Splatting");
					const float* srcPtr = src.ptr<float>();
					const float* refPtr = ref.ptr<float>();
					for (int y = 0; y < src.rows; y++)
					{
						for (int x = 0; x < src.cols; x++)
						{
							memcpy(col, srcPtr, copySize);
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
		}

		/* Constructor
		*     d_ : dimensionality of key vectors
		*    vd_ : dimensionality of value vectors
		* nData_ : number of points in the input
		*/
		PermutohedralLattice(int guideChannels, int srcHomogeneousChannels, int nData_) :
			gdim(guideChannels), shdim(srcHomogeneousChannels), nData(nData_), hashTable(guideChannels, srcHomogeneousChannels)
		{
			CV_Assert(gdim < 127);
			CV_Assert(shdim < 127);
			// Allocate storage for various arrays
			elevated = (float*)_mm_malloc(sizeof(float) * (gdim + 1), AVX_ALIGN);
			scaleFactor = (float*)_mm_malloc(sizeof(float) * (gdim), AVX_ALIGN);
			greedy = (short*)_mm_malloc(sizeof(short) * (gdim + 1), AVX_ALIGN);
			rank = (char*)_mm_malloc(sizeof(char) * (gdim + 1), AVX_ALIGN);
			barycentric = (float*)_mm_malloc(sizeof(float) * (gdim + 2), AVX_ALIGN);
			canonical = (short*)_mm_malloc(sizeof(short) * (gdim + 1) * (gdim + 1), AVX_ALIGN);
			key = (short*)_mm_malloc(sizeof(short) * (gdim + 1), AVX_ALIGN);

			replay = new ReplayEntry[nData * (gdim + 1)];
			nReplay = 0;
			// compute the coordinates of the canonical simplex, in which
			// the difference between a contained point and the zero
			// remainder vertex is always in ascending order. (See pg.4 of paper.)
			for (int i = 0; i <= gdim; i++)
			{
				for (int j = 0; j <= gdim - i; j++)
				{
					canonical[i * (gdim + 1) + j] = i;
				}
				for (int j = gdim - i + 1; j <= gdim; j++)
				{
					canonical[i * (gdim + 1) + j] = i - (gdim + 1);
				}
			}

			// Compute parts of the rotation matrix E. (See pg.4-5 of paper.)      
			for (int i = 0; i < gdim; i++)
			{
				// the diagonal entries for normalization
				scaleFactor[i] = 1.f / (sqrtf((float)(i + 1) * (i + 2)));

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
				scaleFactor[i] *= (gdim + 1) * sqrtf(2.f / 3.f);
			}
		}

		~PermutohedralLattice()
		{
			_mm_free(elevated);
			_mm_free(scaleFactor);
			_mm_free(greedy);
			_mm_free(rank);
			_mm_free(barycentric);
			_mm_free(canonical);
			_mm_free(key);
			delete[] replay;
		}

		/* Performs splatting with given position and value vectors */
		void splat(const float* position, const float* value)
		{
			const int dim1 = gdim + 1;//dim + 1
			// first rotate position into the (d+1)-dimensional hyperplane
			elevated[gdim] = -gdim * position[gdim - 1] * scaleFactor[gdim - 1];

			for (int i = gdim - 1; i > 0; i--)
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

			for (int i = 0; i <= gdim; i++)
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
			for (int i = 0; i < gdim; i++)
			{
				for (int j = i + 1; j <= gdim; j++)
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
				for (int i = 0; i <= gdim; i++)
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
				for (int i = 0; i <= gdim; i++)
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
			memset(barycentric, 0, sizeof(float) * (gdim + 2));
			for (int i = 0; i <= gdim; i++)
			{
				barycentric[gdim - myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
				barycentric[gdim + 1 - myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
			}
			barycentric[0] += 1.f + barycentric[gdim + 1];

			// Splat the value into each vertex of the simplex, with barycentric weights.
			for (int remainder = 0; remainder <= gdim; remainder++)
			{
				// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
				for (int i = 0; i < gdim; i++)
				{
					key[i] = mygreedy[i] + canonical[remainder * (dim1)+myrank[i]];
				}

				// Retrieve pointer to the value at this vertex.
				float* val = hashTable.lookup(key, true);

				// Accumulate values with barycentric weight.
				for (int i = 0; i < shdim; i++)
				{
					val[i] += barycentric[remainder] * value[i];
				}

				// Record this interaction to use later when slicing
				replay[nReplay].offset = (int)(val - hashTable.getValues());
				replay[nReplay].weight = barycentric[remainder];
				nReplay++;
			}
		}

		void splatColor(const float* position, const float* value)
		{
			// first rotate position into the (d+1)-dimensional hyperplane
			elevated[5] = -5.f * position[4] * scaleFactor[4];
			elevated[4] = elevated[5] - 4.f * position[3] * scaleFactor[3] + 6.f * position[4] * scaleFactor[4];
			elevated[3] = elevated[4] - 3.f * position[2] * scaleFactor[2] + 5.f * position[3] * scaleFactor[3];
			elevated[2] = elevated[3] - 2.f * position[1] * scaleFactor[1] + 4.f * position[2] * scaleFactor[2];
			elevated[1] = elevated[2] - 1.f * position[0] * scaleFactor[0] + 3.f * position[1] * scaleFactor[1];
			elevated[0] = elevated[1] + 2.f * position[0] * scaleFactor[0];

			// prepare to find the closest lattice points
			const float scale = 1.f / 6.f;
			char* myrank = rank;
			short* mygreedy = greedy;

			// greedily search for the closest zero-colored lattice point
			int sum;
			const __m256 melevated = _mm256_load_ps(elevated);
			const __m256 mscale = _mm256_set1_ps(scale);
			{
				const __m256 m6 = _mm256_set1_ps(6.f);
				__m256 mv = _mm256_mul_ps(melevated, mscale);
				__m256 mup = _mm256_mul_ps(_mm256_ceil_ps(mv), m6);
				__m256 mdn = _mm256_mul_ps(_mm256_floor_ps(mv), m6);
				mv = _mm256_blendv_ps(mup, mdn, _mm256_cmp_ps(_mm256_sub_ps(mup, melevated), _mm256_sub_ps(melevated, mdn), _CMP_GT_OQ));
				__m256i r = _mm256_packs_epi32(_mm256_cvtps_epi32(mv), _mm256_setzero_si256());
				_mm_storel_epi64((__m128i*)mygreedy, _mm256_castsi256_si128(r));
				short* rr = (short*)&r;
				mygreedy[4] = rr[8];
				mygreedy[5] = rr[9];
				sum = rr[0] + rr[1] + rr[2] + rr[3] + rr[8] + rr[9];
			}
			sum = int(sum *scale);

			// rank differential to find the permutation between this simplex and the canonical one.
			// (See pg. 3-4 in paper.)
			memset(myrank, 0, 6);
			__m256 mgreedysub = _mm256_sub_ps(melevated, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_load_si128((__m128i*)mygreedy))));
			float* sv = (float*)&mgreedysub;
			for (int i = 0; i < 5; i++)
			{
				const float sub = sv[i];
				for (int j = i + 1; j <= 5; j++)
				{
					if (sub < sv[j])
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
				const int sub = 6 - sum;
				for (int i = 0; i <= 5; i++)
				{
					if (myrank[i] >= sub)
					{
						mygreedy[i] -= 6;
						myrank[i] -= sub;
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
				for (int i = 0; i <= 5; i++)
				{
					if (myrank[i] < -sum)
					{
						mygreedy[i] += 6;
						myrank[i] += 6 + sum;
					}
					else
					{
						myrank[i] += sum;
					}
				}
			}

			// Compute barycentric coordinates (See pg.10 of paper.)
			memset(barycentric, 0, sizeof(float) * (7));
			mgreedysub = _mm256_sub_ps(melevated, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_load_si128((__m128i*)mygreedy))));
			mgreedysub = _mm256_mul_ps(mgreedysub, mscale);
			float* v = (float*)&mgreedysub;
			barycentric[5 - myrank[0]] += v[0];
			barycentric[6 - myrank[0]] -= v[0];
			barycentric[5 - myrank[1]] += v[1];
			barycentric[6 - myrank[1]] -= v[1];
			barycentric[5 - myrank[2]] += v[2];
			barycentric[6 - myrank[2]] -= v[2];
			barycentric[5 - myrank[3]] += v[3];
			barycentric[6 - myrank[3]] -= v[3];
			barycentric[5 - myrank[4]] += v[4];
			barycentric[6 - myrank[4]] -= v[4];
			barycentric[5 - myrank[5]] += v[5];
			barycentric[6 - myrank[5]] -= v[5];

			barycentric[0] += 1.f + barycentric[6];

			// Splat the value into each vertex of the simplex, with barycentric weights.
			for (int remainder = 0; remainder <= 5; remainder++)
			{
				// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
				const short* cptr = canonical + remainder * 6;
				key[0] = mygreedy[0] + cptr[myrank[0]];
				key[1] = mygreedy[1] + cptr[myrank[1]];
				key[2] = mygreedy[2] + cptr[myrank[2]];
				key[3] = mygreedy[3] + cptr[myrank[3]];
				key[4] = mygreedy[4] + cptr[myrank[4]];

				// Retrieve pointer to the value at this vertex.
				float* val = hashTable.lookup(key, true);

				// Accumulate values with barycentric weight.
				_mm_storeu_ps(val, _mm_fmadd_ps(_mm_set1_ps(barycentric[remainder]), _mm_loadu_ps(value), _mm_loadu_ps(val)));

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
				for (int i = 0; i < shdim; i++)
				{
					val[i] += barycentric[remainder] * value[i];
				}

				// Record this interaction to use later when slicing
				replay[nReplay].offset = (int)(val - hashTable.getValues());
				replay[nReplay].weight = barycentric[remainder];
				nReplay++;
			}
		}

		// Performs a Gaussian blur along each projected axis in the hyperplane.
		void blur()
		{
			// Prepare arrays
			AutoBuffer<short> neighbor1(gdim + 1);
			AutoBuffer<short> neighbor2(gdim + 1);
			AutoBuffer<float> zero(shdim);
			for (int k = 0; k < shdim; k++) zero[k] = 0.f;

			float* newValue = (float*)_mm_malloc(sizeof(float) * shdim * hashTable.size(), SSE_ALIGN);
			float* oldValue = hashTable.getValues();
			float* hashTableBase = oldValue;

			// For each of d+1 axes,
			for (int j = 0; j <= gdim; j++)
			{
				//printf(" %d", j);fflush(stdout);

				// For each vertex in the lattice,
				for (int i = 0; i < hashTable.size(); i++)
				{
					// blur point i in dimension j
					short* key = hashTable.getKeys() + i * gdim; // keys to current vertex
					for (int k = 0; k < gdim; k++)
					{
						neighbor1[k] = key[k] + 1;
						neighbor2[k] = key[k] - 1;
					}
					neighbor1[j] = key[j] - gdim;
					neighbor2[j] = key[j] + gdim; // keys to the neighbors along the given axis.

					float* oldVal = oldValue + i * shdim;
					float* newVal = newValue + i * shdim;

					float* vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
					vm1 = (vm1) ? vm1 - hashTableBase + oldValue : zero;

					float* vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor	  
					vp1 = (vp1) ? vp1 - hashTableBase + oldValue : zero;

					// Mix values of the three vertices
					for (int k = 0; k < shdim; k++)
					{
						newVal[k] = (0.25f * vm1[k] + 0.5f * oldVal[k] + 0.25f * vp1[k]);
					}
				}
				// the freshest data is now in oldValue, and newValue is ready to be written over
				swap(newValue, oldValue);
			}

			// depending where we ended up, we may have to copy data
			if (oldValue != hashTableBase)
			{
				memcpy(hashTableBase, oldValue, hashTable.size() * shdim * sizeof(float));
				_mm_free(oldValue);
			}
			else
			{
				_mm_free(newValue);
			}
			//printf("\n");
		}

		void blurColor()
		{
			// Prepare arrays
			__m128i _neighbor1 = _mm_setzero_si128();
			__m128i _neighbor2 = _mm_setzero_si128();
			__m128 _zero = _mm_setzero_ps();
			short* neighbor1 = (short*)&_neighbor1;//size is 6 but alloc 8 for SIMD
			short* neighbor2 = (short*)&_neighbor2;//size is 6 but alloc 8 for SIMD	
			float* zero = (float*)&_zero;

			const int hashTableSize = hashTable.size();
			float* newValue = (float*)_mm_malloc(sizeof(float) * 4 * hashTableSize, SSE_ALIGN);
			float* oldValue = hashTable.getValues();
			float* hashTableBase = oldValue;

			// For each of d+1 axes,
			const __m128 m025 = _mm_set1_ps(0.25f);
			const __m128 m05 = _mm_set1_ps(0.5f);
			const __m128i m1s = _mm_set1_epi16(1);
			for (int j = 0; j <= 5; j++)
			{
				// For each vertex in the lattice,
				for (int i = 0; i < hashTableSize; i++)
				{
					// blur point i in dimension j
					short* key = hashTable.getKeys() + i * 5; // keys to current vertex
					_mm_store_si128((__m128i*)neighbor1, _mm_add_epi16(_mm_loadu_si128((__m128i*)key), m1s));
					_mm_store_si128((__m128i*)neighbor2, _mm_sub_epi16(_mm_loadu_si128((__m128i*)key), m1s));
					neighbor1[j] = key[j] - 5;
					neighbor2[j] = key[j] + 5; // keys to the neighbors along the given axis.

					const float* oldVal = oldValue + (i << 2);
					float* newVal = newValue + (i << 2);

					float* vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
					vm1 = (vm1) ? vm1 - hashTableBase + oldValue : zero;

					float* vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor	  
					vp1 = (vp1) ? vp1 - hashTableBase + oldValue : zero;

					// Mix values of the three vertices
					_mm_store_ps(newVal, _mm_fmadd_ps(m025, _mm_load_ps(vp1), _mm_fmadd_ps(m05, _mm_load_ps(oldVal), _mm_mul_ps(m025, _mm_load_ps(vm1)))));
				}
				// the freshest data is now in oldValue, and newValue is ready to be written over
				swap(newValue, oldValue);
			}

			// depending where we ended up, we may have to copy data
			if (oldValue != hashTableBase)
			{
				memcpy(hashTableBase, oldValue, hashTableSize * 4 * sizeof(float));
				_mm_free(oldValue);
			}
			else
			{
				_mm_free(newValue);
			}
			//printf("\n");
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
			const float* base = hashTable.getValues();
			ReplayEntry* r = &replay[nReplay];
			//int i = 0
			{
				for (int j = 0; j < shdim; j++)
				{
					col[j] = r[0].weight * base[r[0].offset + j];
				}
			}
			for (int i = 1; i <= gdim; i++)
			{
				for (int j = 0; j < shdim; j++)
				{
					col[j] += r[i].weight * base[r[i].offset + j];
				}
			}
			nReplay += gdim + 1;
		}

		void sliceColor(float* col)
		{
			const float* base = hashTable.getValues();
			const ReplayEntry* r = &replay[nReplay];
			__m256 v = _mm256_mul_ps(_mm256_set_m128(_mm_set1_ps(r[1].weight), _mm_set1_ps(r[0].weight)), _mm256_loadu2_m128(base + r[1].offset, base + r[0].offset));
			v = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(r[3].weight), _mm_set1_ps(r[2].weight)), _mm256_loadu2_m128(base + r[3].offset, base + r[2].offset), v);
			v = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(r[5].weight), _mm_set1_ps(r[4].weight)), _mm256_loadu2_m128(base + r[5].offset, base + r[4].offset), v);
			_mm_store_ps(col, _mm256_castps256_ps128(_mm256_add_ps(v, _mm256_permute2f128_ps(v, v, 0x01))));
			/*_mm_store_ps(col, _mm_mul_ps(_mm_set1_ps(r[0].weight), _mm_loadu_ps(base + r[0].offset)));//0
			_mm_store_ps(col, _mm_fmadd_ps(_mm_set1_ps(r[1].weight), _mm_loadu_ps(base + r[1].offset), _mm_load_ps(col)));//1
			_mm_store_ps(col, _mm_fmadd_ps(_mm_set1_ps(r[2].weight), _mm_loadu_ps(base + r[2].offset), _mm_load_ps(col)));//2
			_mm_store_ps(col, _mm_fmadd_ps(_mm_set1_ps(r[3].weight), _mm_loadu_ps(base + r[3].offset), _mm_load_ps(col)));//3
			_mm_store_ps(col, _mm_fmadd_ps(_mm_set1_ps(r[4].weight), _mm_loadu_ps(base + r[4].offset), _mm_load_ps(col)));//4
			_mm_store_ps(col, _mm_fmadd_ps(_mm_set1_ps(r[5].weight), _mm_loadu_ps(base + r[5].offset), _mm_load_ps(col)));//5*/
			nReplay += 6;
		}

	private:
		const int gdim;// guide image dimention
		const int shdim;//src image homogeneous dimension 
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
		//Timer t;
		const float inv = 1.f / 255.f;
		const int gch = guide.channels();
		if (src.depth() == CV_8U)
		{
			for (int y = 0; y < src.rows; y++)
			{
				const uchar* gptr = guide.ptr<uchar>(y);
				for (int x = 0; x < src.cols; x++)
				{
					ref.ptr<float>(y, x)[0] = invSpatialStdev * x;
					ref.ptr<float>(y, x)[1] = invSpatialStdev * y;
					for (int c = 0; c < gch; c++)
					{
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * (float)gptr[gch * x + c] * inv;
					}
				}
			}
		}
		else if (src.depth() == CV_32F)
		{
			for (int y = 0; y < src.rows; y++)
			{
				const float* gptr = guide.ptr<float>(y);
				for (int x = 0; x < src.cols; x++)
				{
					ref.ptr<float>(y, x)[0] = invSpatialStdev * x;
					ref.ptr<float>(y, x)[1] = invSpatialStdev * y;
					for (int c = 0; c < gch; c++)
					{
						ref.ptr<float>(y, x)[2 + c] = invColorStdev * gptr[gch * x + c] * inv;
					}
				}
			}
		}
		// Filter the input with respect to the position vectors. 
		if (src.depth() == CV_8U)
		{
			Mat src32f; src.convertTo(src32f, CV_32F);
			Mat dst32f(src.size(), src32f.type());
			PermutohedralLattice::filter(src32f, ref, dst32f);
			dst32f.convertTo(dest, CV_8U);
		}
		else
		{
			PermutohedralLattice::filter(src, ref, dest);
		}
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