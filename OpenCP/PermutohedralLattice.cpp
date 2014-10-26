#include "opencp.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;


class Window {
public:
	Window() {
		xstride = ystride = tstride = width = height = frames = channels = 0;
		data = NULL;
	}

	Window(Window im, int mint_, int minx_, int miny_, int frames_, int width_, int height_) {
		int mint = max(0, mint_);
		int maxt = min(im.frames, mint_ + frames_);
		int minx = max(0, minx_);
		int maxx = min(im.width, minx_ + width_);
		int miny = max(0, miny_);
		int maxy = min(im.height, miny_ + height_);

		xstride = im.xstride;
		ystride = im.ystride;
		tstride = im.tstride;

		width = maxx - minx;
		height = maxy - miny;
		frames = maxt - mint;
		channels = im.channels;

		data = im.data + mint * tstride + miny * ystride + minx * xstride;
	}

	float *operator()(int t, int x, int y) {
		return data + t * tstride + x * xstride + y * ystride;
	}

	float *operator()(int x, int y) {
		return data + x * xstride + y * ystride;
	}

	float *operator()(int x) {
		return data + x * xstride;
	}


	int width, height, frames, channels;
	int xstride, ystride, tstride;
	float *data;    

};

class Image : public Window {
public:
	Image() : refCount(NULL)
	{
		width = frames = height = channels = 0;
		xstride = ystride = tstride = 0;
		data = NULL;
	}

	Image(int frames_, int width_, int height_, int channels_, const float *data_ = NULL) 
	{
		frames = frames_;
		width = width_;
		height = height_;
		channels = channels_;

		long long memory = ((long long)frames_ * 
			(long long)height_ *
			(long long)width_ * 
			(long long)channels_);

		data = new float[memory];
		if (!data_) memset(data, 0, memory * sizeof(float));
		else memcpy(data, data_, memory * sizeof(float));

		xstride = channels;
		ystride = xstride * width;
		tstride = ystride * height;
		refCount = new int;
		*refCount = 1;

		//printf("Making new image "); 
		//debug();
	}

	// does not copy data

	Image &operator=(const Image &im) {
		if (refCount) {
			refCount[0]--;
			if (*refCount <= 0) {
				delete refCount;
				delete[] data;
			}
		}

		width = im.width;
		height = im.height;
		channels = im.channels;
		frames = im.frames;

		data = im.data;

		xstride = channels;
		ystride = xstride * width;
		tstride = ystride * height;	

		refCount = im.refCount;       
		if (refCount) refCount[0]++;

		return *this;
	}

	Image(const Image &im) {
		width = im.width;
		height = im.height;
		channels = im.channels;
		frames = im.frames;

		data = im.data;       
		xstride = channels;
		ystride = xstride * width;
		tstride = ystride * height;	

		refCount = im.refCount;        
		if (refCount) refCount[0]++;       
	}

	// copies data from the window
	Image(Window im) {
		width = im.width;
		height = im.height;
		channels = im.channels;
		frames = im.frames;

		xstride = channels;
		ystride = xstride * width;
		tstride = ystride * height;	

		refCount = new int;
		*refCount = 1;
		long long memory = ((long long)width *
			(long long)height *
			(long long)channels *
			(long long)frames);
		data = new float[memory];

		for (int t = 0; t < frames; t++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					for (int c = 0; c < channels; c++) {
						(*this)(t, x, y)[c] = im(t, x, y)[c];
					}
				}
			}
		}

	}

	// makes a new copy of this image
	Image copy() {
		return Image(*((Window *)this));
	}


	~Image() 
	{    
		if (!refCount)
		{
			return; // the image was a dummy
		}

		refCount[0]--;
		if (*refCount <= 0) {
			delete refCount;
			delete[] data;
		}
	}

	int *refCount;

protected:
	Image &operator=(Window im) {
		return *this;
	}
};

/***************************************************************/
/* Hash table implementation for permutohedral lattice
* 
* The lattice points are stored sparsely using a hash table.
* The key for each point is its spatial location in the (d+1)-
* dimensional space.
*/
/***************************************************************/
class HashTablePermutohedral {
public:
	/* Constructor
	*  kd_: the dimensionality of the position vectors on the hyperplane.
	*  vd_: the dimensionality of the value vectors
	*/
	HashTablePermutohedral(int kd_, int vd_) : kd(kd_), vd(vd_) {
		capacity = 1 << 15;
		filled = 0;
		entries = new Entry[capacity];
		keys = new short[kd*capacity/2];
		values = new float[vd*capacity/2];
		memset(values, 0, sizeof(float)*vd*capacity/2);
	}

	// Returns the number of vectors stored.
	int size() { return filled; }

	// Returns a pointer to the keys array.
	short *getKeys() { return keys; }

	// Returns a pointer to the values array.
	float *getValues() { return values; }

	/* Returns the index into the hash table for a given key.
	*     key: a pointer to the position vector.
	*       h: hash of the position vector.
	*  create: a flag specifying whether an entry should be created,
	*          should an entry with the given key not found.
	*/
	int lookupOffset(short *key, size_t h, bool create = true) {

		// Double hash table size if necessary
		if (filled >= (capacity/2)-1) { grow(); }

		// Find the entry with the given key
		while (1) {
			Entry e = entries[h];
			// check if the cell is empty
			if (e.keyIdx == -1) {
				if (!create) return -1; // Return not found.
				// need to create an entry. Store the given key.
				for (int i = 0; i < kd; i++)
					keys[filled*kd+i] = key[i];

				e.keyIdx = filled*kd;
				e.valueIdx = filled*vd;
				entries[h] = e;
				filled++;
				return e.valueIdx;
			}

			// check if the cell has a matching key
			bool match = true;
			for (int i = 0; i < kd && match; i++)
				match = keys[e.keyIdx+i] == key[i];
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
	float *lookup(short *k, bool create = true) {
		size_t h = hash(k) % capacity;
		int offset = lookupOffset(k, h, create);
		if (offset < 0) return NULL;
		else return values + offset;
	};

	/* Hash function used in this implementation. A simple base conversion. */  
	size_t hash(const short *key) {
		size_t k = 0;
		for (int i = 0; i < kd; i++) {
			k += key[i];
			k *= 2531011; 
		}
		return k;
	}

private:
	/* Grows the size of the hash table */
	void grow() {
		printf("Resizing hash table\n");

		size_t oldCapacity = capacity;
		capacity *= 2;

		// Migrate the value vectors.
		float *newValues = new float[vd*capacity/2];
		memset(newValues, 0, sizeof(float)*vd*capacity/2);
		memcpy(newValues, values, sizeof(float)*vd*filled);
		delete[] values;
		values = newValues;	   

		// Migrate the key vectors.
		short *newKeys = new short[kd*capacity/2];
		memcpy(newKeys, keys, sizeof(short)*kd*filled);
		delete[] keys;
		keys = newKeys;

		Entry *newEntries = new Entry[capacity];

		// Migrate the table of indices.
		for (size_t i = 0; i < oldCapacity; i++) {		
			if (entries[i].keyIdx == -1) continue;
			size_t h = hash(keys + entries[i].keyIdx) % capacity;
			while (newEntries[h].keyIdx != -1) {
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

	short *keys;
	float *values;
	Entry *entries;
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
class PermutohedralLattice {
public:

	static void filter(Mat& im, Mat& ref, Mat& dest)
	{
		// Create lattice

		PermutohedralLattice lattice(ref.channels(), im.channels()+1, im.cols*im.rows);

		// Splat into the lattice
		printf("Splatting...\n");


		float *col = new float[im.channels()+1]; 
		col[im.channels()] = 1.f; // homogeneous coordinate

		float *imPtr = im.ptr<float>(0);
		float *refPtr = ref.ptr<float>(0);

		{
			CalcTime t("Splatting");
			for (int y = 0; y < im.rows; y++)
			{
				for (int x = 0; x < im.cols; x++)
				{
					memcpy(col,imPtr,sizeof(float)*im.channels());
					imPtr+=im.channels();

					lattice.splat(refPtr, col);
					refPtr += ref.channels();
				}
			}
		}

		// Blur the lattice
		{
			CalcTime t("Blurring");
			lattice.blur();
		}


		// Slice from the lattice
		{
			CalcTime t("Slicing");
			lattice.beginSlice();
			float* dst = dest.ptr<float>(0);
			for (int y = 0; y < im.rows; y++)
			{
				for (int x = 0; x < im.cols; x++)
				{
					lattice.slice(col);
					float scale = 1.0f/col[im.channels()];


					for (int c = 0; c < im.channels(); c++)
					{
						*dst++ = col[c]*scale;
					}
				}
			}
		}
		delete[] col;
	}

	/* Filters given image against a reference image.
	*   im : image to be bilateral-filtered.
	*  ref : reference image whose edges are to be respected.
	*/
	static Image filter(Image im, Image ref) {
		// Create lattice

		PermutohedralLattice lattice(ref.channels, im.channels+1, im.width*im.height*im.frames);

		// Splat into the lattice
		printf("Splatting...\n");


		float *col = new float[im.channels+1]; 
		col[im.channels] = 1.f; // homogeneous coordinate

		float *imPtr = im(0, 0, 0);
		float *refPtr = ref(0, 0, 0);

		{
			CalcTime t("Splatting");
			for (int t = 0; t < im.frames; t++)
			{
				for(int y = 0; y < im.height; y++)
				{
					for (int x = 0; x < im.width; x++)
					{
						memcpy(col,imPtr,sizeof(float)*im.channels);
						imPtr+=im.channels;

						lattice.splat(refPtr, col);
						refPtr += ref.channels;
					}
				}
			}
		}

		// Blur the lattice
		{
			CalcTime t("Blurring");
			lattice.blur();
		}

		Image out(im.frames, im.width, im.height, im.channels);
		// Slice from the lattice
		{
			CalcTime t("Slicing");
			lattice.beginSlice();
			float *outPtr = out(0, 0, 0);
			for (int t = 0; t < im.frames; t++)
			{
				for (int y = 0; y < im.height; y++)
				{
					for (int x = 0; x < im.width; x++)
					{
						lattice.slice(col);
						float scale = 1.0f/col[im.channels];
						for (int c = 0; c < im.channels; c++)
						{
							*outPtr++ = col[c]*scale;
						}
					}
				}
			}
		}
		delete[] col;
		return out;
	}

	/* Constructor
	*     d_ : dimensionality of key vectors
	*    vd_ : dimensionality of value vectors
	* nData_ : number of points in the input
	*/
	PermutohedralLattice(int d_, int vd_, int nData_) :
		d(d_), vd(vd_), nData(nData_), hashTable(d_, vd_) {

			// Allocate storage for various arrays
			elevated = new float[d+1];
			scaleFactor = new float[d];

			greedy = new short[d+1];
			rank = new char[d+1];	
			barycentric = new float[d+2];
			replay = new ReplayEntry[nData*(d+1)];
			nReplay = 0;
			canonical = new short[(d+1)*(d+1)];
			key = new short[d+1];

			// compute the coordinates of the canonical simplex, in which
			// the difference between a contained point and the zero
			// remainder vertex is always in ascending order. (See pg.4 of paper.)
			for (int i = 0; i <= d; i++) {
				for (int j = 0; j <= d-i; j++)
					canonical[i*(d+1)+j] = i; 
				for (int j = d-i+1; j <= d; j++)
					canonical[i*(d+1)+j] = i - (d+1); 
			}

			// Compute parts of the rotation matrix E. (See pg.4-5 of paper.)      
			for (int i = 0; i < d; i++) {
				// the diagonal entries for normalization
				scaleFactor[i] = 1.0f/(sqrtf((float)(i+1)*(i+2)));

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
				scaleFactor[i] *= (d+1)*sqrtf(2.0f/3.f);
			}
	}


	/* Performs splatting with given position and value vectors */
	void splat(float *position, float *value)
	{
		// first rotate position into the (d+1)-dimensional hyperplane
		elevated[d] = -d*position[d-1]*scaleFactor[d-1];

		for (int i = d-1; i > 0; i--)
			elevated[i] = (elevated[i+1] - 
			i*position[i-1]*scaleFactor[i-1] + 
			(i+2)*position[i]*scaleFactor[i]);
		elevated[0] = elevated[1] + 2*position[0]*scaleFactor[0];

		// prepare to find the closest lattice points
		float scale = 1.0f/(d+1);	
		char* myrank = rank;
		short* mygreedy = greedy;

		// greedily search for the closest zero-colored lattice point
		int sum = 0;

		for (int i = 0; i <= d; i++)
		{
			float v = elevated[i]*scale;
			float up = ceilf(v)*(d+1);
			float down = floorf(v)*(d+1);

			if (up - elevated[i] < elevated[i] - down) mygreedy[i] = (short)up;
			else mygreedy[i] = (short)down;

			sum += mygreedy[i];
		}
		sum /= d+1;

		// rank differential to find the permutation between this simplex and the canonical one.
		// (See pg. 3-4 in paper.)
		memset(myrank, 0, sizeof(char)*(d+1));
		for (int i = 0; i < d; i++)
		{
			for (int j = i+1; j <= d; j++)
			{
				if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j]) myrank[i]++; else myrank[j]++;
			}
		}

		if (sum > 0)
		{ 
			// sum too large - the point is off the hyperplane.
			// need to bring down the ones with the smallest differential
			for (int i = 0; i <= d; i++)
			{
				if (myrank[i] >= d + 1 - sum)
				{
					mygreedy[i] -= d+1;
					myrank[i] += sum - (d+1);
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
			for (int i = 0; i <= d; i++) 
			{
				if (myrank[i] < -sum) 
				{
					mygreedy[i] += d+1;
					myrank[i] += (d+1) + sum;
				}
				else
				{
					myrank[i] += sum;
				}
			}
		}

		// Compute barycentric coordinates (See pg.10 of paper.)
		memset(barycentric, 0, sizeof(float)*(d+2));
		for (int i = 0; i <= d; i++)
		{
			barycentric[d-myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
			barycentric[d+1-myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
		}
		barycentric[0] += 1.0f + barycentric[d+1];

		// Splat the value into each vertex of the simplex, with barycentric weights.
		for (int remainder = 0; remainder <= d; remainder++)
		{
			// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
			for (int i = 0; i < d; i++)
				key[i] = mygreedy[i] + canonical[remainder*(d+1) + myrank[i]];

			// Retrieve pointer to the value at this vertex.
			float * val = hashTable.lookup(key, true);

			// Accumulate values with barycentric weight.
			for (int i = 0; i < vd; i++)
				val[i] += barycentric[remainder]*value[i];

			// Record this interaction to use later when slicing
			replay[nReplay].offset = val - hashTable.getValues();
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
	void slice(float *col) {
		float *base = hashTable.getValues();
		for (int j = 0; j < vd; j++) col[j] = 0;
		for (int i = 0; i <= d; i++)
		{
			ReplayEntry r = replay[nReplay++];
			for (int j = 0; j < vd; j++)
			{
				col[j] += r.weight*base[r.offset + j];
			}
		}
	}

	/* Performs a Gaussian blur along each projected axis in the hyperplane. */
	void blur()
	{
		// Prepare arrays
		short *neighbor1 = new short[d+1];
		short *neighbor2 = new short[d+1];
		float *newValue = new float[vd*hashTable.size()];
		float *oldValue = hashTable.getValues();
		float *hashTableBase = oldValue;

		float *zero = new float[vd];
		for (int k = 0; k < vd; k++) zero[k] = 0;

		// For each of d+1 axes,
		for (int j = 0; j <= d; j++)
		{
			//printf(" %d", j);fflush(stdout);

			// For each vertex in the lattice,
			for (int i = 0; i < hashTable.size(); i++)
			{ // blur point i in dimension j
				short *key    = hashTable.getKeys() + i*(d); // keys to current vertex
				for (int k = 0; k < d; k++)
				{
					neighbor1[k] = key[k] + 1;
					neighbor2[k] = key[k] - 1;
				}
				neighbor1[j] = key[j] - d;
				neighbor2[j] = key[j] + d; // keys to the neighbors along the given axis.

				float *oldVal = oldValue + i*vd;		
				float *newVal = newValue + i*vd;

				float *vm1, *vp1;

				vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
				if (vm1) vm1 = vm1 - hashTableBase + oldValue;
				else vm1 = zero;

				vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor	  
				if (vp1) vp1 = vp1 - hashTableBase + oldValue;
				else vp1 = zero;

				// Mix values of the three vertices
				for (int k = 0; k < vd; k++)
					newVal[k] = (0.25f*vm1[k] + 0.5f*oldVal[k] + 0.25f*vp1[k]);
			}  
			float *tmp = newValue;
			newValue = oldValue;
			oldValue = tmp;
			// the freshest data is now in oldValue, and newValue is ready to be written over
		}

		// depending where we ended up, we may have to copy data
		if (oldValue != hashTableBase) {
			memcpy(hashTableBase, oldValue, hashTable.size()*vd*sizeof(float));
			delete oldValue;
		} else {
			delete newValue;
		}
		printf("\n");

		delete zero;
		delete neighbor1; 
		delete neighbor2;
	}

private:

	const int d, vd, nData;
	float *elevated, *scaleFactor, *barycentric;
	short *canonical;    
	short *key;

	// slicing is done by replaying splatting (ie storing the sparse matrix)
	struct ReplayEntry {
		int offset;
		float weight;
	} *replay;
	int nReplay, nReplaySub;

public:
	char  *rank;
	short *greedy;
	HashTablePermutohedral hashTable;
};


void bilateralFilterPermutohedralLattice(Mat& src, Mat& dest, float sigma_space, float sigma_color)
{
	if(dest.empty()) dest.create(src.size(),src.type());

	float invSpatialStdev = 1.0f/sigma_space;
	float invColorStdev = 1.0f/(sigma_color/255.f);
	float ColorStdev = 255.f/invColorStdev;
	// Construct the position vectors out of x, y, r, g, and b.

	Image input(1, src.cols, src.rows, 3);
	Image positions(1, src.cols, src.rows, 5);


	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			positions(x, y)[0] = invSpatialStdev * x;
			positions(x, y)[1] = invSpatialStdev * y;
			input(x, y)[0] = positions(x, y)[2] = invColorStdev * (float)src.at<uchar>(y,3*x+0)/255.f;
			input(x, y)[1] = positions(x, y)[3] = invColorStdev * (float)src.at<uchar>(y,3*x+1)/255.f;
			input(x, y)[2] = positions(x, y)[4] = invColorStdev * (float)src.at<uchar>(y,3*x+2)/255.f;
		}
	}

	// Filter the input with respect to the position vectors. (see permutohedral.h)
	Image out = PermutohedralLattice::filter(input, positions);

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dest.at<uchar>(y,3*x+0)= saturate_cast<uchar>(ColorStdev*out(x, y)[0]+0.5f);
			dest.at<uchar>(y,3*x+1)= saturate_cast<uchar>(ColorStdev*out(x, y)[1]+0.5f);
			dest.at<uchar>(y,3*x+2)= saturate_cast<uchar>(ColorStdev*out(x, y)[2]+0.5f);
		}
	}

}
