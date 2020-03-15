# Box Filter

```cpp
void boxFilter_64f(cv::Mat& src, cv::Mat& dest, const int r, const int boxType, const int parallelType);

void boxFilter_32f(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType);

void boxFilter_8u(cv::Mat& src, cv::Mat& dest, int r, int boxType, int parallelType);
```
I will merge there 3 functions into one boxFilter.
these functions are for double/float/uchar image.

* src image
* dest image
* kernel radius
* box filter type (details are presented)
* parallel type (3 type, NAIVE: sirial, OMP: paralleized by openmp, PARALLEL_FOR_ parallelized by OpenCV framework)

* O(r^2) convolution
* O(r) separable convolution
* O(1) integral image
* O(1) separable summed area table (SSAT)
* O(1) one pass SAT

enum BoxTypes
{
	BOX_OPENCV,//using opencv function

	BOX_NAIVE,// naive convolution O(r^2) implementation
	BOX_NAIVE_SSE,// SSE convolution O(r^2) implementation
	BOX_NAIVE_AVX,//  AVXconvolution O(r^2) implementation

	BOX_SEPARABLE_HV, //naive separable implementation, horizontal then vertical O(r) implementation
	BOX_SEPARABLE_HV_SSE, //SSE separable implementation, horizontal then vertical O(r) implementation
	BOX_SEPARABLE_HV_AVX, //AVX separable implementation, horizontal then vertical O(r) implementation
	BOX_SEPARABLE_VH_AVX, //AVX separable implementation, vertical then horizontal O(r) implementation (no naive, SSE)
	BOX_SEPARABLE_VHI, //naive separable implementation, intereleve vertical filtering andhorizontal filtering O(r) implementation
	BOX_SEPARABLE_VHI_SSE, //naive separable implementation, intereleve vertical filtering andhorizontal filtering O(r) implementation
	BOX_SEPARABLE_VHI_AVX, //naive separable implementation, intereleve vertical filtering andhorizontal filtering O(r) implementation

	BOX_INTEGRAL, //naive integral image based O(1) implementation
	BOX_INTEGRAL_SSE, //SSE integral image based O(1) implementation
	BOX_INTEGRAL_AVX, //AVX integral image based O(1) implementation
	BOX_INTEGRAL_ONEPASS, //naive integral image based O(1) implementation (we cannot vectorize)
	BOX_INTEGRAL_ONEPASS_AREA,// ??

	BOX_SSAT_HV, //naive O(1) separable horizontal then vertical
	BOX_SSAT_HV_SSE, //SSE O(1) separable horizontal then vertical
	BOX_SSAT_HV_AVX, //AVX O(1) separable horizontal then vertical
	BOX_SSAT_HV_BLOCKING,
	BOX_SSAT_HV_BLOCKING_SSE,
	BOX_SSAT_HV_BLOCKING_AVX,
	BOX_SSAT_HV_4x4,
	BOX_SSAT_HV_8x8,
	BOX_SSAT_HV_ROWSUM_GATHER_SSE,
	BOX_SSAT_HV_ROWSUM_GATHER_AVX,
	BOX_SSAT_HtH,
	BOX_SSAT_HtH_SSE,
	BOX_SSAT_HtH_AVX,

	BOX_SSAT_VH, //naive O(1) separable vertical then horizontal filtering
	BOX_SSAT_VH_SSE, //SSE O(1) separable vertical then horizontal filtering
	BOX_SSAT_VH_AVX, //AVX
	BOX_SSAT_VH_ROWSUM_GATHER_SSE,
	BOX_SSAT_VH_ROWSUM_GATHER_AVX,
	BOX_SSAT_VtV,
	BOX_SSAT_VtV_SSE,
	BOX_SSAT_VtV_AVX,

	BOX_OPSAT, // one pas SAT
	BOX_OPSAT_2Div,
	BOX_OPSAT_nDiv,

	NumBoxTypes	// num of boxtypes. must be last element
};
