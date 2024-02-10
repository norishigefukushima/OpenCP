#include <intrin.h>
#include <fmath/fmath.hpp>

#define EXP_ARGUMENT_CLIP_VALUE_SP (-87.3f)
const __m256 expthresh = _mm256_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);

#define SUBNORMALCLIP 

template<int use_fmath>
inline __m256 v_exp_ps(__m256 src)
{
	return _mm256_exp_ps(src);
}

template<>
inline __m256 v_exp_ps<0>(__m256 src)
{
#ifdef SUBNORMALCLIP
	return _mm256_exp_ps(_mm256_max_ps(src, expthresh));
#else
	return _mm256_exp_ps(src);
#endif
}

template<>
inline __m256 v_exp_ps<1>(__m256 src)
{
#ifdef SUBNORMALCLIP
	return fmath::exp_ps256(_mm256_max_ps(src, expthresh));
#else
	return fmath::exp_ps256(src);
#endif
}