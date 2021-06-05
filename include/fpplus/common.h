#pragma once
#ifndef FPPLUS_COMMON_H
#define FPPLUS_COMMON_H

#if defined(__OPENCL_VERSION__)
	#if defined(cl_khr_fp64)
		/*
		 * Since OpenCL 1.2 cl_khr_fp64 is an optional core feature and doesn't need a pragma to enable.
		 * In fact, using the pragma results in warning on some OpenCL implementations.
		 */
		#if __OPENCL_VERSION__ < 120
			#pragma OPENCL EXTENSION cl_khr_fp64 : enable
		#endif
	#elif defined(cl_amd_fp64)
		#pragma OPENCL EXTENSION cl_amd_fp64 : enable
	#elif defined(cl_APPLE_fp64_basic_ops)
		#pragma OPENCL EXTENSION cl_APPLE_fp64_basic_ops : enable
	#else
		#error "The code must be compiled for a device with double-precision support"
	#endif
	#ifndef FP_FAST_FMA
		#error "The code must be compiler for a device with performant fma operation"
	#endif
#elif defined(__CUDA_ARCH__)
	/* nvcc targeting a CUDA device */
	#if __CUDA_ARCH__ < 200
		#error "The code must be compiled for a CUDA device with fused multiply-add (compute capability 2.0+)"
	#endif
#elif !defined(__FP_FAST_FMA) && !defined(__FMA__) && !defined(__AVX2__) && !defined(__KNC__)
	//#error "The code must be compiled for a processor with fused multiply-add (FMA)"
#endif

#if defined(__OPENCL_VERSION__) && defined(__FAST_RELAXED_MATH__)
	#error "The code must be compiled without -cl-fast-relaxed-math options: the implemented algorithms depend on precise floating-point behaviour"
#elif defined(__FAST_MATH__) && !defined(__CUDA_ARCH__)
	/* On CUDA the code uses intrinsics which guarantee floating-point behaviour regardless of optimization mode */
	//#error "The code must be compiled without -ffast-math option: the implemented algorithms depend on precise floating-point behaviour"
#endif

#ifndef FPPLUS_USE_FPADDRE
	#ifdef FPPLUS_EMULATE_FPADDRE
		#define FPPLUS_USE_FPADDRE 1
	#else
		#define FPPLUS_USE_FPADDRE 0
	#endif
#endif /* !defined(FPPLUS_USE_FPADDRE) */

#if defined(__GNUC__) && defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__VSX__)
#include <altivec.h>
#elif defined(__ARM_ARCH_8A__)
#include <arm_neon.h>
#else
#include <intrin.h>
#endif

#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	#define FPPLUS_C99_SYNTAX 1
#else
	#define FPPLUS_C99_SYNTAX 0
#endif

#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__OPENCL_VERSION__)
	#define FPPLUS_RESTRICT restrict
#elif defined(__GNUC__)
	#define FPPLUS_RESTRICT __restrict__
#elif defined(_MSC_VER)
	#define FPPLUS_RESTRICT __restrict
#else
	#define FPPLUS_RESTRICT
#endif

#if defined(_MSC_VER)
	#define FPPLUS_STATIC_INLINE static __forceinline
//#define FPPLUS_STATIC_INLINE inline
#elif defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__cplusplus) || defined(__OPENCL_VERSION__)
	#define FPPLUS_STATIC_INLINE static inline
#else
	#define FPPLUS_STATIC_INLINE static
#endif

#if defined(__GNUC__)
	#define FPPLUS_NONNULL_POINTER_ARGUMENTS __attribute__((__nonnull__))
#else
	#define FPPLUS_NONNULL_POINTER_ARGUMENTS
#endif

#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__OPENCL_VERSION__)
	#define FPPLUS_NONNULL_POINTER(name) name[restrict static 1]
#else
	#define FPPLUS_NONNULL_POINTER(name) *FPPLUS_RESTRICT name
#endif

#if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__OPENCL_VERSION__)
	#define FPPLUS_ARRAY_POINTER(name, min_size) name[restrict static min_size]
#else
	#define FPPLUS_ARRAY_POINTER(name, min_size) *FPPLUS_RESTRICT name
#endif

#ifdef FPPLUS_EMULATE_FPADDRE
	#include <fpplus/fpaddre.h>
#endif

#endif /* FPPLUS_COMMON_H */
