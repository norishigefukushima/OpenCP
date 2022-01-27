#pragma once

#include "common.hpp"
#include "separableFilterCore.hpp"

namespace cp
{
	/// <summary>
	/// non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void nonLocalMeansFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void nonLocalMeansFilter(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// joint non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointNonLocalMeansFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// joint non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointNonLocalMeansFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void patchBilateralFilter(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (square)</param>
	/// <param name="kernelWindowSize">kernel size (square)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void patchBilateralFilter(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// joint patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointPatchBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// joint patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (square)</param>
	/// <param name="kernelWindowSize">kernel size (square)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointPatchBilateralFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void nonLocalMeansFilterSeparable(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void nonLocalMeansFilterSeparable(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// joint non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointNonLocalMeansFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// joint non-local means filter (NLM).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointNonLocalMeansFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);


	/// <summary>
	/// separable patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void patchBilateralFilterSeparable(cv::InputArray src, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// separable patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (square)</param>
	/// <param name="kernelWindowSize">kernel size (square)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void patchBilateralFilterSeparable(cv::InputArray src, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// separable joint patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (rectangle)</param>
	/// <param name="kernelWindowSize">kernel size (rectangle)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointPatchBilateralFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const cv::Size patchWindowSize, const cv::Size kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);

	/// <summary>
	/// separable joint patch-based bilateral filter (non-local means+spatial weight).
	/// weight[i] = (-pow(abs(i/sigma), powexp) / powexp) for range and space.
	/// powexp=2: Gaussian, powexp=1: Laplacian, powexp=infinity(input 0): Box.
	/// </summary>
	/// <param name="src">input</param>
	/// <param name="guide">guide</param>
	/// <param name="dest">output</param>
	/// <param name="patchWindowSize">patch size (square)</param>
	/// <param name="kernelWindowSize">kernel size (square)</param>
	/// <param name="sigma_range">sigma for range</param>
	/// <param name="powexp_range">order of pow for range</param>
	/// <param name="patchnorm">patch distance metrics L1(1) or L2(2)</param>
	/// <param name="sigma_space">sigma for range</param>
	/// <param name="powexp_space">order of pow for range</param>
	/// <param name="SEPARABLE_METHOD">type of separable filtering</param>
	/// <param name="alpha">parameter of range sigma for second pass</param>
	/// <param name="borderType">borderType</param>
	CP_EXPORT void jointPatchBilateralFilterSeparable(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, const int patchWindowSize, const int kernelWindowSize, const double sigma, const double powexp = 2.0, const int patchNorm = 2, const double sigma_space = -1.0, const double powexp_space = 2.0, SEPARABLE_METHOD method = SEPARABLE_METHOD::SWITCH_VH, const double alpha = 0.8, const int borderType = cv::BORDER_DEFAULT);


	//not tested
	CP_EXPORT void weightedJointNonLocalMeansFilter(cv::Mat& src, cv::Mat& weightMap, cv::Mat& guide, cv::Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma);
}