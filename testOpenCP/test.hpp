#pragma once

#include "opencp.hpp"

#ifdef _DEBUG
#pragma comment(lib, "opencpd.lib")
#pragma comment(lib, "MultiScaleFilterd.lib")
#pragma comment(lib, "SpatialFilterd.lib")
#else
#pragma comment(lib, "opencp.lib")
#pragma comment(lib, "MultiScaleFilter.lib")
#pragma comment(lib, "SpatialFilter.lib")
#endif

//core
void testKMeans(cv::Mat& src);
void testAlphaBlend(cv::Mat& src1, cv::Mat& src2);
void testAlphaBlendMask(cv::Mat& src1, cv::Mat& src2);
void consoleImageTest();
void testAddNoise(cv::Mat& src);
void testIsSame();
void testConcat();
void copyMakeBorderTest(cv::Mat& src);
void testCorrelationCoefficient();
void testCropZoom();
void testHistogram();
void testHistogram2(cv::Mat& src);
void testPlot();
void testPlot2D();
void testPSNR(cv::Mat& ref);
void testRGBHistogram();
void testRGBHistogram2();
void testSplitMerge(cv::Mat& src);
void testMatInfo();
void testStat();
void testStreamConvert8U();
void testTimer(cv::Mat& src);
void testDestinationTimePrediction(cv::Mat& src);
void testTiling(cv::Mat& src);
void testLocalPSNR(cv::Mat& ref);
void testVideoSubtitle();
void testWebPAnimation();
void testWindowFunction();

//imgproc
void guiPixelizationTest();
void guiSLICTest(cv::Mat& src);
void guiDisparityPlaneFitSLICTest(cv::Mat& leftim, cv::Mat& rightim, cv::Mat& GT);
void fitPlaneTest();
void guiCvtColorPCATest();
void guiColorCorrectionTest(cv::Mat& src, cv::Mat& ref);

//filter
void testGuidedImageFilter(cv::Mat& img_p, cv::Mat& img_I);
void guiHazeRemoveTest();

void guiEdgePresevingFilterOpenCV(cv::Mat& src);
void guiWeightMapTest();
void guiCrossBasedLocalFilter(cv::Mat& src);
void guiBilateralFilterTest(cv::Mat& src);
void guiSeparableBilateralFilterTest(cv::Mat& src);

void testWeightedHistogramFilter(cv::Mat& src, cv::Mat& guide);
void testWeightedHistogramFilterDisparity();


void highDimentionalGaussianFilterTest(cv::Mat& src);
void highDimentionalGaussianFilterHSITest();

void guiGausianFilterTest(cv::Mat& src_);
void guiRecursiveBilateralFilterTest(cv::Mat& src);
void guiRealtimeO1BilateralFilterTest(cv::Mat& src);
void getPSNRRealtimeO1BilateralFilterKodak();
void guiJointRealtimeO1BilateralFilterTest(cv::Mat& src_, cv::Mat& guide_);

void timeBirateralTest(cv::Mat& src);
void guiDualBilateralFilterTest(cv::Mat& src1, cv::Mat& src2);
void guiJointBirateralFilterTest(cv::Mat& src, cv::Mat& guide);
void guiDomainTransformFilterTest(cv::Mat& src);
void guiJointDomainTransformFilterTest(cv::Mat& src, cv::Mat& guide);
void guiCodingDistortionRemoveTest(cv::Mat& src);
void guiBinalyWeightedRangeFilterTest(cv::Mat& src);
void guiJointBinalyWeightedRangeFilterTest(cv::Mat& src, cv::Mat& guide);
void guiDomainTransformFilter(cv::Mat& src);
void guiNonLocalMeansTest(cv::Mat& src);
void guiSeparableNLMTest(cv::Mat& src);
void guiIterativeBackProjectionTest(cv::Mat& src);

//stereo
void testCVStereoBM();
void testCVStereoSGBM();
void testStereoBase();

//for application
void guiDetailEnhancement(cv::Mat& src);
void guiDenoiseTest(cv::Mat& src);
void guiViewSynthesis();
void guiJointNearestFilterTest(cv::Mat& src);
void fftTest(cv::Mat& src);
void guiHazeRemoveTest(cv::Mat& haze);
void qualityMetricsTest();
void guiCoherenceEnhancingShockFilter(cv::Mat& src, cv::Mat& dest);
void guiUpsampleTest(cv::Mat& src_);

void guiAnalysisCompare(cv::Mat& src1, cv::Mat& src2);
void imshowAnalysisCompare(cv::String winname, cv::Mat& src1, cv::Mat& src2);
void imshowAnalysis(cv::String winname, std::vector<cv::Mat>& s);
void imshowAnalysis(cv::String winname, cv::Mat& src);

void testMultiScaleFilter();
void testUnnormalizedBilateralFilter();

//SpatialFilter
void testSpatialFilter(cv::Mat& src);