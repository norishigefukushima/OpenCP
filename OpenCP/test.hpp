#include "opencp.hpp"

void guiSLICTest(Mat& src);

void guiBilateralFilterTest(Mat& src);
void guiSeparableBilateralFilterTest(Mat& src);

void timeBirateralTest(Mat& src);

void guiDualBilateralFilterTest(Mat& src1, Mat& src2);

void guiGausianFilterTest(Mat& src_); 

void guiRecursiveBilateralFilterTest(Mat& src);
void guiRealtimeO1BilateralFilterTest(Mat& src);

void guiJointBirateralFilterTest(Mat& src, Mat& guide);

void guiGuidedFilterTest(Mat& src);
void timeGuidedFilterTest(Mat& src);


void guiDomainTransformFilterTest(Mat& src);
void guiJointDomainTransformFilterTest(Mat& src, Mat& guide);

void guiCodingDistortionRemoveTest(Mat& src);
void guiBinalyWeightedRangeFilterTest(Mat& src);
void guiJointBinalyWeightedRangeFilterTest(Mat& src, Mat& guide);

void guiNonLocalMeansTest(Mat& src);
void guiSeparableNLMTest(Mat& src);

void guiIterativeBackProjectionTest(Mat& src);

//for application
void guiDetailEnhancement(Mat& src);
void guiDenoiseTest(Mat& src);


void guiViewSynthesis();

void guiJointNearestFilterTest(Mat& src);

void fftTest(Mat& src);


