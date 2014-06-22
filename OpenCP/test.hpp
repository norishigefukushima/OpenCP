#include "opencp.hpp"

void guiSLICTest(Mat& src);

void guiBilateralFilterTest(Mat& src);
void timeBirateralTest(Mat& src);

void guiRecursiveBilateralFilterTest(Mat& src);

void guiJointBirateralFilterTest(Mat& src, Mat& guide);

void guiGuidedFilterTest(Mat& src);
void timeGuidedFilterTest(Mat& src);

void guiDomainTransformFilterTest(Mat& src);

void guiBinalyWeightedRangeFilterTest(Mat& src);
void guiJointBinalyWeightedRangeFilterTest(Mat& src, Mat& guide);

void guiNonLocalMeansTest(Mat& src);


void guiIterativeBackProjectionTest(Mat& src);

//for application
void guiDetailEnhancement(Mat& src);
void guiDenoiseTest(Mat& src);


void guiViewSynthesis();

void guiJointNearestFilterTest(Mat& src);


