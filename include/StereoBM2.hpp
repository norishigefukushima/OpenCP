#pragma once

#include "common.hpp"
#include "stereoEval.hpp"

namespace cp
{
#if CV_MAJOR_VERSION <=3
	class CP_EXPORT StereoBM2
	{
	public:
		enum {
			PREFILTER_NORMALIZED_RESPONSE = 0, PREFILTER_XSOBEL = 1,
			BASIC_PRESET = 0, FISH_EYE_PRESET = 1, NARROW_PRESET = 2
		};

		//! the default constructor
		StereoBM2();
		//! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
		StereoBM2(int preset, int ndisparities = 0, int SADWindowSize = 21);
		//! the method that reinitializes the state. The previous content is destroyed
		void init(int preset, int ndisparities = 0, int SADWindowSize = 21);

		//! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
		void operator()(cv::InputArray left, cv::InputArray right, cv::OutputArray disparity, int disptype = CV_16S);

		//! pointer to the underlying CvStereoBMState
		cv::Ptr<CvStereoBMState> state;
	};
#endif
}