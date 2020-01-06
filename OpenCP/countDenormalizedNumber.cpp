#include "updateCheck.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	int countDenormalizedNumber(const Mat& src)
	{
		const float* s = src.ptr<float>(0);
		int count = 0;
		for (int i = 0; i < src.size().area(); i++)
		{
			if (fpclassify(s[i]) == FP_SUBNORMAL)
			{
				count++;
			}
		}
		return count;
	}

	double countDenormalizedNumberRatio(const Mat& src)
	{
		return countDenormalizedNumber(src) / (double)src.size().area();
	}
}