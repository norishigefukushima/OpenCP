#include "opencp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void detailEnhancementBilateral(Mat& src, Mat& dest, int d, float sigma_color, float sigma_space, float boost, int color)
	{
		if (color == PROCESS_LAB)
		{
			Mat lab;
			cvtColor(src, lab, COLOR_BGR2Lab);
			vector<Mat> v;
			split(lab, v);

			Mat epf;

			//bilateralFilter(v[0],epf,Size(d,d),sigma_color,sigma_space);
			recursiveBilateralFilter(v[0], epf, sigma_color, sigma_space);

			subtract(v[0], epf, epf);
			addWeighted(v[0], 1.0, epf, boost, 0, v[0]);
			merge(v, lab);
			cvtColor(lab, dest, COLOR_Lab2BGR);
		}
		else if (color == PROCESS_BGR)
		{
			Mat epf;

			bilateralFilter(src, epf, Size(d, d), sigma_color, sigma_space);

			subtract(src, epf, epf);
			addWeighted(src, 1.0, epf, boost, 0, dest);
		}
	}
}
//for various filter