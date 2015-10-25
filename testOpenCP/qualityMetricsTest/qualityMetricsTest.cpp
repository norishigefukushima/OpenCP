#include <opencp.hpp>

using namespace std;
using namespace cv;

namespace cp
{

	void qualityMetricsTest()
	{
		vector<string> str(50);
		int n = 0;
		str[n++] = "conesH";
		str[n++] = "teddyH";
		str[n++] = "Aloe";
		str[n++] = "Art";
		str[n++] = "Baby1";
		str[n++] = "Baby2";
		str[n++] = "Baby3";
		str[n++] = "Books";
		str[n++] = "Bowling1";
		//str[n++]="Bowling2";
		str[n++] = "Cloth1";
		str[n++] = "Cloth2";
		str[n++] = "Cloth3";
		str[n++] = "Cloth4";

		str[n++] = "Dolls";
		str[n++] = "Flowerpots";
		str[n++] = "Lampshade1";
		str[n++] = "Lampshade2";
		str[n++] = "Laundry";
		str[n++] = "Midd1";
		str[n++] = "Midd2";
		str[n++] = "Moebius";
		str[n++] = "Monopoly";
		str[n++] = "Plastic";
		str[n++] = "Reindeer";
		str[n++] = "Rocks1";
		str[n++] = "Rocks2";

		str[n++] = "Wood1";
		str[n++] = "Wood2";


		int bb = 15;

		for (int i = 0; i < 2; i++)
		{
			cout << str[i] << endl;

			Mat src3 = imread(format("datasets/input/%s/view3.png", str[i]));
			Mat src4 = imread(format("datasets/input/%s/view4.png", str[i]));
			Mat src5 = imread(format("datasets/input/%s/view5.png", str[i]));

			Mat prop3 = imread(format("datasets/result/%s/prop3.png", str[i]));
			Mat prop4 = imread(format("datasets/result/%s/prop4.png", str[i]));
			Mat prop5 = imread(format("datasets/result/%s/prop5.png", str[i]));

			Mat conv3 = imread(format("datasets/result/%s/conv3.png", str[i]));
			Mat conv4 = imread(format("datasets/result/%s/conv4.png", str[i]));
			Mat conv5 = imread(format("datasets/result/%s/conv5.png", str[i]));


			{
				int method = IQM_PSNR;
				double c3 = calcImageQualityMetric(src3, conv3, method, bb);
				double c4 = calcImageQualityMetric(src4, conv4, method, bb);
				double c5 = calcImageQualityMetric(src5, conv5, method, bb);
				double p3 = calcImageQualityMetric(src3, prop3, method, bb);
				double p4 = calcImageQualityMetric(src4, prop4, method, bb);
				double p5 = calcImageQualityMetric(src5, prop5, method, bb);
				cout << format("CONV   PSNR, %.3f, %.3f, %.3f \n", c3, c4, c5);
				cout << format("PROP   PSNR, %.3f, %.3f, %.3f \n", p3, p4, p5);
				cout << format("delt   PSNR, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
			}
		{
			int method = IQM_MSSSIM_FAST;
			double c3 = calcImageQualityMetric(src3, conv3, method, bb);
			double c4 = calcImageQualityMetric(src4, conv4, method, bb);
			double c5 = calcImageQualityMetric(src5, conv5, method, bb);
			double p3 = calcImageQualityMetric(src3, prop3, method, bb);
			double p4 = calcImageQualityMetric(src4, prop4, method, bb);
			double p5 = calcImageQualityMetric(src5, prop5, method, bb);
			cout << format("CONV   MSSSIM, %.3f, %.3f, %.3f \n", c3, c4, c5);
			cout << format("PROP   MSSSIM, %.3f, %.3f, %.3f \n", p3, p4, p5);
			cout << format("delt   MSSSIM, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
		}
		{
			int method = IQM_CWSSIM_FAST;
			double c3 = calcImageQualityMetric(src3, conv3, method, bb);
			double c4 = calcImageQualityMetric(src4, conv4, method, bb);
			double c5 = calcImageQualityMetric(src5, conv5, method, bb);
			double p3 = calcImageQualityMetric(src3, prop3, method, bb);
			double p4 = calcImageQualityMetric(src4, prop4, method, bb);
			double p5 = calcImageQualityMetric(src5, prop5, method, bb);
			cout << format("CONV   CWSSIM, %.3f, %.3f, %.3f \n", c3, c4, c5);
			cout << format("PROP   CWSSIM, %.3f, %.3f, %.3f \n", p3, p4, p5);
			cout << format("delt   CWSSIM, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
		}

		}

		for (int i = 2; i < n; i++)
		{
			cout << str[i] << endl;

			Mat src3 = imread(format("datasets/input/%s/view2.png", str[i]));
			Mat src4 = imread(format("datasets/input/%s/view3.png", str[i]));
			Mat src5 = imread(format("datasets/input/%s/view4.png", str[i]));

			Mat prop3 = imread(format("datasets/result/%s/prop3.png", str[i]));
			Mat prop4 = imread(format("datasets/result/%s/prop4.png", str[i]));
			Mat prop5 = imread(format("datasets/result/%s/prop5.png", str[i]));

			Mat conv3 = imread(format("datasets/result/%s/conv3.png", str[i]));
			Mat conv4 = imread(format("datasets/result/%s/conv4.png", str[i]));
			Mat conv5 = imread(format("datasets/result/%s/conv5.png", str[i]));


			{
				int method = IQM_PSNR;
				double c3 = calcImageQualityMetric(src3, conv3, method, bb);
				double c4 = calcImageQualityMetric(src4, conv4, method, bb);
				double c5 = calcImageQualityMetric(src5, conv5, method, bb);
				double p3 = calcImageQualityMetric(src3, prop3, method, bb);
				double p4 = calcImageQualityMetric(src4, prop4, method, bb);
				double p5 = calcImageQualityMetric(src5, prop5, method, bb);
				cout << format("CONV   PSNR, %.3f, %.3f, %.3f \n", c3, c4, c5);
				cout << format("PROP   PSNR, %.3f, %.3f, %.3f \n", p3, p4, p5);
				cout << format("delt   PSNR, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
			}
		{
			int method = IQM_MSSSIM_FAST;
			double c3 = calcImageQualityMetric(src3, conv3, method, bb);
			double c4 = calcImageQualityMetric(src4, conv4, method, bb);
			double c5 = calcImageQualityMetric(src5, conv5, method, bb);
			double p3 = calcImageQualityMetric(src3, prop3, method, bb);
			double p4 = calcImageQualityMetric(src4, prop4, method, bb);
			double p5 = calcImageQualityMetric(src5, prop5, method, bb);
			cout << format("CONV   MSSSIM, %.3f, %.3f, %.3f \n", c3, c4, c5);
			cout << format("PROP   MSSSIM, %.3f, %.3f, %.3f \n", p3, p4, p5);
			cout << format("delt   MSSSIM, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
		}
		{
			int method = IQM_CWSSIM_FAST;
			double c3 = calcImageQualityMetric(src3, conv3, method, bb);
			double c4 = calcImageQualityMetric(src4, conv4, method, bb);
			double c5 = calcImageQualityMetric(src5, conv5, method, bb);
			double p3 = calcImageQualityMetric(src3, prop3, method, bb);
			double p4 = calcImageQualityMetric(src4, prop4, method, bb);
			double p5 = calcImageQualityMetric(src5, prop5, method, bb);
			cout << format("CONV   CWSSIM, %.3f, %.3f, %.3f \n", c3, c4, c5);
			cout << format("PROP   CWSSIM, %.3f, %.3f, %.3f \n", p3, p4, p5);
			cout << format("delt   CWSSIM, %.3f, %.3f, %.3f \n", p3 - c3, p4 - c4, p5 - c5);
		}

		}
	}
}