#include "pch.h"
#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

using namespace cv;
using namespace std;
namespace cp
{
//https://sourceforge.net/projects/fourier-ipal/
	static void convertBGRtoLabel_Center(const cv::Mat& src, const int K, cv::Mat& labels, cv::Mat& centers)
	{
		labels.create(src.rows * src.cols, 1, CV_32SC1);
		centers.create(K, 1, CV_32FC3);

		int k = 0;
		centers.ptr<cv::Vec3f>(0)[0][0] = (float)src.ptr<cv::Vec3b>(0)[0][0];
		centers.ptr<cv::Vec3f>(0)[0][1] = (float)src.ptr<cv::Vec3b>(0)[0][1];
		centers.ptr<cv::Vec3f>(0)[0][2] = (float)src.ptr<cv::Vec3b>(0)[0][2];
		k++;

		for (int y = 0; y < src.rows; y++)
		{
			const cv::Vec3b* bgr = src.ptr<cv::Vec3b>(y);
			for (int x = 0; x < src.cols; x++)
			{
				const uchar b = bgr[x][0];
				const uchar g = bgr[x][1];
				const uchar r = bgr[x][2];

				for (int i = 0; i <= k; i++)
				{
					if (i == k)
					{
						centers.ptr<cv::Vec3f>(k)[0][0] = (float)b;
						centers.ptr<cv::Vec3f>(k)[0][1] = (float)g;
						centers.ptr<cv::Vec3f>(k)[0][2] = (float)r;
						labels.ptr<int>(y * src.cols + x)[0] = i;
						k++;
						break;
					}

					if (centers.ptr<cv::Vec3f>(i)[0][0] == (float)b &&
						centers.ptr<cv::Vec3f>(i)[0][1] == (float)g &&
						centers.ptr<cv::Vec3f>(i)[0][2] == (float)r)
					{
						labels.ptr<int>(y * src.cols + x)[0] = i;
						break;
					}
				}
			}
		}
	}

	
	void quantization(const cv::Mat& input_image, int K, cv::Mat& centers, cv::Mat& labels, const int method)
	{
		cv::imwrite("Fourier/input.ppm", input_image);

		std::string cmd = "Fourier\\Fourier.exe Fourier\\input.ppm " + std::to_string(K) + " > tmp.txt";
		//cout << cmd << endl;
		
		FILE* fp = _popen(cmd.c_str(), "w");
		if (NULL == fp)
		{
			printf("file open error ! \n");
			return;
		}
		_pclose(fp);
		string name;
		if (method == 0) name = "out_wan.ppm";
		if (method == 1) name = "out_wu.ppm";
		if (method == 2) name = "out_neural.ppm";
		Mat input_image8u = cv::imread(name, 1);
		if (input_image8u.empty())cout << "file open error: " << name << endl;

		convertBGRtoLabel_Center(input_image8u, K, labels, centers);
	}

	void nQunat(const cv::Mat& input_image, const int K, cv::Mat& centers, cv::Mat& labels, const ClusterMethod cm)
	{
		cv::imwrite("nQuantCpp/input.png", input_image);
		std::string cmd = "nQuantCpp\\nQuantCpp.exe nQuantCpp\\input.png ";

		switch (cm)
		{
		case ClusterMethod::quantize_DIV:
		case ClusterMethod::kmeans_DIV:
			cmd = cmd + "/a DIV /m ";
			break;
		case ClusterMethod::quantize_PNN:
		case ClusterMethod::kmeans_PNN:
			cmd = cmd + "/a PNN /m ";
			break;
		case ClusterMethod::quantize_EAS:
		case ClusterMethod::kmeans_EAS:
			cmd = cmd + "/a EAS /m ";
			break;
		case ClusterMethod::quantize_SPA:
		case ClusterMethod::kmeans_SPA:
			cmd = cmd + "/a SPA /m ";
			break;
		default:
			break;
		}

		cmd = cmd + std::to_string(K) + " > tmp.txt";
		//cmd = cmd + std::to_string(K);
		//cout << cmd << endl;
		FILE* fp = _popen(cmd.c_str(), "w");

		if (NULL == fp)
		{
			printf("file open error ! \n");
			return;
		}
		_pclose(fp);

		string name;
		switch (cm)
		{
		case ClusterMethod::quantize_DIV:
		case ClusterMethod::kmeans_DIV:
			name = "../../../../input-DIVquant" + std::to_string(K) + ".png";
			break;
		case ClusterMethod::quantize_PNN:
		case ClusterMethod::kmeans_PNN:
			name = "../../../../input-PNNquant" + std::to_string(K) + ".png";
			break;
		case ClusterMethod::quantize_EAS:
		case ClusterMethod::kmeans_EAS:
			name = "../../../../input-EASquant" + std::to_string(K) + ".png";
			break;
		case ClusterMethod::quantize_SPA:
		case ClusterMethod::kmeans_SPA:
			name = "../../../../input-SPAquant" + std::to_string(K) + ".png";
			break;
		default:
			break;
		}
		
		Mat input_image8u = cv::imread(name, 1);
		convertBGRtoLabel_Center(input_image8u, K, labels, centers);
	}
}