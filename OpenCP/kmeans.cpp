#include "kmeans.hpp"
#include "stereo_core.hpp" //for RGB histogram
#include "pointcloud.hpp"

#include "inlineCVFunctions.hpp"
#include "inlineSIMDFunctions.hpp"

#include "debugcp.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	//clusteringAoS or SoA
	// SoA
	// 
	//static int CV_KMEANS_PARALLEL_GRANULARITY = (int)utils::getConfigurationParameterSizeT("OPENCV_KMEANS_PARALLEL_GRANULARITY", 1000);
	static int	CV_KMEANS_PARALLEL_GRANULARITY = 1000;
	enum KMeansDistanceLoop
	{
		KND,
		NKD
	};

	double KMeans::clustering(InputArray dataInput, int K, InputOutputArray bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, Schedule schedule)
	{
		double ret = 0.0;
		int channels = min(dataInput.size().width, dataInput.size().height);
		Mat a = dataInput.getMat();
		switch (schedule)
		{
		case cp::KMeans::Schedule::AoS_NKD:
			ret = clusteringAoS(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoA_KND:
			ret = clusteringSoA(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;
		case cp::KMeans::Schedule::AoS_KND:
			ret = clusteringAoS(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;
		case cp::KMeans::Schedule::SoA_NKD:
			ret = clusteringSoA(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoAoS_NKD:
			ret = clusteringSoAoS(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD); break;
		case cp::KMeans::Schedule::SoAoS_KND:
			ret = clusteringSoAoS(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND); break;

		case cp::KMeans::Schedule::Auto:
		default:
		{
			if (channels < 7)
			{
				ret = clusteringSoA(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::KND);
			}
			else
			{
				ret = clusteringAoS(dataInput, K, bestLabels, criteria, attempts, flags, _centers, function, KMeansDistanceLoop::NKD);
			}
		}
		break;
		}

		return ret;
	}

	void KMeans::setWeightMap(const cv::Mat& weight)
	{
		isUseWeight = true;
		this->weight = weight;
	}

	static void onMouseKmeans3D(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	static void orthographicProjectPoints(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest, const bool isRotationThenTranspose)
	{
		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];
		int size2 = xyz.size().area();
		int i;
		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float r[3][3];
		if (isRotationThenTranspose)
		{
			const float f00 = (float)K.at<double>(0, 0);
			const float xc = (float)K.at<double>(0, 2);
			const float f11 = (float)K.at<double>(1, 1);
			const float yc = (float)K.at<double>(1, 2);

			r[0][0] = (float)R.at<double>(0, 0);
			r[0][1] = (float)R.at<double>(0, 1);
			r[0][2] = (float)R.at<double>(0, 2);

			r[1][0] = (float)R.at<double>(1, 0);
			r[1][1] = (float)R.at<double>(1, 1);
			r[1][2] = (float)R.at<double>(1, 2);

			r[2][0] = (float)R.at<double>(2, 0);
			r[2][1] = (float)R.at<double>(2, 1);
			r[2][2] = (float)R.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0];
				const float y = data[1];
				//const float z = data[2];
				const float z = 0.f;

				const float px = r[0][0] * x + r[0][1] * y + r[0][2] * z + tt[0];
				const float py = r[1][0] * x + r[1][1] * y + r[1][2] * z + tt[1];
				const float pz = r[2][0] * x + r[2][1] * y + r[2][2] * z + tt[2];

				const float div = 1.f / pz;

				dst->x = (f00 * px + xc * pz) * div;
				dst->y = (f11 * py + yc * pz) * div;

				data += 3;
				dst++;
			}
		}
		else
		{
			Mat kr = K * R;

			r[0][0] = (float)kr.at<double>(0, 0);
			r[0][1] = (float)kr.at<double>(0, 1);
			r[0][2] = (float)kr.at<double>(0, 2);

			r[1][0] = (float)kr.at<double>(1, 0);
			r[1][1] = (float)kr.at<double>(1, 1);
			r[1][2] = (float)kr.at<double>(1, 2);

			r[2][0] = (float)kr.at<double>(2, 0);
			r[2][1] = (float)kr.at<double>(2, 1);
			r[2][2] = (float)kr.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0] + tt[0];
				const float y = data[1] + tt[1];
				const float z = data[2] + tt[2];

				const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

				dst->x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
				dst->y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

				data += 3;
				dst++;
			}
		}
	}

	static bool loop(string wname, Size size, Mat& data, Mat& label, Mat& centroid, const int K, bool isWait = true, Mat& additionalData = Mat())
	{
		cv::namedWindow(wname);
		static int f = 1000;  createTrackbar("f", wname, &f, 2000);
		static int z = 1250; createTrackbar("z", wname, &z, 2000);
		static int pitch = 90; createTrackbar("pitch", wname, &pitch, 360);
		static int roll = 0; createTrackbar("roll", wname, &roll, 180);
		static int yaw = 90; createTrackbar("yaw", wname, &yaw, 360);
		static int color = 9; createTrackbar("color", wname, &color, 20);
		static int rad = 1; createTrackbar("plot size", wname, &rad, 5);
		static Point ptMouse = Point(cvRound((size.width - 1) * 0.75), cvRound((size.height - 1) * 0.25));

		cv::setMouseCallback(wname, (MouseCallback)onMouseKmeans3D, (void*)&ptMouse);

		//project RGB2XYZ
		Mat normalizedLabel = label.clone();
		normalizedLabel *= 255 / K;
		normalizedLabel.convertTo(normalizedLabel, CV_8U);
		Mat rgb, xyz;
		data.convertTo(rgb, CV_32F);
		//cout << rgb.cols << endl;
		//cout << rgb.rows << endl;
		//if(rgb.rows==3) 
		if (rgb.cols > rgb.rows) transpose(rgb, rgb);
		if (rgb.channels() == 3) rgb = rgb.reshape(3, rgb.cols * rgb.rows);
		else rgb = rgb.reshape(3, rgb.cols * rgb.rows / 3);
		xyz = rgb.clone();
		const bool isAdditional = !additionalData.empty();
		Mat additionalRGB;
		Mat additionalXYZ;
		if (isAdditional)
		{
			additionalData.convertTo(additionalRGB, CV_32F);
			if (additionalRGB.cols > additionalRGB.rows) transpose(additionalRGB, additionalRGB);
			if (additionalRGB.channels() == 3) additionalRGB = additionalRGB.reshape(3, additionalRGB.cols * additionalRGB.rows);
			else additionalRGB = additionalRGB.reshape(3, additionalRGB.cols * additionalRGB.rows / 3);
			additionalXYZ = additionalRGB.clone();
		}

		Mat centroidRGB = centroid.clone();

		centroidRGB = centroidRGB.reshape(3);
		Mat centerxyz = centroidRGB.clone();
		vector<Point2f> pt(rgb.size().area());

		//set up etc plots
		Mat guide, guideDest;
		/*guide.push_back(Point3f(0.f, 0.f, 0.f)); //rgbzero;
		guide.push_back(Point3f(0.f, 0.f, 255.f)); //rmax
		guide.push_back(Point3f(0.f, 255.f, 0.f)); //gmax
		guide.push_back(Point3f(255.f, 0.f, 0.f)); //rmax
		guide.push_back(Point3f(255.f, 255.f, 255.f)); //rgbmax
		guide.push_back(Point3f(0.f, 255.f, 255.f)); //grmax
		guide.push_back(Point3f(255.f, 0.f, 255.f)); //brmax
		guide.push_back(Point3f(255.f, 255.f, 0.f)); //bgmax*/

		guide.push_back(Point3f(-127.5f, -127.5f, -127.5f)); //rgbzero;
		guide.push_back(Point3f(-127.5, -127.5, 127.5)); //rmax
		guide.push_back(Point3f(-127.5, 127.5, -127.5)); //gmax
		guide.push_back(Point3f(127.5, -127.5, -127.5)); //rmax
		guide.push_back(Point3f(127.5, 127.5, 127.5)); //rgbmax
		guide.push_back(Point3f(-127.5, 127.5, 127.5)); //grmax
		guide.push_back(Point3f(127.5, -127.5, 127.5)); //brmax
		guide.push_back(Point3f(127.5, 127.5, -127.5)); //bgmax

		vector<Point2f> guidept(guide.rows);
		vector<Point2f> centerpt(centroid.rows);
		vector<Point2f> additionalpt;
		if (isAdditional)additionalpt.resize(additionalXYZ.rows);

		//vector<Point2f> additionalpt(additionalPoints.rows);
		//vector<Point2f> additional_start_line(additionalStartLines.rows);
		//vector<Point2f> additional_end_line(additionalEndLines.rows);

		int key = 0;
		Mat show = Mat::zeros(size, CV_8UC3);
		Mat k = Mat::eye(3, 3, CV_64F);
		Mat R = Mat::eye(3, 3, CV_64F);
		Mat t = Mat::zeros(3, 1, CV_64F);
		Mat colorIndex;

		//rotate & plot additionalPoint
/*if (!additionalPoints.empty())
{
	cp::moveXYZ(additionalPoints, additionalPointsDest, rot, Mat::zeros(3, 1, CV_64F), true);
	if (sw_projection == 0) projectPointsParallel(additionalPointsDest, R, t, k, additionalpt, true);
	if (sw_projection == 1) projectPoints(additionalPointsDest, R, t, k, additionalpt, true);
}*/

/*if (!additionalStartLines.empty())
{
	cp::moveXYZ(additionalStartLines, additionalStartLinesDest, rot, Mat::zeros(3, 1, CV_64F), true);
	cp::moveXYZ(additionalEndLines, additionalEndLinesDest, rot, Mat::zeros(3, 1, CV_64F), true);
	if (sw_projection == 0) projectPointsParallel(additionalStartLinesDest, R, t, k, additional_start_line, true);
	if (sw_projection == 1) projectPoints(additionalStartLinesDest, R, t, k, additional_start_line, true);
	if (sw_projection == 0) projectPointsParallel(additionalEndLinesDest, R, t, k, additional_end_line, true);
	if (sw_projection == 1) projectPoints(additionalEndLinesDest, R, t, k, additional_end_line, true);
}*/
		Mat muindex(K, 1, CV_8U);
		Mat mucolor;
		for (int i = 0; i < K; i++) muindex.at<uchar>(i) = saturate_cast<uchar>(255.0 / K * i);

		while (key != 'q')
		{
			//cout <<"xyz  :"<< rgb.cols << "," << rgb.rows << "," << rgb.channels() << endl;
			//cout <<"index: "<< src.cols << "," << src.rows << "," << src.channels() << endl;
			//cv::applyColorMap(normalizedLabel, colorIndex, color);
			//cv::applyColorMap(muindex, mucolor, color);

			{
				colorIndex.create(label.size(), CV_8UC3);
				mucolor.create(K, 1, CV_8UC3);
				const int* lbl = label.ptr<int>();
				uchar* dest = colorIndex.ptr<uchar>();
				uchar* dest2 = mucolor.ptr<uchar>();
				//print_matinfo(centroidRGB);
				//print_matinfo(label);
				for (int i = 0; i < label.size().area(); i++)
				{
					const int k = lbl[i];
					dest[3 * i + 0] = saturate_cast<uchar>(centroidRGB.ptr<float>(k)[0]);
					dest[3 * i + 1] = saturate_cast<uchar>(centroidRGB.ptr<float>(k)[1]);
					dest[3 * i + 2] = saturate_cast<uchar>(centroidRGB.ptr<float>(k)[2]);
					//cout << centroidRGB.ptr<float>(k)[0] << "," << centroidRGB.ptr<float>(k)[1] << "," << centroidRGB.ptr<float>(k)[2] << endl;
				}
				for (int i = 0; i < K; i++)
				{
					dest2[3 * i + 0] = saturate_cast<uchar>(centroidRGB.ptr<float>(i)[0] * 0.8f);
					dest2[3 * i + 1] = saturate_cast<uchar>(centroidRGB.ptr<float>(i)[1] * 0.8f);
					dest2[3 * i + 2] = saturate_cast<uchar>(centroidRGB.ptr<float>(i)[2] * 0.8f);
				}
			}

			pitch = (int)(180 * (double)ptMouse.y / (double)(size.height - 1) + 0.5);
			yaw = (int)(180 * (double)ptMouse.x / (double)(size.width - 1) + 0.5);
			setTrackbarPos("pitch", wname, pitch);
			setTrackbarPos("yaw", wname, yaw);

			//intrinsic
			k.at<double>(0, 2) = (size.width - 1) * 0.5;
			k.at<double>(1, 2) = (size.height - 1) * 0.5;
			k.at<double>(0, 0) = show.cols * 0.001 * f;
			k.at<double>(1, 1) = show.cols * 0.001 * f;
			t.at<double>(2) = z - 800;

			//rotate & plot RGB plots
			Mat rot;
			cp::Eular2Rotation(pitch - 90.0, roll - 90, yaw - 90, rot);
			Mat tt = Mat::ones(3, 1, CV_64F);
			tt.at<double>(0) = -127.5;
			tt.at<double>(1) = -127.5;
			tt.at<double>(2) = -127.5;
			cp::moveXYZ(rgb, xyz, rot, tt, false);
			orthographicProjectPoints(xyz, R, t, k, pt, true);

			if (isAdditional)
			{
				cp::moveXYZ(additionalRGB, additionalXYZ, rot, tt, false);
				orthographicProjectPoints(additionalXYZ, R, t, k, additionalpt, true);
			}
			//rotate & plot guide information
			cp::moveXYZ(guide, guideDest, rot, Mat::zeros(3, 1, CV_64F), true);
			orthographicProjectPoints(guideDest, R, t, k, guidept, true);

			/*cout << center.cols << endl;
			cout << center.rows << endl;
			cout << center.channels() << endl;*/
			cp::moveXYZ(centroidRGB, centerxyz, rot, tt, false);
			orthographicProjectPoints(centerxyz, R, t, k, centerpt, true);

			//draw lines for etc points
			Point rgbzero = Point(cvRound(guidept[0].x), cvRound(guidept[0].y));
			Point rmax = Point(cvRound(guidept[1].x), cvRound(guidept[1].y));
			Point gmax = Point(cvRound(guidept[2].x), cvRound(guidept[2].y));
			Point bmax = Point(cvRound(guidept[3].x), cvRound(guidept[3].y));
			Point bgrmax = Point(cvRound(guidept[4].x), cvRound(guidept[4].y));
			Point grmax = Point(cvRound(guidept[5].x), cvRound(guidept[5].y));
			Point brmax = Point(cvRound(guidept[6].x), cvRound(guidept[6].y));
			Point bgmax = Point(cvRound(guidept[7].x), cvRound(guidept[7].y));
			Point ezero = Point(cvRound(guidept[8].x), cvRound(guidept[8].y));
			line(show, bgmax, bgrmax, COLOR_WHITE);
			line(show, brmax, bgrmax, COLOR_WHITE);
			line(show, grmax, bgrmax, COLOR_WHITE);
			line(show, bgmax, bmax, COLOR_WHITE);
			line(show, brmax, bmax, COLOR_WHITE);
			line(show, brmax, rmax, COLOR_WHITE);
			line(show, grmax, rmax, COLOR_WHITE);
			line(show, grmax, gmax, COLOR_WHITE);
			line(show, bgmax, gmax, COLOR_WHITE);
			circle(show, rgbzero, 3, COLOR_WHITE, cv::FILLED);
			arrowedLine(show, rgbzero, rmax, COLOR_RED, 2);
			arrowedLine(show, rgbzero, gmax, COLOR_GREEN, 2);
			arrowedLine(show, rgbzero, bmax, COLOR_BLUE, 2);
			//arrowedLine(show, rgbzero, bgrmax, Scalar::all(50), 1);

			//rendering addtional plots (background for main plot)
			Vec3b addtionalPlotColor = Vec3b(128, 128, 128);
			if (isAdditional)
			{
				for (int i = 0; i < additionalpt.size(); i++)
				{
					const int x = cvRound(additionalpt[i].x);
					const int y = cvRound(additionalpt[i].y);
					if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
					{
						show.at<Vec3b>(Point(x, y)) = addtionalPlotColor;

						show.at<Vec3b>(Point(x + 1, y + 0)) = addtionalPlotColor;
						show.at<Vec3b>(Point(x - 1, y + 0)) = addtionalPlotColor;
						show.at<Vec3b>(Point(x + 0, y + 1)) = addtionalPlotColor;
						show.at<Vec3b>(Point(x + 0, y - 1)) = addtionalPlotColor;
					}
				}
			}

			//rendering RGB plots
			for (int i = 0; i < normalizedLabel.size().area(); i++)
			{
				const int x = cvRound(pt[i].x);
				const int y = cvRound(pt[i].y);
				if (x >= rad && x < show.cols - rad && y >= rad && y < show.rows - rad)
				{
					for (int b = -rad; b <= rad; b++)
					{
						for (int a = -rad; a <= rad; a++)
						{
							show.at<Vec3b>(Point(x + a, y + b)) = colorIndex.at<Vec3b>(i);
						}
					}
				}
			}

			for (int i = 0; i < centerxyz.size().area(); i++)
			{
				const int x = cvRound(centerpt[i].x);
				const int y = cvRound(centerpt[i].y);
				if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
				{
					circle(show, Point(x, y), 5, mucolor.at<Vec3b>(i), 3);
				}
			}

			imshow(wname, show);
			key = waitKey(1);
			if (key == 's')
			{
				cout << "write rgb_histogram.png" << endl;
				imwrite("rgb_histogram.png", show);
			}
			if (key == 't')
			{
				ptMouse.x = cvRound((size.width - 1) * 0.5);
				ptMouse.y = cvRound((size.height - 1) * 0.5);
			}
			if (key == 'r')
			{
				ptMouse.x = cvRound((size.width - 1) * 0.75);
				ptMouse.y = cvRound((size.height - 1) * 0.25);
			}
			if (key == '?')
			{
				cout << "v: switching rendering method" << endl;
				cout << "r: reset viewing direction for parallel view" << endl;
				cout << "t: reset viewing direction for paspective view" << endl;
				cout << "s: save 3D RGB plot" << endl;
			}
			if (key == 'w')
			{
				return true;
			}
			show.setTo(0);

			if (!isWait)break;
		}
		return false;
	}

	void KMeans::gui(InputArray data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, Schedule schedule, InputArray additionalData)
	{
		bool isWait = false;
		bool isPerIteration = false;
		cv::Size windowSize = cv::Size(512 * 2, 512 * 2);
		string wname = "kmeans";

		if (isPerIteration)
		{
			clustering(data, K, _bestLabels, cv::TermCriteria(criteria.type, 1, criteria.epsilon), attempts, flags, _centers, function, schedule);
			loop(wname, windowSize, data.getMat(), _bestLabels.getMat(), _centers.getMat(), K, isWait, additionalData.getMat());
			for (int i = 0; i < criteria.maxCount; i++)
			{
				cout << i << endl;
				clustering(data, K, _bestLabels, cv::TermCriteria(criteria.type, 1, criteria.epsilon), attempts, cv::KMEANS_USE_INITIAL_LABELS, _centers, function, schedule);
				bool isexit = loop(wname, windowSize, data.getMat(), _bestLabels.getMat(), _centers.getMat(), K, isWait, additionalData.getMat());
				if (isexit) return;
			}
		}
		else
		{
			clustering(data, K, _bestLabels, criteria, attempts, flags, _centers, function, schedule);
			loop(wname, windowSize, data.getMat(), _bestLabels.getMat(), _centers.getMat(), K, isWait, additionalData.getMat());
		}
	}

	double kmeans(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers)
	{
		KMeans km;
		return km.clustering(_data, K, _bestLabels, criteria, attempts, flags, _centers);
	}

#pragma region SoA
	inline float normL2Sqr(float a, float b)
	{
		float temp = a - b;
		return temp * temp;
	}

	inline float normL2SqrAdd(float a, float b, float c)
	{
		float temp = a - b;
		return temp * temp + c;
	}

	//(a-b)^2
	inline __m256 normL2Sqr(__m256 a, __m256 b)
	{
		__m256 temp = _mm256_sub_ps(a, b);
		return _mm256_mul_ps(temp, temp);
	}

	//(a-b)^2 + c=fma(a-b, a-b, c);
	inline __m256 normL2SqrAdd(__m256 a, __m256 b, __m256 c)
	{
		__m256 temp = _mm256_sub_ps(a, b);
		return _mm256_fmadd_ps(temp, temp, c);
	}

#pragma region initialCentroid
	void KMeans::generateKmeansRandomBoxInitialCentroidSoA(const cv::Mat& data_points, cv::Mat& dest_centroids, const int K, cv::RNG& rng)
	{
		const int N = data_points.cols;
		const int dims = data_points.rows;
		cv::AutoBuffer<Vec2f, 64> box(dims);//min-max value for each dimension

		{
			int i = 0;
			for (int j = 0; j < dims; j++)
			{
				const float* sample = data_points.ptr<float>(j);
				box[j] = Vec2f(sample[i], sample[i]);
			}
		}
		for (int d = 0; d < dims; d++)
		{
			for (int i = 1; i < N; i++)
			{
				const float* sample = data_points.ptr<float>(d);
				float v = sample[i];
				box[d][0] = std::min(box[d][0], v);
				box[d][1] = std::max(box[d][1], v);
			}
		}

		for (int k = 0; k < K; k++)
		{
			for (int d = 0; d < dims; d++)
			{
				dest_centroids.ptr<float>(k)[d] = rng.uniform(box[d][0], box[d][1]);
			}
		}
	}


	//slow
	float KMeansPPDistanceComputerSoASingleDNLoop(__m256* dest_distance, const Mat& data_points, const __m256* src_distance, const int centroid_index)
	{
		const int dims = data_points.rows;
		const int simd_width = data_points.cols / 8;

		std::vector<__m256*> dim(dims);
		{
			int d = 0;
			const float* p = data_points.ptr<float>(d);
			dim[d] = (__m256*)p;
			const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

			for (int i = 0; i < simd_width; i++)
			{
				dest_distance[i] = normL2Sqr(dim[d][i], centers_value);
			}
		}
		for (int d = 1; d < dims; d++)
		{
			const float* p = data_points.ptr<float>(d);
			dim[d] = (__m256*)p;
			const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

			for (int i = 0; i < simd_width; i++)
			{
				dest_distance[i] = normL2SqrAdd(dim[d][i], centers_value, dest_distance[i]);
			}
		}

		float ret = 0.f;
		__m256 tdist2_acc = _mm256_setzero_ps();
		for (int i = 0; i < simd_width; i++)
		{
			dest_distance[i] = _mm256_min_ps(dest_distance[i], src_distance[i]);
			tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance[i]);
		}
		ret += _mm256_reduceadd_ps(tdist2_acc);
		return ret;
	}

	template<int dims>
	float KMeansPPDistanceComputerSoASingleNDLoop(__m256* dest_distance, const Mat& data_points, const __m256* src_distance, const int centroid_index)
	{
		const int simd_width = data_points.cols / 8;

		AutoBuffer<__m256*> dim(dims);
		AutoBuffer<__m256> centers_value(dims);
		for (int d = 0; d < dims; d++)
		{
			const float* p = data_points.ptr<float>(d);
			dim[d] = (__m256*)p;
			centers_value[d] = _mm256_set1_ps(p[centroid_index]);
		}

		float ret = 0.f;
		__m256 tdist2_acc = _mm256_setzero_ps();
		for (int i = 0; i < simd_width; i++)
		{
			__m256 v = normL2Sqr(dim[0][i], centers_value[0]);
			for (int d = 1; d < dims; d++)
			{
				v = normL2SqrAdd(dim[d][i], centers_value[d], v);
			}
			dest_distance[i] = _mm256_min_ps(v, src_distance[i]);
			tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance[i]);
		}
		ret += _mm256_reduceadd_ps(tdist2_acc);
		return ret;
	}

	template<>
	float KMeansPPDistanceComputerSoASingleNDLoop<3>(__m256* dest_distance, const Mat& data_points, const __m256* src_distance, const int centroid_index)
	{
		const int simd_width = data_points.cols / 8;

		AutoBuffer<__m256*> dim(3);
		AutoBuffer<__m256> centers_value(3);
		for (int d = 0; d < 3; d++)
		{
			const float* p = data_points.ptr<float>(d);
			dim[d] = (__m256*)p;
			centers_value[d] = _mm256_set1_ps(p[centroid_index]);
		}

		float ret = 0.f;
		__m256 tdist2_acc = _mm256_setzero_ps();
		for (int i = 0; i < simd_width; i++)
		{
			__m256 v = normL2Sqr(dim[0][i], centers_value[0]);
			v = normL2SqrAdd(dim[1][i], centers_value[1], v);
			v = normL2SqrAdd(dim[2][i], centers_value[2], v);
			dest_distance[i] = _mm256_min_ps(v, src_distance[i]);
			tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance[i]);
		}
		ret += _mm256_reduceadd_ps(tdist2_acc);
		return ret;
	}

	float KMeansPPDistanceComputerSoASingleNDLoop(__m256* dest_distance, const Mat& data_points, const __m256* src_distance, const int centroid_index)
	{
		const int dims = data_points.rows;
		if (dims == 1) return KMeansPPDistanceComputerSoASingleNDLoop<1>(dest_distance, data_points, src_distance, centroid_index);
		if (dims == 2) return KMeansPPDistanceComputerSoASingleNDLoop<2>(dest_distance, data_points, src_distance, centroid_index);
		if (dims == 3) return KMeansPPDistanceComputerSoASingleNDLoop<3>(dest_distance, data_points, src_distance, centroid_index);
		if (dims == 4) return KMeansPPDistanceComputerSoASingleNDLoop<4>(dest_distance, data_points, src_distance, centroid_index);
		if (dims == 5) return KMeansPPDistanceComputerSoASingleNDLoop<5>(dest_distance, data_points, src_distance, centroid_index);

		const int simd_width = data_points.cols / 8;

		AutoBuffer<__m256*> dim(dims);
		AutoBuffer<__m256> centers_value(dims);
		for (int d = 0; d < dims; d++)
		{
			const float* p = data_points.ptr<float>(d);
			dim[d] = (__m256*)p;
			centers_value[d] = _mm256_set1_ps(p[centroid_index]);
		}

		__m256 tdist2_acc = _mm256_setzero_ps();
		for (int i = 0; i < simd_width; i++)
		{
			__m256 v = normL2Sqr(dim[0][i], centers_value[0]);
			for (int d = 1; d < dims; d++)
			{
				v = normL2SqrAdd(dim[d][i], centers_value[d], v);
			}
			dest_distance[i] = _mm256_min_ps(v, src_distance[i]);
			tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance[i]);
		}
		float ret = _mm256_reduceadd_ps(tdist2_acc);
		return ret;
	}

	float KMeansMSPPDistanceComputerSoASingleInitNDLoop(__m256* dest_distance1st, __m256* dest_distance2nd, __m256* dest_index1st, __m256* dest_index2nd, const Mat& data_points, const __m256* src_distance1st, const __m256* src_distance2nd, const __m256* src_index1st, const __m256* src_index2nd, const int centroid_index, const int kindex)
	{
		if (false)
		{
			const int dims = data_points.rows;
			const int simd_width = data_points.cols / 8;

			AutoBuffer<__m256*> dim(dims);
			AutoBuffer<__m256> centers_value(dims);
			for (int d = 0; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				centers_value[d] = _mm256_set1_ps(p[centroid_index]);
			}

			float ret = 0.f;
			__m256 tdist2_acc = _mm256_setzero_ps();
			for (int i = 0; i < simd_width; i++)
			{
				__m256 v = normL2Sqr(dim[0][i], centers_value[0]);
				for (int d = 1; d < dims; d++)
				{
					v = normL2SqrAdd(dim[d][i], centers_value[d], v);
				}
				dest_index1st[i] = src_index1st[i];
				_mm256_argmin_ps(src_distance1st[i], v, dest_index1st[i], kindex);
				dest_distance1st[i] = v;
				tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance1st[i]);
			}
			ret += _mm256_reduceadd_ps(tdist2_acc);
			return ret;
		}
		else
		{
			const int dims = data_points.rows;
			AutoBuffer<const float*> dim(dims);
			AutoBuffer<float> centers_value(dims);
			for (int d = 0; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = p;
				centers_value[d] = p[centroid_index];
			}
			const float* sdist_1st = (const float*)src_distance1st;
			const float* sdist_2nd = (const float*)src_distance2nd;
			float* ddist_1st = (float*)dest_distance1st;
			float* ddist_2nd = (float*)dest_distance2nd;
			const float* sidx_1st = (const float*)src_index1st;
			const float* sidx_2nd = (const float*)src_index2nd;
			float* didx_1st = (float*)dest_index1st;
			float* didx_2nd = (float*)dest_index2nd;
			float ret = 0.f;
			for (int i = 0; i < data_points.cols; i++)
			{
				float v = normL2Sqr(dim[0][i], centers_value[0]);
				for (int d = 1; d < dims; d++)
				{
					v = normL2SqrAdd(dim[d][i], centers_value[d], v);
				}

				ddist_1st[i] = sdist_1st[i];
				didx_1st[i] = sidx_1st[i];
				ddist_2nd[i] = sdist_2nd[i];
				didx_2nd[i] = sidx_2nd[i];

				if (v < sdist_2nd[i])
				{
					ddist_2nd[i] = v;
					didx_2nd[i] = kindex;
				}
				if (v < sdist_1st[i])
				{
					ddist_2nd[i] = sdist_1st[i];
					didx_2nd[i] = sidx_1st[i];
					ddist_1st[i] = v;
					didx_1st[i] = kindex;
				}
				ret += ddist_1st[i];
			}
			return ret;
		}
	}

	float KMeansMSPPDistanceComputerSoASingleNDLoop(__m256* dest_distance1st, __m256* dest_distance2nd, __m256* dest_index1st, __m256* dest_index2nd, const Mat& data_points,
		const __m256* src_distance1st, const __m256* src_distance2nd, const __m256* src_index1st, const __m256* src_index2nd, const int centroid_index, const int remove_kindex)
	{
		const int dims = data_points.rows;
		if (false)
		{
			const int simd_width = data_points.cols / 8;
			const __m256 mkindex = _mm256_set1_ps((float)remove_kindex);
			AutoBuffer<__m256*> dim(dims);
			AutoBuffer<__m256> centers_value(dims);
			for (int d = 0; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				centers_value[d] = _mm256_set1_ps(p[centroid_index]);
			}

			float ret = 0.f;
			__m256 tdist2_acc = _mm256_setzero_ps();
			for (int i = 0; i < simd_width; i++)
			{
				__m256 v = normL2Sqr(dim[0][i], centers_value[0]);
				for (int d = 1; d < dims; d++)
				{
					v = normL2SqrAdd(dim[d][i], centers_value[d], v);
				}
				const __m256 mask = _mm256_cmp_ps(mkindex, src_index1st[i], 0);
				//dest_distance[i] = _mm256_blendv_ps(src_distance[i], v, mask);
				//dest_index[i] = _mm256_blendv_ps(src_index[i], mkindex, mask);
				dest_distance1st[i] = _mm256_blendv_ps(v, src_distance1st[i], mask);
				dest_index1st[i] = _mm256_blendv_ps(mkindex, src_index1st[i], mask);

				tdist2_acc = _mm256_add_ps(tdist2_acc, dest_distance1st[i]);
			}
			ret += _mm256_reduceadd_ps(tdist2_acc);
			return ret;
		}
		else
		{
			AutoBuffer<const float*> dim(dims);
			AutoBuffer<float> centers_value(dims);
			for (int d = 0; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = p;
				centers_value[d] = p[centroid_index];
			}
			const float* sdst_1st = (const float*)src_distance1st;
			const float* sdst_2nd = (const float*)src_distance2nd;
			float* ddst_1st = (float*)dest_distance1st;
			float* ddst_2nd = (float*)dest_distance2nd;
			const float* sidx_1st = (const float*)src_index1st;
			const float* sidx_2nd = (const float*)src_index2nd;
			float* didx_1st = (float*)dest_index1st;
			float* didx_2nd = (float*)dest_index2nd;
			float ret = 0.f;
			for (int i = 0; i < data_points.cols; i++)
			{
				float v = normL2Sqr(dim[0][i], centers_value[0]);
				for (int d = 1; d < dims; d++)
				{
					v = normL2SqrAdd(dim[d][i], centers_value[d], v);
				}

				ddst_1st[i] = sdst_1st[i];
				didx_1st[i] = sidx_1st[i];
				ddst_2nd[i] = sdst_2nd[i];
				didx_2nd[i] = sidx_2nd[i];

				if (sidx_2nd[i] == float(remove_kindex))
				{
					ddst_2nd[i] = v;
				}

				if (sidx_1st[i] == float(remove_kindex))
				{
					if (v < sdst_1st[i])
					{
						ddst_1st[i] = v;
						didx_1st[i] = remove_kindex;
					}
					else
					{
						ddst_1st[i] = sdst_2nd[i];
						didx_1st[i] = sidx_2nd[i];
					}
				}

				ret += ddst_1st[i];
			}
			return ret;
		}
	}

	class KMeansPPDistanceComputerSoA_AVX : public ParallelLoopBody
	{
	private:
		const __m256* src_distance;
		__m256* dest_distance;
		const Mat& data_points;
		const int centroid_index;
	public:
		KMeansPPDistanceComputerSoA_AVX(__m256* dest_dist, const Mat& data_points, const __m256* src_distance, const int centroid_index) :
			dest_distance(dest_dist), data_points(data_points), src_distance(src_distance), centroid_index(centroid_index)
		{ }

		void operator()(const cv::Range& range) const CV_OVERRIDE
		{
			const int begin = range.start;
			const int end = range.end;
			const int dims = data_points.rows;

			const int simd_width = data_points.cols / 8;

			std::vector<__m256*> dim(dims);
			{
				int d = 0;
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

				for (int i = 0; i < simd_width; i++)
				{
					dest_distance[i] = normL2Sqr(dim[d][i], centers_value);
				}
			}
			for (int d = 1; d < dims; d++)
			{
				const float* p = data_points.ptr<float>(d);
				dim[d] = (__m256*)p;
				const __m256 centers_value = _mm256_set1_ps(p[centroid_index]);

				for (int i = 0; i < simd_width; i++)
				{
					dest_distance[i] = normL2SqrAdd(dim[d][i], centers_value, dest_distance[i]);
				}
			}

			for (int i = 0; i < simd_width; i++)
			{
				dest_distance[i] = _mm256_min_ps(dest_distance[i], src_distance[i]);
			}
		}
	};

	float getWeightedSamplingDistance(const Mat& data_points, const Mat& weight, const int dims, const int size, const int index, __m256* dist)
	{
		AutoBuffer<const float*> p(dims);
		AutoBuffer<__m256> centers_value(dims);
		const float* wp = weight.ptr<float>();
		for (int d = 0; d < dims; d++)
		{
			p[d] = data_points.ptr<float>(d);
			centers_value[d] = _mm256_set1_ps(p[d][index]);
		}

		float distance_sum = 0.f;
		__m256 dist_value_acc = _mm256_setzero_ps();
		for (int i = 0; i < size; i++)
		{
			const __m256 mw = _mm256_sub_ps(_mm256_set1_ps(1.f), _mm256_loadu_ps(wp + 8 * i));
			//const __m256 mw =  _mm256_loadu_ps(wp + 8 * i);
			__m256 dist_value = cp::normL2Sqr(_mm256_loadu_ps(p[0] + 8 * i), centers_value[0]);
			for (int d = 1; d < dims; d++)
			{
				dist_value = cp::normL2SqrAdd(_mm256_loadu_ps(p[d] + 8 * i), centers_value[d], dist_value);
			}
			dist_value = _mm256_mul_ps(dist_value, mw);
			dist[i] = dist_value;
			dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
		}
		distance_sum = _mm256_reduceadd_ps(dist_value_acc);
		return distance_sum;
	}

	float getSamplingDistance(const Mat& data_points, const int dims, const int size, const int index, __m256* dist)
	{
		AutoBuffer<const float*> p(dims);
		AutoBuffer<__m256> centers_value(dims);
		for (int d = 0; d < dims; d++)
		{
			p[d] = data_points.ptr<float>(d);
			centers_value[d] = _mm256_set1_ps(p[d][index]);
		}

		float distance_sum = 0.f;
		__m256 dist_value_acc = _mm256_setzero_ps();
		for (int i = 0; i < size; i++)
		{
			{
				const __m256 md = _mm256_loadu_ps(p[0] + 8 * i);
				const __m256 dist_value = cp::normL2Sqr(md, centers_value[0]);
				dist[i] = dist_value;
				dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
			}
			for (int d = 1; d < dims; d++)
			{
				const __m256 md = _mm256_loadu_ps(p[d] + 8 * i);
				const __m256 dist_value = cp::normL2Sqr(md, centers_value[d]);
				dist[i] = _mm256_add_ps(dist[i], dist_value);
				// compute accumulate
				dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
			}
		}
		distance_sum = _mm256_reduceadd_ps(dist_value_acc);
		return distance_sum;
	}

	int getIndexProbabilitySampling(const float* disttop, const float prob, const int N)
	{
		float p = prob;
		int centroidIndex = 0;
		for (; centroidIndex < N - 8; centroidIndex += 8)
		{
			const float sub = _mm256_reduceadd_ps(_mm256_loadu_ps(disttop + centroidIndex));

			if (p - sub <= 0.f)
			{
				for (int v = 0; v < 8; v++)
				{
					p -= disttop[centroidIndex + v];
					if (p <= 0.f)
					{
						centroidIndex += v;
						return centroidIndex;
					}
				}
			}
			else
			{
				p -= sub;
			}
		}
		for (; centroidIndex < N - 1; centroidIndex++)
		{
			p -= disttop[centroidIndex];
			if (p <= 0.f) return centroidIndex;;
		}
		return centroidIndex;
	}

	float getWeightedDistance(const float* distance, const float* weight, const int size)
	{
		__m256 msum = _mm256_setzero_ps();
		for (int i = 0; i < size; i += 8)
		{
			msum = _mm256_fmadd_ps(_mm256_loadu_ps(distance + i), _mm256_loadu_ps(weight + i), msum);
		}
		return _mm256_reduceadd_ps(msum);
	}

	int getIndexWeightedProbabilitySampling(const float* disttop, const float* weight, const float prob, const int N)
	{
		float p = prob;
		int centroidIndex = 0;
		for (; centroidIndex < N - 8; centroidIndex += 8)
		{
			__m256 v = _mm256_mul_ps(_mm256_loadu_ps(weight + centroidIndex), _mm256_loadu_ps(disttop + centroidIndex));
			const float sub = _mm256_reduceadd_ps(v);

			if (p - sub <= 0.f)
			{
				for (int v = 0; v < 8; v++)
				{
					p -= weight[centroidIndex + v] * disttop[centroidIndex + v];
					if (p <= 0.f)
					{
						centroidIndex += v;
						return centroidIndex;
					}
				}
			}
			else
			{
				p -= sub;
			}
		}
		for (; centroidIndex < N - 1; centroidIndex++)
		{
			p -= weight[centroidIndex] * disttop[centroidIndex];
			if (p <= 0.f) return centroidIndex;;
		}
		return centroidIndex;
	}

	void weightInv(const Mat& weight, Mat& dest)
	{
		dest.create(weight.size(), CV_32F);
		const int size = weight.size().area();
		const float* w = weight.ptr<float>();
		float* d = dest.ptr<float>();
		__m256 mone = _mm256_set1_ps(1.f);
		for (int i = 0; i < size; i += 8)
		{
			//__m256 v = _mm256_sub_ps(mone, _mm256_loadu_ps(w + i));
			__m256 v = _mm256_rcp_ps(_mm256_loadu_ps(w + i));
			//v = _mm256_mul_ps(v, v);
			//v = _mm256_mul_ps(v, v);
			/*v = _mm256_sqrt_ps(v);
			v = _mm256_sqrt_ps(v);
			v = _mm256_sqrt_ps(v);*/
			_mm256_storeu_ps(d + i, v);
			//_mm256_storeu_ps(d + i, mone);
		}
	}

	void KMeans::generateWeightedKmeansPPInitialCentroidSoA(const Mat& data_points, const Mat& weight, Mat& dest_centroids, int K, RNG& rng, int trials)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int simdNfloor = N / 8;
		cv::AutoBuffer<int, 64> centersData(K);
		int* centersPtr = &centersData[0];

		//3 buffers; dist, tdist, tdist2.
		if (distancePP.size() != N * 3) distancePP.allocate(N * 3);

		__m256* dist = (__m256*)(&distancePP[0]);
		__m256* dist_tmp = dist + simdNfloor;
		__m256* dist_swp = dist_tmp + simdNfloor;

		//randomize the first centroid
		centersPtr[0] = (unsigned int)rng % N;//uniform sampling
		//centersPtr[0] = N / 2;
		//float distance_sum = getWeightedSamplingDistance(data_points, weight, dims, simdNfloor, centersPtr[0], dist);//summation of L2 distance between first sample and all samples
		Mat wi;
		//cout << sum(weight) << endl;
		weightInv(weight, wi);
		float distance_sum = getSamplingDistance(data_points, dims, simdNfloor, centersPtr[0], dist);//summation of L2 distance between first sample and all samples
		for (int k = 1; k < K; k++)
		{
			float bestSum = FLT_MAX;
			int bestCenter = -1;
			const float* disttop = (float*)dist;
			const int iter = trials;
			//const int iter = 1 + 2*trials * k / (K - 1);
			//const int iter = 1 + trials * (1.0 - k / (K - 1));
			for (int j = 0; j < iter; j++)
			{
				//float dd = getWeightedDistance((const float*)dist, wi.ptr<float>(), N);
				//print_debug2(dd, distance_sum);
				const float p = (float)rng * getWeightedDistance((const float*)dist, wi.ptr<float>(), N);
				//const float p = (float)rng * distance_sum;//original
				//float p = rng.uniform(0.1f, 1.f) * distance_sum;
				//const int centroidIndex = getIndexProbabilitySampling(disttop, p, N);
				const int centroidIndex = getIndexWeightedProbabilitySampling(disttop, wi.ptr<float>(), p, N);

				float distance_sum_local = 0.f;
				const int parallel = cv::getNumThreads();
				if (parallel != 1)
				{
					KMeansPPDistanceComputerSoA_AVX plb(dist_swp, data_points, dist, centroidIndex);
					parallel_for_(Range(0, N), plb, parallel);
					__m256 tdist2_acc = _mm256_setzero_ps();
					for (int i = 0; i < simdNfloor; i++)
					{
						tdist2_acc = _mm256_add_ps(tdist2_acc, dist_swp[i]);
					}
					distance_sum_local += _mm256_reduceadd_ps(tdist2_acc);
				}
				else
				{
					distance_sum_local = KMeansPPDistanceComputerSoASingleNDLoop(dist_swp, data_points, dist, centroidIndex);
				}

				if (distance_sum_local < bestSum)
				{
					bestSum = distance_sum_local;
					bestCenter = centroidIndex;
					std::swap(dist_tmp, dist_swp);
				}
			}

			if (bestCenter < 0)
			{
				CV_Error(Error::StsNoConv, "kmeans (SoA): can't update cluster center (check input for huge or NaN values)");
			}

			centersPtr[k] = bestCenter;//in intensity index, where have minimum distance
			distance_sum = bestSum;
			std::swap(dist, dist_tmp);
		}

		for (int k = 0; k < K; k++)
		{
			float* dst = dest_centroids.ptr<float>(k);
			const int idx = centersPtr[k];
			for (int d = 0; d < dims; d++)
			{
				dst[d] = data_points.at<float>(d, idx);
			}
		}
	}


	//k - means center initialization using the following algorithm :
	//Arthur & Vassilvitskii(2007) k-means++ : The Advantages of Careful Seeding
	//centroids: dims x k
	void KMeans::generateKmeansPPInitialCentroidSoA(const Mat& data_points, Mat& dest_centroids, int K, RNG& rng, int trials)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int simdNfloor = N / 8;
		cv::AutoBuffer<int, 64> centersData(K);
		int* centersPtr = &centersData[0];

		//3 buffers; dist, tdist, tdist2.
		if (distancePP.size() != N * 3) distancePP.allocate(N * 3);

		__m256* dist = (__m256*)(&distancePP[0]);
		__m256* dist_tmp = dist + simdNfloor;
		__m256* dist_swp = dist_tmp + simdNfloor;

		//randomize the first centroid
		centersPtr[0] = (unsigned int)rng % N;//uniform sampling
		//centersPtr[0] = N / 2;
		float distance_sum = getSamplingDistance(data_points, dims, simdNfloor, centersPtr[0], dist);//summation of L2 distance between first sample and all samples

		for (int k = 1; k < K; k++)
		{
			float bestSum = FLT_MAX;
			int bestCenter = -1;
			const float* disttop = (float*)dist;
			const int iter = trials;
			//const int iter = 1 + 2*trials * k / (K - 1);
			//const int iter = 1 + trials * (1.0 - k / (K - 1));
			for (int j = 0; j < iter; j++)
			{
				const float p = (float)rng * distance_sum;//original
				//float p = rng.uniform(0.1f, 1.f) * distance_sum;
				const int centroidIndex = getIndexProbabilitySampling(disttop, p, N);

				float distance_sum_local = 0.f;
				const int parallel = cv::getNumThreads();
				if (parallel != 1)
				{
					KMeansPPDistanceComputerSoA_AVX plb(dist_swp, data_points, dist, centroidIndex);
					parallel_for_(Range(0, N), plb, parallel);
					__m256 tdist2_acc = _mm256_setzero_ps();
					for (int i = 0; i < simdNfloor; i++)
					{
						tdist2_acc = _mm256_add_ps(tdist2_acc, dist_swp[i]);
					}
					distance_sum_local += _mm256_reduceadd_ps(tdist2_acc);
				}
				else
				{
					distance_sum_local = KMeansPPDistanceComputerSoASingleNDLoop(dist_swp, data_points, dist, centroidIndex);
				}

				if (distance_sum_local < bestSum)
				{
					bestSum = distance_sum_local;
					bestCenter = centroidIndex;
					std::swap(dist_tmp, dist_swp);
				}
			}

			if (bestCenter < 0)
			{
				CV_Error(Error::StsNoConv, "kmeans (SoA): can't update cluster center (check input for huge or NaN values)");
			}

			centersPtr[k] = bestCenter;//in intensity index, where have minimum distance
			distance_sum = bestSum;
			std::swap(dist, dist_tmp);
		}

		for (int k = 0; k < K; k++)
		{
			float* dst = dest_centroids.ptr<float>(k);
			const int idx = centersPtr[k];
			for (int d = 0; d < dims; d++)
			{
				dst[d] = data_points.at<float>(d, idx);
			}
		}
	}

	void KMeans::generateKmeansMSPPInitialCentroidSoA(const Mat& data_points, Mat& dest_centroids, int K, RNG& rng, int trials)
	{
#if 1
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int simdNfloor = N / 8;
		cv::AutoBuffer<int, 64> centersData(K);
		int* centersPtr = &centersData[0];

		//3 buffers; dist, tdist, tdist2.
		//if (distancePP.size() != N * 6) distancePP.allocate(N * 12);

		if (true)
		{
			//print_debug2(trials, KMEANSREPP_TRIALS);
			if (distancePP.size() != N * 3) distancePP.allocate(N * 3);
			cv::AutoBuffer<int, 64> centersDataRetry(K);
			int* centersRetryPtr = &centersDataRetry[0];
			float distance_retry = FLT_MAX;
			for (int n = 0; n < KMEANSREPP_TRIALS; n++)
			{
				__m256* dist = (__m256*)(&distancePP[0]);
				__m256* dist_tmp = dist + simdNfloor;
				__m256* dist_swp = dist_tmp + simdNfloor;

				//randomize the first centroid
				if (n == 0)
				{
					centersPtr[0] = (unsigned int)rng % N;
				}
				else
				{
					centersPtr[0] = (unsigned int)rng % N;
					//const float* disttop = (float*)dist;
					//const float p = (float)rng * distance_retry;//original
					//centersPtr[0] = getIndexProbabilitySampling(disttop, p, N);
				}
				//centersPtr[0] = N / 2;				
				float distance_sum = getSamplingDistance(data_points, dims, simdNfloor, centersPtr[0], dist);//summation of L2 distance between first sample and all samples

				for (int k = 1; k < K; k++)
				{
					float bestSum = FLT_MAX;
					int bestCenter = -1;
					const float* disttop = (float*)dist;

					//const int iter = trials;
					//const int iter = 1 + 2 * trials * k / (K - 1);
					const int iter = 1 + saturate_cast<int>(trials * (float(k) / (K - 1)));
					for (int j = 0; j < iter; j++)
					{
						const float p = (float)rng * distance_sum;//original
						const int centroidIndex = getIndexProbabilitySampling(disttop, p, N);

						float distance_sum_local = 0.f;
						const int parallel = cv::getNumThreads();
						if (parallel != 1)
						{
							KMeansPPDistanceComputerSoA_AVX plb(dist_swp, data_points, dist, centroidIndex);
							parallel_for_(Range(0, N), plb, parallel);
							__m256 tdist2_acc = _mm256_setzero_ps();
							for (int i = 0; i < simdNfloor; i++)
							{
								tdist2_acc = _mm256_add_ps(tdist2_acc, dist_swp[i]);
							}
							distance_sum_local += _mm256_reduceadd_ps(tdist2_acc);
						}
						else
						{
							distance_sum_local = KMeansPPDistanceComputerSoASingleNDLoop(dist_swp, data_points, dist, centroidIndex);
						}

						if (distance_sum_local < bestSum)
						{
							bestSum = distance_sum_local;
							bestCenter = centroidIndex;
							std::swap(dist_tmp, dist_swp);
						}
					}

					if (bestCenter < 0)
					{
						CV_Error(Error::StsNoConv, "kmeans (SoA): can't update cluster center (check input for huge or NaN values)");
					}

					centersPtr[k] = bestCenter;//in intensity index, where have minimum distance
					distance_sum = bestSum;
					std::swap(dist, dist_tmp);
				}

				if (distance_sum < distance_retry)
				{
					distance_retry = distance_sum;
					for (int k = 0; k < K; k++)
					{
						centersRetryPtr[k] = centersPtr[k];
					}
				}
			}

			for (int k = 0; k < K; k++)
			{
				float* dst = dest_centroids.ptr<float>(k);
				const int idx = centersRetryPtr[k];
				for (int d = 0; d < dims; d++)
				{
					dst[d] = data_points.at<float>(d, idx);
				}
			}
		}
		else
		{
			__m256* dist_1st = (__m256*)(&distancePP[0]);
			__m256* dist_1st_tmp = dist_1st + simdNfloor;
			__m256* dist_1st_swp = dist_1st_tmp + simdNfloor;
			__m256* idx_1st = dist_1st_swp + simdNfloor;
			__m256* idx_1st_tmp = idx_1st + simdNfloor;
			__m256* idx_1st_swp = idx_1st_tmp + simdNfloor;
			__m256* dist_2nd = idx_1st_swp + simdNfloor;
			__m256* dist_2nd_tmp = dist_2nd + simdNfloor;
			__m256* dist_2nd_swp = dist_2nd_tmp + simdNfloor;
			__m256* idx_2nd = dist_2nd_swp + simdNfloor;
			__m256* idx_2nd_tmp = idx_2nd + simdNfloor;
			__m256* idx_2nd_swp = idx_2nd_tmp + simdNfloor;

			float distance_sum = 0.f;//summation of L2 distance between first sample and all samples
			if (true)
			{
				//randomize the first centroid
				centersPtr[0] = (unsigned int)rng % N;
				//centersPtr[0] = N / 2;
				for (int i = 0; i < simdNfloor; i++)
				{
					dist_1st[i] = _mm256_setzero_ps();
					idx_1st[i] = _mm256_setzero_ps();
					dist_2nd[i] = _mm256_set1_ps(FLT_MAX);
					idx_1st[i] = _mm256_setzero_ps();
				}
				for (int d = 0; d < dims; d++)
				{
					const float* p = data_points.ptr<float>(d);
					__m256 centers_value = _mm256_set1_ps(p[centersPtr[0]]);
					__m256 dist_value_acc = _mm256_setzero_ps();
					for (int i = 0; i < simdNfloor; i++)
					{
						// compute dist[i]
						const __m256 md = _mm256_loadu_ps(p + 8 * i);
						const __m256 dist_value = cp::normL2Sqr(md, centers_value);
						dist_1st[i] = _mm256_add_ps(dist_1st[i], dist_value);
						// compute accumulate
						dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
					}
					distance_sum += _mm256_reduceadd_ps(dist_value_acc);
				}
			}
			else //retry
			{
				__m256* dist_psh = dist_1st_swp + simdNfloor;

				//float error_min = FLT_MAX;
				float error_min = 0.f;
				for (int j = 0; j < trials; j++)
				{
					//randomize the first centroid
					const int ridx = (unsigned int)rng % N;
					for (int i = 0; i < simdNfloor; i++)
					{
						dist_psh[i] = _mm256_setzero_ps();
					}
					float distance_sum_1st = 0.f;
					for (int d = 0; d < dims; d++)
					{
						const float* p = data_points.ptr<float>(d);
						const __m256 centers_value = _mm256_set1_ps(p[ridx]);
						__m256 dist_value_acc = _mm256_setzero_ps();
						for (int i = 0; i < simdNfloor; i++)
						{
							// compute dist[i]
							const __m256 md = _mm256_loadu_ps(p + 8 * i);
							const __m256 dist_value = cp::normL2Sqr(md, centers_value);
							dist_psh[i] = _mm256_add_ps(dist_psh[i], dist_value);
							// compute accumulate
							dist_value_acc = _mm256_add_ps(dist_value_acc, dist_value);
						}
						distance_sum_1st += _mm256_reduceadd_ps(dist_value_acc);
					}

					//if (distance_sum_1st < error_min)
					if (distance_sum_1st > error_min)
					{
						error_min = distance_sum_1st;
						centersPtr[0] = ridx;
						for (int i = 0; i < simdNfloor; i++)
						{
							dist_1st[i] = dist_psh[i];
						}
					}
				}
				distance_sum = error_min;
			}

			for (int k = 1; k < K; k++)
			{
				float bestSum = FLT_MAX;
				int bestCenter = -1;
				const float* disttop = (float*)dist_1st;
				const int iter = trials;
				//const int iter = 1 + 2*trials * k / (K - 1);
				//const int iter = 1 + trials * (1.0 - k / (K - 1));
				for (int j = 0; j < iter; j++)
				{
					int centroidIndex = 0;
					{
						float p = (float)rng * distance_sum;//original
						//float p = rng.uniform(0.1f, 1.f) * distance_sum;
						for (; centroidIndex < N - 1; centroidIndex++)
						{
							p -= disttop[centroidIndex];
							if (p <= 0.f) break;
						}
					}

					float distance_sum_local = 0.f;
					const int parallel = cv::getNumThreads();
					if (parallel != 1)
					{
						/*
						KMeansPPDistanceComputerSoA_AVX plb(dist_swp, data_points, dist, centroidIndex);
						parallel_for_(Range(0, N), plb, parallel);
						__m256 tdist2_acc = _mm256_setzero_ps();
						for (int i = 0; i < simdNfloor; i++)
						{
							tdist2_acc = _mm256_add_ps(tdist2_acc, dist_swp[i]);
						}
						distance_sum_local += _mm256_reduceadd_ps(tdist2_acc);
						*/
					}
					else
					{
						//distance_sum_local = KMeansPPDistanceComputerSoASingleNDLoop(dist_swp, data_points, dist, centroidIndex);
						distance_sum_local = KMeansMSPPDistanceComputerSoASingleInitNDLoop(dist_1st_swp, dist_2nd_swp, idx_1st_swp, idx_2nd_swp, data_points, dist_1st, dist_2nd, idx_1st, idx_2nd, centroidIndex, k);
					}

					if (distance_sum_local < bestSum)
					{
						bestSum = distance_sum_local;
						bestCenter = centroidIndex;
						std::swap(dist_1st_tmp, dist_1st_swp);
						std::swap(dist_2nd_tmp, dist_2nd_swp);
						std::swap(idx_1st_tmp, idx_1st_swp);
						std::swap(idx_2nd_tmp, idx_2nd_swp);
					}
				}

				if (bestCenter < 0)
				{
					CV_Error(Error::StsNoConv, "kmeans (SoA): can't update cluster center (check input for huge or NaN values)");
				}

				centersPtr[k] = bestCenter;//in intensity index, where have minimum distance
				distance_sum = bestSum;
				std::swap(dist_1st, dist_1st_tmp);
				std::swap(dist_2nd, dist_2nd_tmp);
				std::swap(idx_1st, idx_1st_tmp);
				std::swap(idx_2nd, idx_2nd_tmp);
			}
#if 1
			const int iter = 30;
			int count = 0;
			for (int l = 0; l < iter; l++)
			{
				float bestSum = distance_sum;
				const float* disttop = (float*)dist_1st;
				int centroidIndex = 0;
				{
					float p = (float)rng * distance_sum;//original
					for (; centroidIndex < N - 1; centroidIndex++)
					{
						p -= disttop[centroidIndex];
						if (p <= 0.f) break;
					}
				}

				int argk = -1;
				for (int k = 0; k < K; k++)
				{
					float distance_sum_local = KMeansMSPPDistanceComputerSoASingleNDLoop(dist_1st_swp, dist_2nd_swp, idx_1st_swp, idx_2nd_swp, data_points, dist_1st, dist_2nd, idx_1st, idx_2nd, centroidIndex, k);
					if (distance_sum_local < bestSum)
					{
						argk = k;
						bestSum = distance_sum_local;
						std::swap(dist_1st_tmp, dist_1st_swp);
						std::swap(dist_2nd_tmp, dist_2nd_swp);
						std::swap(idx_1st_tmp, idx_1st_swp);
						std::swap(idx_2nd_tmp, idx_2nd_swp);
					}
					//print_debug3(k, distance_sum_local, distance_sum);
				}
				//print_debug4(l, argk, bestSum, distance_sum);
				if (argk >= 0)
				{
					count++;
					centersPtr[argk] = centroidIndex;//in intensity index, where have minimum distance
					distance_sum = bestSum;
					std::swap(dist_1st, dist_1st_tmp);
					std::swap(dist_2nd, dist_2nd_tmp);
					std::swap(idx_1st, idx_1st_tmp);
					std::swap(idx_2nd, idx_2nd_tmp);
				}
			}
			print_debug(count);
#endif
		}
		/*
		for (int k = 0; k < K; k++)
		{
			float* dst = dest_centroids.ptr<float>(k);
			for (int d = 0; d < dims; d++)
			{
				dst[d] = data_points.at<float>(d, centersPtr[k]);
			}
		}
		*/
#endif
	}
#pragma endregion

#pragma region updateCentroid
	void KMeans::getOuterSample(cv::Mat& src_centroids, cv::Mat& dest_centroids, const cv::Mat& data_points, const cv::Mat& labels)
	{
		const int N = data_points.cols;
		const int dims = data_points.rows;
		const int K = src_centroids.rows;
		cv::AutoBuffer<float, 64> Hcounters(K);

		for (int k = 0; k < K; k++) Hcounters[k] = 0.f;

		const int* l = labels.ptr<int>();
		for (int i = 0; i < N; i++)
		{
			const int arg_k = l[i];

			float dist = 0.f;
			for (int d = 0; d < dims; d++)
			{
				const float* dataPtr = data_points.ptr<float>(d);
				float diff = (src_centroids.ptr<float>(arg_k)[d] - dataPtr[i]);
				dist += diff * diff;
			}
			if (dist > Hcounters[arg_k])
			{
				Hcounters[arg_k] = dist;
				for (int d = 0; d < dims; d++)
				{
					const float* dataPtr = data_points.ptr<float>(d);
					dest_centroids.ptr<float>(arg_k)[d] = dataPtr[i];
				}
			}
		}
	}

	void KMeans::minmaxCentroidSoA(const Mat& data_points, const int* labels, Mat& dest_centroid, int* counters, const int K)
	{
		//cannot vectorize it without scatter
		const int dims = data_points.rows;
		const int N = data_points.cols;
		AutoBuffer<Vec3f> minv(K);
		AutoBuffer<Vec3f> maxv(K);
		for (int k = 0; k < K; k++)
		{
			minv[k] = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
			maxv[k] = -Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
		}
		{
			int d = 0;
			const float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				const int arg_k = labels[i];
				minv[arg_k].val[d] = min(dataPtr[i], minv[arg_k].val[d]);
				maxv[arg_k].val[d] = max(dataPtr[i], maxv[arg_k].val[d]);
				counters[arg_k]++;
			}
		}
		for (int d = 1; d < dims; d++)
		{
			const float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				const int arg_k = labels[i];
				minv[arg_k].val[d] = min(dataPtr[i], minv[arg_k].val[d]);
				maxv[arg_k].val[d] = max(dataPtr[i], maxv[arg_k].val[d]);
			}
		}

		for (int k = 0; k < K; k++)
		{
			if (counters[k] != 0)
			{
				for (int d = 0; d < dims; d++)
				{
					dest_centroid.ptr<float>(k)[d] = (maxv[k].val[d] + minv[k].val[d]) * 0.5f;
					//print_debug3(maxv[k].val[d], minv[k].val[d], dest_centroid.ptr<float>(k)[d]);
				}
			}
		}
	}
	//Nxdims
	void KMeans::boxMeanCentroidSoA(const Mat& data_points, const int* labels, Mat& dest_centroid, int* counters)
	{
		//cannot vectorize it without scatter
		const int dims = data_points.rows;
		const int N = data_points.cols;
		{
			int d = 0;
			const float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				int arg_k = labels[i];
				dest_centroid.ptr<float>(arg_k)[d] += dataPtr[i];
				counters[arg_k]++;
			}
		}
		for (int d = 1; d < dims; d++)
		{
			const float* dataPtr = data_points.ptr<float>(d);
			for (int i = 0; i < N; i++)
			{
				int arg_k = labels[i];
				dest_centroid.ptr<float>(arg_k)[d] += dataPtr[i];
			}
		}
	}

	//N*dims
	template<int dims>
	void weightedMeanCentroid_(const Mat& data_points, const int* labels, const Mat& src_centroid, const float* Table, const int tableSize, Mat& dest_centroid, float* dest_centroid_weight, int* dest_counters)
	{
		const int N = data_points.cols;
		const int K = src_centroid.rows;

		for (int k = 0; k < K; k++) dest_centroid_weight[k] = 0.f;

		cv::AutoBuffer<const float*, 64> dataTop(dims);
		for (int d = 0; d < dims; d++)
		{
			dataTop[d] = data_points.ptr<float>(d);
		}

#if 0
		//scalar
		cv::AutoBuffer<const float*, 64> centroidTop(K);
		for (int k = 0; k < K; k++)
		{
			centroidTop[k] = src_centroid.ptr<float>(k);
		}
		for (int i = 0; i < N; i++)
		{
			const int arg_k = labels[i];

			float dist = 0.f;
			for (int d = 0; d < dims; d++)
			{
				float diff = (centroidTop[arg_k][d] - dataTop[d][i]);
				dist += diff * diff;
			}
			const float wi = Table[int(sqrt(dist))];
			centroid_weight[arg_k] += wi;
			counters[arg_k]++;
			for (int d = 0; d < dims; d++)
			{
				dest_centroid.ptr<float>(arg_k)[d] += wi * dataTop[d][i];
			}
		}
#else
		if (true)
		{
			const float* centroidPtr = src_centroid.ptr<float>();//dim*K
			const __m256i mtsize = _mm256_set1_epi32(tableSize - 1);
			for (int i = 0; i < N; i += 8)
			{
				const __m256i marg_k = _mm256_load_si256((__m256i*)(labels + i));
				const __m256i midx = _mm256_mullo_epi32(marg_k, _mm256_set1_epi32(dims));
				__m256 mdist = _mm256_setzero_ps();
				for (int d = 0; d < dims; d++)
				{
					__m256 mc = _mm256_i32gather_ps(centroidPtr, _mm256_add_epi32(midx, _mm256_set1_epi32(d)), 4);
					mc = _mm256_sub_ps(mc, _mm256_load_ps(&dataTop[d][i]));
					mdist = _mm256_fmadd_ps(mc, mc, mdist);
				}

				const __m256 mwi = _mm256_i32gather_ps(Table, _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist))), 4);
				for (int v = 0; v < 8; v++)
				{
					const int arg_k = ((int*)&marg_k)[v];
					const float wi = ((float*)&mwi)[v];
					dest_centroid_weight[arg_k] += wi;
					dest_counters[arg_k]++;
					float* dstCentroidPtr = dest_centroid.ptr<float>(arg_k);
					for (int d = 0; d < dims; d++)
					{
						dstCentroidPtr[d] += wi * dataTop[d][i + v];
					}
				}
			}
		}
		else
		{
			Mat st; transpose(src_centroid, st);
			const __m256i mtsize = _mm256_set1_epi32(tableSize - 1);
			AutoBuffer <const float*> ctptr(dims);
			for (int d = 0; d < dims; d++)
			{
				ctptr[d] = st.ptr<float>(d);
			}
			for (int i = 0; i < N; i += 8)
			{
				const __m256i marg_k = _mm256_load_si256((__m256i*)(labels + i));
				__m256 mdist = _mm256_setzero_ps();

				for (int d = 0; d < dims; d++)
				{
					//__m256 mc = _mm256_permutevar8x32_ps(_mm256_load_ps(ctptr[d]), marg_k);
					__m256 mc = _mm256_i32gather_ps(ctptr[d], marg_k, 4);
					mc = _mm256_sub_ps(mc, _mm256_load_ps(&dataTop[d][i]));
					mdist = _mm256_fmadd_ps(mc, mc, mdist);
				}
				//__m256 mwi = _mm256_set1_ps(1);
				__m256 mwi = _mm256_i32gather_ps(Table, _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist))), 4);

				for (int v = 0; v < 8; v++)
				{
					const int arg_k = ((int*)&marg_k)[v];
					const float wi = ((float*)&mwi)[v];
					dest_centroid_weight[arg_k] += wi;
					dest_counters[arg_k]++;
					float* dstCentroidPtr = dest_centroid.ptr<float>(arg_k);
					for (int d = 0; d < dims; d++)
					{
						dstCentroidPtr[d] += wi * dataTop[d][i + v];
					}
				}
			}
		}
#endif 
	}

	void KMeans::weightedMeanCentroid(const Mat& data_points, const int* labels, const Mat& src_centroid, const float* Table, const int tableSize, Mat& dest_centroid, float* dest_centroid_weight, int* dest_counters)
	{
		const int dims = data_points.rows;
		if (dims == 1) weightedMeanCentroid_<1>(data_points, labels, src_centroid, Table, tableSize, dest_centroid, dest_centroid_weight, dest_counters);
		else if (dims == 2) weightedMeanCentroid_<2>(data_points, labels, src_centroid, Table, tableSize, dest_centroid, dest_centroid_weight, dest_counters);
		else if (dims == 3) weightedMeanCentroid_<3>(data_points, labels, src_centroid, Table, tableSize, dest_centroid, dest_centroid_weight, dest_counters);
		else
		{
			const int N = data_points.cols;
			const int K = src_centroid.rows;

			for (int k = 0; k < K; k++) dest_centroid_weight[k] = 0.f;

			cv::AutoBuffer<const float*, 64> dataTop(dims);
			for (int d = 0; d < dims; d++)
			{
				dataTop[d] = data_points.ptr<float>(d);
			}

#if 0
			//scalar
			cv::AutoBuffer<const float*, 64> centroidTop(K);
			for (int k = 0; k < K; k++)
			{
				centroidTop[k] = src_centroid.ptr<float>(k);
			}
			for (int i = 0; i < N; i++)
			{
				const int arg_k = labels[i];

				float dist = 0.f;
				for (int d = 0; d < dims; d++)
				{
					float diff = (centroidTop[arg_k][d] - dataTop[d][i]);
					dist += diff * diff;
				}
				const float wi = Table[int(sqrt(dist))];
				centroid_weight[arg_k] += wi;
				counters[arg_k]++;
				for (int d = 0; d < dims; d++)
				{
					dest_centroid.ptr<float>(arg_k)[d] += wi * dataTop[d][i];
				}
			}
#else
			const float* centroidPtr = src_centroid.ptr<float>();//dim*K
			const __m256i mtsize = _mm256_set1_epi32(tableSize - 1);
			for (int i = 0; i < N; i += 8)
			{
				const __m256i marg_k = _mm256_load_si256((__m256i*)(labels + i));
				const __m256i midx = _mm256_mullo_epi32(marg_k, _mm256_set1_epi32(dims));
				__m256 mdist = _mm256_setzero_ps();

				for (int d = 0; d < dims; d++)
				{
					__m256 mc = _mm256_i32gather_ps(centroidPtr, _mm256_add_epi32(midx, _mm256_set1_epi32(d)), 4);
					mc = _mm256_sub_ps(mc, _mm256_load_ps(&dataTop[d][i]));
					mdist = _mm256_fmadd_ps(mc, mc, mdist);
					//mdist = _mm256_add_ps(_mm256_abs_ps(mc), mdist);
				}
				//__m256i a = _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist)));
				//print_m256i_int(a);
				// 
				//__m256 mwi = _mm256_i32gather_ps(Table, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist)), 4);
				//__m256 mwi = _mm256_i32gather_ps(Table, _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist))), 4);
				__m256 mwi = _mm256_i32gather_ps(Table, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(_mm256_sqrt_ps(mdist)))), 4);
				//__m256 mwi = _mm256_i32gather_ps(Table, _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(mtsize, _mm256_cvtps_epi32(mdist))), 4);
				for (int v = 0; v < 8; v++)
				{
					const int arg_k = ((int*)&marg_k)[v];
					const float wi = ((float*)&mwi)[v];
					dest_centroid_weight[arg_k] += wi;
					dest_counters[arg_k]++;
					float* dstCentroidPtr = dest_centroid.ptr<float>(arg_k);
					for (int d = 0; d < dims; d++)
					{
						dstCentroidPtr[d] += wi * dataTop[d][i + v];
					}
				}
			}
#endif 
		}
	}

	//N*dims
	void KMeans::harmonicMeanCentroid(const Mat& data_points, const int* labels, const Mat& src_centroid, Mat& dest_centroid, float* centroid_weight, int* counters)
	{
		const int dims = data_points.rows;
		const int N = data_points.cols;
		const int K = src_centroid.rows;
		for (int k = 0; k < K; k++) centroid_weight[k] = 0.f;

		for (int i = 0; i < N; i++)
		{
			float w = 0.f;
			const float p = 3.5f;
			const int arg_k = labels[i];

			float w0 = 0.f;
			float w1 = 0.f;
			for (int k = 0; k < K; k++)
			{
				float w0_ = 0.f;
				float w1_ = 0.f;
				for (int d = 0; d < dims; d++)
				{
					const float* dataPtr = data_points.ptr<float>(d);
					float diff = abs(src_centroid.ptr<float>(k)[d] - dataPtr[i]);
					if (diff == 0.f)diff += FLT_EPSILON;
					w0_ += pow(diff, -p - 2.f);
					w1_ += pow(diff, -p);
				}
				w0 += pow(w0_, 1.f / (-p - 2.f));
				w1 += pow(w1_, 1.f / (-p));
			}
			w = w0 / (w1 * w1);
			//std::cout <<i<<":"<< w <<", "<<w0<<","<<w1<< std::endl;

			centroid_weight[arg_k] += w;
			counters[arg_k]++;
			for (int d = 0; d < dims; d++)
			{
				const float* dataPtr = data_points.ptr<float>(d);
				dest_centroid.ptr<float>(arg_k)[d] += w * dataPtr[i];
			}
		}
	}
#pragma endregion

#pragma region assignCentroid
	template<bool onlyDistance, int loop>
	class KMeansDistanceComputer_SoADim : public ParallelLoopBody
	{
	private:
		KMeansDistanceComputer_SoADim& operator=(const KMeansDistanceComputer_SoADim&); // = delete

		float* distances;
		int* labels;
		const Mat& dataPoints;
		const Mat& centroids;

	public:
		KMeansDistanceComputer_SoADim(float* dest_distance,
			int* dest_labels,
			const Mat& dataPoints,
			const Mat& centroids)
			: distances(dest_distance),
			labels(dest_labels),
			dataPoints(dataPoints),
			centroids(centroids)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			const int dims = centroids.cols;//when color case, dim= 3
			//CV_TRACE_FUNCTION();
			const int K = centroids.rows;
			const int BEGIN = range.start;
			const int END = (range.end / 8) * 8;
			const bool PAD = ((range.end - range.start) % 8 == 0) ? false : true;
			const __m256i pmask = get_simd_residualmask_epi32(range.end - range.start);

			if constexpr (onlyDistance)
			{
				AutoBuffer<const float*, 64> dptr(dims);
				AutoBuffer<__m256> mc(dims);
				{
					const float* center = centroids.ptr<float>(0);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = dataPoints.ptr<float>(d);
						mc[d] = _mm256_set1_ps(center[d]);
					}
				}
				for (int n = BEGIN; n < END; n += 8)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
					}
					_mm256_storeu_ps(distances + n, mdist);
				}
				if (PAD)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + END, pmask), mc[d], mdist);
					}
					_mm256_maskstore_ps(distances + END, pmask, mdist);
				}
			}
			else
			{
				if (loop == KMeansDistanceLoop::KND)//loop k-n-d
				{
					AutoBuffer<const float*, 64> dptr(dims);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = dataPoints.ptr<float>(d);
					}

					AutoBuffer<__m256> mc(dims);
					//k=0
					{
						const float* center = centroids.ptr<float>(0);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n += 8)
						{
							__m256 mdist = normL2Sqr(_mm256_loadu_ps(dptr[0] + n), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
							}
							_mm256_storeu_ps(distances + n, mdist);
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						}
						if (PAD)
						{
							__m256 mdist = normL2Sqr(_mm256_maskload_ps(dptr[0] + END, pmask), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + END, pmask), mc[d], mdist);
							}
							_mm256_maskstore_ps(distances + END, pmask, mdist);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						}
					}
					for (int k = 1; k < K; k++)
					{
						const float* center = centroids.ptr<float>(k);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n += 8)
						{
							__m256 mdist = normL2Sqr(_mm256_loadu_ps(dptr[0] + n), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
							}
							const __m256 mpredist = _mm256_loadu_ps(distances + n);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_storeu_ps(distances + n, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_loadu_si256((__m256i*)(labels + n));
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
						if (PAD)
						{
							__m256 mdist = normL2Sqr(_mm256_maskload_ps(dptr[0] + END, pmask), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + END, pmask), mc[d], mdist);
							}
							const __m256 mpredist = _mm256_maskload_ps(distances + END, pmask);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_maskstore_ps(distances + END, pmask, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_maskload_epi32(labels + END, pmask);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
				}
				else //loop n-k-d
				{
					__m256* mdp = (__m256*)_mm_malloc(sizeof(__m256) * dims, AVX_ALIGN);
					for (int n = BEGIN; n < END; n += 8)
					{
						_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = _mm256_loadu_ps(dataPoints.ptr<float>(d, n));
						}
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							_mm256_storeu_ps(distances + n, mdist);
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));//d=0
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							const __m256 mpredist = _mm256_loadu_ps(distances + n);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_storeu_ps(distances + n, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_loadu_si256((__m256i*)(labels + n));
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
					if (PAD)
					{
						_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = _mm256_maskload_ps(dataPoints.ptr<float>(d, END), pmask);
						}
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							_mm256_maskstore_ps(distances + END, pmask, mdist);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));//d=0
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							const __m256 mpredist = _mm256_maskload_ps(distances + END, pmask);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_maskstore_ps(distances + END, pmask, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_maskload_epi32(labels + END, pmask);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
					_mm_free(mdp);
				}
			}
		}
	};

	//copy from KMeansDistanceComputer_SoADim
	template<bool onlyDistance, int loop, int dims>
	class KMeansDistanceComputer_SoA : public ParallelLoopBody
	{
	private:
		KMeansDistanceComputer_SoA& operator=(const KMeansDistanceComputer_SoA&); // = delete

		float* distances;
		int* labels;
		const Mat& dataPoints;
		const Mat& centroids;

	public:
		KMeansDistanceComputer_SoA(float* dest_distance,
			int* dest_labels,
			const Mat& dataPoints,
			const Mat& centroids)
			: distances(dest_distance),
			labels(dest_labels),
			dataPoints(dataPoints),
			centroids(centroids)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			//CV_TRACE_FUNCTION();
			const int K = centroids.rows;
			const int BEGIN = range.start;
			const int END = (range.end / 8) * 8;
			const bool PAD = ((range.end - range.start) % 8 == 0) ? false : true;
			const __m256i pmask = get_simd_residualmask_epi32(range.end - range.start);

			if constexpr (onlyDistance)
			{
				AutoBuffer<const float*, 64> dptr(dims);
				AutoBuffer<__m256> mc(dims);
				{
					const float* center = centroids.ptr<float>(0);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = dataPoints.ptr<float>(d);
						mc[d] = _mm256_set1_ps(center[d]);
					}
				}
				for (int n = BEGIN; n < END; n += 8)
				{
					__m256 mdist = normL2Sqr(_mm256_loadu_ps(dptr[0] + n), mc[0]);
					for (int d = 1; d < dims; d++)
					{
						mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
					}
					_mm256_storeu_ps(distances + n, mdist);
				}
				if (PAD)
				{
					__m256 mdist = _mm256_setzero_ps();
					for (int d = 0; d < dims; d++)
					{
						mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + n, pmask), mc[d], mdist);
					}
					_mm256_maskstore_ps(distances + n, pmask, mdist);
				}
			}
			else
			{
				if (loop == KMeansDistanceLoop::KND)//loop k-n-d
				{
					AutoBuffer<const float*, 64> dptr(dims);
					for (int d = 0; d < dims; d++)
					{
						dptr[d] = dataPoints.ptr<float>(d);
					}

					AutoBuffer<__m256> mc(dims);
					//k=0
					{
						const float* center = centroids.ptr<float>(0);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n += 8)
						{
							__m256 mdist = normL2Sqr(_mm256_loadu_ps(dptr[0] + n), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
							}
							_mm256_storeu_ps(distances + n, mdist);
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						}
						if (PAD)
						{
							__m256 mdist = normL2Sqr(_mm256_maskload_ps(dptr[0] + END, pmask), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + END, pmask), mc[d], mdist);
							}
							_mm256_maskstore_ps(distances + END, pmask, mdist);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						}
					}
					for (int k = 1; k < K; k++)
					{
						const float* center = centroids.ptr<float>(k);
						for (int d = 0; d < dims; d++)
						{
							mc[d] = _mm256_set1_ps(center[d]);
						}

						for (int n = BEGIN; n < END; n += 8)
						{
							__m256 mdist = normL2Sqr(_mm256_loadu_ps(dptr[0] + n), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_loadu_ps(dptr[d] + n), mc[d], mdist);
							}
							const __m256 mpredist = _mm256_loadu_ps(distances + n);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_storeu_ps(distances + n, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_loadu_si256((__m256i*)(labels + n));
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
						if (PAD)
						{
							__m256 mdist = normL2Sqr(_mm256_maskload_ps(dptr[0] + END, pmask), mc[0]);
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(_mm256_maskload_ps(dptr[d] + END, pmask), mc[d], mdist);
							}
							const __m256 mpredist = _mm256_maskload_ps(distances + END, pmask);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_maskstore_ps(distances + END, pmask, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_maskload_epi32(labels + END, pmask);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
				}
				else //loop n-k-d
				{
					__m256* mdp = (__m256*)_mm_malloc(sizeof(__m256) * dims, AVX_ALIGN);
					for (int n = BEGIN; n < END; n += 8)
					{
						_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = _mm256_loadu_ps(dataPoints.ptr<float>(d, n));
						}
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							_mm256_storeu_ps(distances + n, mdist);
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_setzero_si256());
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));//d=0
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							const __m256 mpredist = _mm256_loadu_ps(distances + n);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_storeu_ps(distances + n, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_loadu_si256((__m256i*)(labels + n));
							_mm256_storeu_si256((__m256i*)(labels + n), _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
					if (PAD)
					{
						_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						for (int d = 0; d < dims; d++)
						{
							mdp[d] = _mm256_maskload_ps(dataPoints.ptr<float>(d, END), pmask);
						}
						{
							int k = 0;
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							_mm256_maskstore_ps(distances + END, pmask, mdist);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_setzero_si256());
						}
						for (int k = 1; k < K; k++)
						{
							const float* center = centroids.ptr<float>(k);
							__m256 mdist = normL2Sqr(mdp[0], _mm256_set1_ps(center[0]));//d=0
							for (int d = 1; d < dims; d++)
							{
								mdist = normL2SqrAdd(mdp[d], _mm256_set1_ps(center[d]), mdist);
							}
							const __m256 mpredist = _mm256_maskload_ps(distances + END, pmask);
							const __m256 mask = _mm256_cmp_ps(mdist, mpredist, _CMP_GT_OQ);
							_mm256_maskstore_ps(distances + END, pmask, _mm256_blendv_ps(mdist, mpredist, mask));

							const __m256i label_mask = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_cvtps_epi32(mask));
							const __m256i mprelabel = _mm256_maskload_epi32(labels + END, pmask);
							_mm256_maskstore_epi32(labels + END, pmask, _mm256_blendv_epi8(mprelabel, _mm256_set1_epi32(k), label_mask));
						}
					}
					_mm_free(mdp);
				}
			}
		}
	};

#pragma endregion

	double KMeans::clusteringSoA(cv::InputArray dataInput, int K, cv::InputOutputArray bestLabels, cv::TermCriteria criteria, int attempts, int flags, OutputArray dest_centroids, MeanFunction function, int loop)
	{
		Mat src = dataInput.getMat();
		const bool isrow = (src.rows == 1);
		const int N = max(src.cols, src.rows);//input data size
		const int dims = min(src.cols, src.rows) * src.channels();//input dimensions
		const int type = src.depth();

		//std::cout << "KMeans::clustering" << std::endl;
		//std::cout << "sigma" << sigma << std::endl;
		weightTableSize = (int)ceil(sqrt(signal_max * signal_max * dims));//for 3channel 255 max case 442=ceil(sqrt(3*255^2))
		//std::cout << "tableSize" << tableSize << std::endl;
		float* weight_table = (float*)_mm_malloc(sizeof(float) * weightTableSize, AVX_ALIGN);
		if (function == MeanFunction::GaussInv)
		{
			//cout << "MeanFunction::GaussInv sigma " << sigma << endl;

			for (int i = 0; i < weightTableSize; i++)
			{
				//weight_table[i] = 1.f;
				weight_table[i] = max(1.f - exp(i * i / (-2.f * sigma * sigma)), 0.001f);
				//weight_table[i] =Huber(i, sigma) + 0.001f;
				//weight_table[i] = i< sigma ? 0.001f: 1.f;
				//float n = 2.2f;
				//weight_table[i] = 1.f - exp(pow(i,n) / (-n * pow(sigma,n))) + 0.02f;
				//weight_table[i] = pow(i,sigma*0.1)+0.01;
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
				//w = exp(w / (-2.0 * sigma * sigma));
				//w = 1.0 - exp(sqrt(w) / (-1.0 * sigma));
			}
		}
		if (function == MeanFunction::LnNorm)
		{
			for (int i = 0; i < weightTableSize; i++)
			{
				weight_table[i] = (float)pow(i, min(sigma, 10.f)) + FLT_EPSILON;
			}
		}
		if (function == MeanFunction::Gauss)
		{
			for (int i = 0; i < weightTableSize; i++)
			{
				weight_table[i] = exp(i * i / (-2.f * sigma * sigma));
				//w = 1.0 - exp(pow(sqrt(w), n) / (-n * pow(sigma, n)));
			}
		}

		//AoS to SoA by using transpose
		Mat	src_t = (src.cols < src.rows) ? src.t() : src;

		attempts = std::max(attempts, 1);
		CV_Assert(src.dims <= 2 && type == CV_32F && K > 0);
		CV_CheckGE(N, K, "Number of clusters should be more than number of elements");

		//	data format
		//	Mat::Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP)
		//	data0.step = data0 byte(3*4byte(size_of_float)=12byte)
		//	Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

		// N x dims 32F
		Mat data_points(dims, N, CV_32F, src_t.ptr(), isrow ? N * sizeof(float) : static_cast<size_t>(src_t.step));

		bestLabels.create(N, 1, CV_32S, -1, true);//8U is better for small label cases
		Mat best_labels = bestLabels.getMat();

		if (flags & cv::KMEANS_USE_INITIAL_LABELS)// for KMEANS_USE_INITIAL_LABELS
		{
			CV_Assert((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous());

			best_labels.copyTo(labels_internal);
			for (int i = 0; i < N; i++)
			{
				CV_Assert((unsigned)labels_internal.at<int>(i) < (unsigned)K);
			}
		}
		else //alloc buffer
		{
			if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous()))
			{
				bestLabels.create(N, 1, CV_32S);
				best_labels = bestLabels.getMat();
			}
			labels_internal.create(best_labels.size(), best_labels.type());
		}

		int* labelsPtr = labels_internal.ptr<int>();

		Mat centroids(K, dims, type);//dims x K
		if ((flags & KMEANS_USE_INITIAL_LABELS) && (function == MeanFunction::Gauss || function == MeanFunction::GaussInv || function == MeanFunction::LnNorm))
		{
			dest_centroids.copyTo(centroids);
		}
		Mat old_centroids(K, dims, type);
		Mat temp(1, dims, type);

		cv::AutoBuffer<float, 64> centroid_weight(K);
		cv::AutoBuffer<int, 64> label_count(K);
		cv::AutoBuffer<float, 64> dists(N);//double->float
		RNG& rng = theRNG();

		if (criteria.type & TermCriteria::EPS) criteria.epsilon = std::max(criteria.epsilon, 0.0);
		else criteria.epsilon = FLT_EPSILON;

		criteria.epsilon *= criteria.epsilon;

		if (criteria.type & TermCriteria::COUNT) criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
		else criteria.maxCount = 100;

		if (K == 1)
		{
			attempts = 1;
			criteria.maxCount = 2;
		}

		float best_compactness = FLT_MAX;
		for (int attempt_index = 0; attempt_index < attempts; attempt_index++)
		{
			float compactness = 0.f;

			//main loop
			for (int iter = 0; ;)
			{
				float max_center_shift = (iter == 0) ? FLT_MAX : 0.f;

				swap(centroids, old_centroids);

				const bool isInit = ((iter == 0) && (attempt_index > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)));//initial attemp && KMEANS_USE_INITIAL_LABELS is true
				//print_debug(isInit);
				if (isInit)//initialization for first loop
				{
					//cout << "init" << endl;
					//cp::Timer t("generate sample"); //<1ns

					if (flags & KMEANS_PP_CENTERS)//kmean++
					{
						//cout << "kmeans++ SoA" << endl;
						if (isUseWeight) generateWeightedKmeansPPInitialCentroidSoA(data_points, weight, centroids, K, rng, KMEANSPP_TRIALS);
						else generateKmeansPPInitialCentroidSoA(data_points, centroids, K, rng, KMEANSPP_TRIALS);
					}
					else if (flags & KMEANS_MSPP_CENTERS)//kmean++
					{
						//cout << "kmeans ms++ SoA" << endl;
						generateKmeansMSPPInitialCentroidSoA(data_points, centroids, K, rng, KMEANSPP_TRIALS);
					}
					else //random initialization
					{
						generateKmeansRandomBoxInitialCentroidSoA(data_points, centroids, K, rng);
					}
					//cout << "init done" << endl;
				}
				else
				{
					//cout << "no init" << endl;
					//cp::Timer t("compute centroid"); //<1msD
					//update centroid 
					centroids.setTo(0.f);
					for (int k = 0; k < K; k++) label_count[k] = 0;

					//compute centroid without normalization; loop: N x d 
					if (function == MeanFunction::Mean)
					{
						boxMeanCentroidSoA(data_points, labelsPtr, centroids, label_count);
					}
					else if (function == MeanFunction::MinMax)
					{
						minmaxCentroidSoA(data_points, labelsPtr, centroids, label_count, K);
					}
					else if (function == MeanFunction::Gauss || function == MeanFunction::GaussInv || function == MeanFunction::LnNorm)
					{
						weightedMeanCentroid(data_points, labelsPtr, old_centroids, weight_table, weightTableSize, centroids, centroid_weight, label_count);
					}
					else if (function == MeanFunction::Harmonic)
					{
						harmonicMeanCentroid(data_points, labelsPtr, old_centroids, centroids, centroid_weight, label_count);
					}

					//processing for empty cluster
					//loop: N x K loop; but the most parts are skipped
					//if some cluster appeared to be empty then:
					//   1. find the biggest cluster
					//   2. find the farthest from the center point in the biggest cluster
					//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
	//#define DEBUG_SHOW_SKIP 
#ifdef DEBUG_SHOW_SKIP
					int count = 0; //for cout
#endif
					for (int k = 0; k < K; k++)
					{
						if (label_count[k] != 0) continue;

						//std::cout << "empty: " << k << std::endl;

						int k_count_max = 0;
						for (int k1 = 1; k1 < K; k1++)
						{
							if (label_count[k_count_max] < label_count[k1])
								k_count_max = k1;
						}

						float max_dist = 0.f;
						int farthest_i = -1;
						float* base_centroids = centroids.ptr<float>(k_count_max);
						float* normalized_centroids = temp.ptr<float>(); // normalized
						const float count_normalize = 1.f / label_count[k_count_max];
						for (int j = 0; j < dims; j++)
						{
							normalized_centroids[j] = base_centroids[j] * count_normalize;
						}
						for (int i = 0; i < N; i++)
						{
							if (labelsPtr[i] != k_count_max) continue;

#ifdef DEBUG_SHOW_SKIP
							count++; //for cout 
#endif
							float dist = 0.f;
							for (int d = 0; d < dims; d++)
							{
								dist += (data_points.ptr<float>(d)[i] - normalized_centroids[d]) * (data_points.ptr<float>(d)[i] - normalized_centroids[d]);
							}

							if (max_dist <= dist)
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						label_count[k_count_max]--;
						label_count[k]++;
						labelsPtr[farthest_i] = k;

						float* cur_center = centroids.ptr<float>(k);

						for (int d = 0; d < dims; d++)
						{
							base_centroids[d] -= data_points.ptr<float>(d)[farthest_i];
							cur_center[d] += data_points.ptr<float>(d)[farthest_i];
						}
					}
#ifdef DEBUG_SHOW_SKIP
					cout << iter << ": compute " << count / ((float)N * K) * 100.f << " %" << endl;
#endif

					//normalization and compute max shift distance between old centroid and new centroid
					//small loop: K x d
					for (int k = 0; k < K; k++)
					{
						float* centroidsPtr = centroids.ptr<float>(k);
						CV_Assert(label_count[k] != 0);

						float count_normalize = 0.f;
						if (function == MeanFunction::Mean)
						{
							count_normalize = 1.f / label_count[k];
						}
						else if (function == MeanFunction::MinMax)
						{
							count_normalize = 1.f;
						}
						else
						{
							count_normalize = 1.f / centroid_weight[k];//weighted mean
						}

						for (int d = 0; d < dims; d++) centroidsPtr[d] *= count_normalize;

						if (iter > 0)
						{
							float dist = 0.f;
							const float* old_center = old_centroids.ptr<float>(k);
							for (int d = 0; d < dims; d++)
							{
								float t = centroidsPtr[d] - old_center[d];
								dist += t * t;
							}
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				//compute distance and relabel
				//image size x dimensions x K (the most large loop)
				iter++;
				const bool isLastIter = (iter == 1) ? false : (iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);
				{
					//cp::Timer t(format("%d: distant computing", iter)); //last loop is fast
					if (isLastIter)
					{
						// compute distance only
						parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<true, KMeansDistanceLoop::KND>(dists.data(), labelsPtr, data_points, centroids), cv::getNumThreads());
						compactness = float(sum(Mat(Size(N, 1), CV_32F, &dists[0]))[0]);
						//getOuterSample(centroids, old_centroids, data_points, labels_internal);
						//swap(centroids, old_centroids);		
						break;
					}
					else
					{
						// assign labels
						//int parallel = CV_KMEANS_PARALLEL_GRANULARITY;
						int parallel = cv::getNumThreads();
						//int parallel = 1;//for debug
						//print_debug(dims);
						if (loop == KMeansDistanceLoop::KND)
						{
							switch (dims)
							{
							case 1: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 1>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 2: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 2>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 3: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 3>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 64:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::KND, 64>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							default:parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::KND>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							}
							//parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel);
						}
						else if (loop == KMeansDistanceLoop::NKD)
						{
							//cout << labels_internal << endl;
							switch (dims)
							{
							case 1: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 1>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 2: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 2>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 3: parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 3>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							case 64:parallel_for_(Range(0, N), KMeansDistanceComputer_SoA<false, KMeansDistanceLoop::NKD, 64>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							default:parallel_for_(Range(0, N), KMeansDistanceComputer_SoADim<false, KMeansDistanceLoop::NKD>(dists.data(), labelsPtr, data_points, centroids), parallel); break;
							}
						}
					}
				}
			}

			//reshape data structure for output			
			if (compactness < best_compactness)
			{
				best_compactness = compactness;
				if (dest_centroids.needed())
				{
					if (dest_centroids.fixedType() && dest_centroids.channels() == dims)
					{
						centroids.reshape(dims).copyTo(dest_centroids);
					}
					else
					{
						centroids.copyTo(dest_centroids);
					}
				}
				labels_internal.copyTo(best_labels);
			}
		}

		_mm_free(weight_table);
		return best_compactness;
	}
#pragma endregion

#pragma region AoS
	//static int CV_KMEANS_PARALLEL_GRANULARITY = (int)utils::getConfigurationParameterSizeT("OPENCV_KMEANS_PARALLEL_GRANULARITY", 1000);

#pragma region normL2Sqr
	inline float normL2Sqr_(const float* a, const float* b, const int avxend, const int issse, const int rem)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		__m256 msum = _mm256_mul_ps(v, v);
		for (int j = 0; j < avxend; j++)
		{
			__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
			msum = _mm256_fmadd_ps(v, v, msum);
			a += 8;
			b += 8;
		}
		float d = _mm256_reduceadd_ps(msum);
		if (issse)
		{
			__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
			v = _mm_mul_ps(v, v);
			d += _mm_reduceadd_ps(v);
			a += 4;
			b += 4;
		}
		for (int j = 0; j < rem; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}

		return d;
	}

	inline float normL2Sqr_(const float* a, const float* b, int n)
	{
		float d = 0.f;
		for (int j = 0; j < n; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}
		return d;
	}

	template<int n>
	inline float normL2Sqr_(const float* a, const float* b)
	{
		float d = 0.f;
		for (int j = 0; j < n; j++)
		{
			float t = a[j] - b[j];
			d += t * t;
		}
		return d;
	}

	template<>
	inline float normL2Sqr_<3>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);

		return v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2];
		//return _mm_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<4>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		return _mm_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<5>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t = a[4] - b[4];
		return _mm_reduceadd_ps(v) + t * t;
	}

	template<>
	inline float normL2Sqr_<6>(const float* a, const float* b)
	{
		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t1 = a[4] - b[4];
		const float t2 = a[5] - b[5];
		return _mm_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<7>(const float* a, const float* b)
	{
		/*__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		v = *(__m256*) &_mm256_insert_epi32(*(__m256i*) & v, 0, 7);
		return _mm256_reduceadd_ps(v);*/

		__m128 v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		v = _mm_mul_ps(v, v);
		const float t1 = a[4] - b[4];
		const float t2 = a[5] - b[5];
		const float t3 = a[6] - b[6];
		return _mm_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<8>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<9>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		return _mm256_reduceadd_ps(v) + t1 * t1;
	}

	template<>
	inline float normL2Sqr_<10>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		const float t2 = a[9] - b[9];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<11>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		const float t1 = a[8] - b[8];
		const float t2 = a[9] - b[9];
		const float t3 = a[10] - b[10];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<12>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m128 v2 = _mm_sub_ps(_mm_loadu_ps(a + 8), _mm_loadu_ps(b + 8));
		v2 = _mm_mul_ps(v2, v2);
		v = _mm256_add_ps(v, _mm256_castps128_ps256(v2));
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<16>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<24>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<32>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<40>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<41>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		return _mm256_reduceadd_ps(v) + t1 * t1;
	}

	template<>
	inline float normL2Sqr_<42>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		const float t2 = a[41] - b[41];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2;
	}

	template<>
	inline float normL2Sqr_<43>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		const float t1 = a[40] - b[40];
		const float t2 = a[41] - b[41];
		const float t3 = a[42] - b[42];
		return _mm256_reduceadd_ps(v) + t1 * t1 + t2 * t2 + t3 * t3;
	}

	template<>
	inline float normL2Sqr_<44>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);

		__m128 v3 = _mm_sub_ps(_mm_loadu_ps(a + 40), _mm_loadu_ps(b + 40));
		v3 = _mm_mul_ps(v3, v3);
		v = _mm256_add_ps(v, _mm256_castps128_ps256(v3));
		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<48>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<56>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

	template<>
	inline float normL2Sqr_<64>(const float* a, const float* b)
	{
		__m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		v = _mm256_mul_ps(v, v);
		__m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
		v = _mm256_fmadd_ps(v2, v2, v);
		v2 = _mm256_sub_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56));
		v = _mm256_fmadd_ps(v2, v2, v);

		return _mm256_reduceadd_ps(v);
	}

#pragma endregion

	void KMeans::generateKmeansRandomInitialCentroidAoS(const cv::Mat& data_points, Mat& dest_centroids, const int K, RNG& rng)
	{
		const int dims = data_points.cols;
		const int N = data_points.rows;
		cv::AutoBuffer<Vec2f, 64> box(dims);
		{
			const float* sample = data_points.ptr<float>(0);
			for (int j = 0; j < dims; j++)
				box[j] = Vec2f(sample[j], sample[j]);
		}
		for (int i = 1; i < N; i++)
		{
			const float* sample = data_points.ptr<float>(i);
			for (int j = 0; j < dims; j++)
			{
				float v = sample[j];
				box[j][0] = std::min(box[j][0], v);
				box[j][1] = std::max(box[j][1], v);
			}
		}

		const bool isUseMargin = false;//using margin is OpenCV's implementation
		if (isUseMargin)
		{
			const float margin = 1.f / dims;
			for (int k = 0; k < K; k++)
			{
				float* dptr = dest_centroids.ptr<float>(k);
				for (int d = 0; d < dims; d++)
				{
					dptr[d] = ((float)rng * (1.f + margin * 2.f) - margin) * (box[d][1] - box[d][0]) + box[d][0];
				}
			}
		}
		else
		{
			for (int k = 0; k < K; k++)
			{
				float* dptr = dest_centroids.ptr<float>(k);
				for (int d = 0; d < dims; d++)
				{
					dptr[d] = rng.uniform(box[d][0], box[d][1]);
				}
			}
		}

	}

	class KMeansPPDistanceComputerAoS : public ParallelLoopBody
	{
	public:
		KMeansPPDistanceComputerAoS(float* tdist2_, const Mat& data_, const float* dist_, int ci_) :
			tdist2(tdist2_), data(data_), dist(dist_), ci(ci_)
		{ }

		void operator()(const cv::Range& range) const CV_OVERRIDE
		{
			//CV_TRACE_FUNCTION();
			const int begin = range.start;
			const int end = range.end;
			const int dims = data.cols;

			for (int i = begin; i < end; i++)
			{
				tdist2[i] = std::min(normL2Sqr_(data.ptr<float>(i), data.ptr<float>(ci), dims), dist[i]);
			}
		}

	private:
		KMeansPPDistanceComputerAoS& operator=(const KMeansPPDistanceComputerAoS&); // = delete

		float* tdist2;
		const Mat& data;
		const float* dist;
		const int ci;
	};

	void KMeans::generateKmeansPPInitialCentroidAoS(const Mat& data, Mat& _out_centers, int K, RNG& rng, int trials)
	{
		//CV_TRACE_FUNCTION();
		const int dims = data.cols;
		const int N = data.rows;
		cv::AutoBuffer<int, 64> _centers(K);
		int* centers = &_centers[0];
		cv::AutoBuffer<float, 0> _dist(N * 3);
		float* dist = &_dist[0], * tdist = dist + N, * tdist2 = tdist + N;
		double sum0 = 0;

		centers[0] = (unsigned)rng % N;

		for (int i = 0; i < N; i++)
		{
			dist[i] = normL2Sqr_(data.ptr<float>(i), data.ptr<float>(centers[0]), dims);
			sum0 += dist[i];
		}

		for (int k = 1; k < K; k++)
		{
			double bestSum = DBL_MAX;
			int bestCenter = -1;

			for (int j = 0; j < trials; j++)
			{
				double p = (double)rng * sum0;
				int ci = 0;
				for (; ci < N - 1; ci++)
				{
					p -= dist[ci];
					if (p <= 0.0)
						break;
				}

				parallel_for_(Range(0, N),
					KMeansPPDistanceComputerAoS(tdist2, data, dist, ci),
					(double)divUp((size_t)(dims * N), CV_KMEANS_PARALLEL_GRANULARITY));
				double s = 0;
				for (int i = 0; i < N; i++)
				{
					s += tdist2[i];
				}

				if (s < bestSum)
				{
					bestSum = s;
					bestCenter = ci;
					std::swap(tdist, tdist2);
				}
			}
			if (bestCenter < 0)
				CV_Error(Error::StsNoConv, "kmeans: can't update cluster center (check input for huge or NaN values)");
			centers[k] = bestCenter;
			sum0 = bestSum;
			std::swap(dist, tdist);
		}

		for (int k = 0; k < K; k++)
		{
			const float* src = data.ptr<float>(centers[k]);
			float* dst = _out_centers.ptr<float>(k);
			for (int j = 0; j < dims; j++)
				dst[j] = src[j];
		}
	}


	template<bool onlyDistance, int loop>
	class KMeansDistanceComputerAoSDim : public ParallelLoopBody
	{
	public:
		KMeansDistanceComputerAoSDim(float* distances_,
			int* labels_,
			const Mat& data_,
			const Mat& centers_)
			: distances(distances_),
			labels(labels_),
			data(data_),
			centers(centers_)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			const int begin = range.start;
			const int end = range.end;
			const int K = centers.rows;
			const int dims = centers.cols;

			const int avxend = dims / 8;
			const int issse = (dims - avxend * 8) / 4;
			const int rem = dims - avxend * 8 - issse * 4;

			for (int i = begin; i < end; ++i)
			{
				const float* sample = data.ptr<float>(i);
				if (onlyDistance)
				{
					const float* center = centers.ptr<float>(labels[i]);
					distances[i] = normL2Sqr_(sample, center, dims);
					continue;
				}
				else
				{
					int k_best = 0;
					float min_dist = FLT_MAX;

					for (int k = 0; k < K; k++)
					{
						const float* center = centers.ptr<float>(k);
						const float dist = normL2Sqr_(sample, center, dims);
						//const float dist = normL2Sqr_(sample, center, avxend, issse, rem);

						if (min_dist > dist)
						{
							min_dist = dist;
							k_best = k;
						}
					}

					distances[i] = min_dist;
					labels[i] = k_best;
				}
			}
		}

	private:
		KMeansDistanceComputerAoSDim& operator=(const KMeansDistanceComputerAoSDim&); // = delete

		float* distances;
		int* labels;
		const Mat& data;
		const Mat& centers;
	};

	template<bool onlyDistance, int dims, int loop>
	class KMeansDistanceComputerAoS : public ParallelLoopBody
	{
	public:
		KMeansDistanceComputerAoS(float* distances_, int* labels_, const Mat& data_, const Mat& centers_)
			: distances(distances_), labels(labels_), data(data_), centers(centers_)
		{
		}

		void operator()(const Range& range) const CV_OVERRIDE
		{
			const int begin = range.start;
			const int end = range.end;
			const int K = centers.rows;
			//n-k-d
			if (onlyDistance)
			{
				for (int n = begin; n < end; ++n)
				{
					const float* sample = data.ptr<float>(n);
					{
						const float* center = centers.ptr<float>(labels[n]);
						distances[n] = normL2Sqr_<dims>(sample, center);
						continue;
					}
				}
			}
			else
			{
				if (loop == KMeansDistanceLoop::NKD)
				{
					for (int n = begin; n < end; ++n)
					{
						const float* sample = data.ptr<float>(n);
						int k_best = 0;
						float min_dist = FLT_MAX;

						for (int k = 0; k < K; ++k)
						{
							const float* center = centers.ptr<float>(k);
							const float dist = normL2Sqr_<dims>(sample, center);

							if (min_dist > dist)
							{
								min_dist = dist;
								k_best = k;
							}
						}

						distances[n] = min_dist;
						labels[n] = k_best;
					}
				}
				else //k-n-d
				{
					{
						//int k = 0;
						const float* center = centers.ptr<float>(0);
						for (int n = begin; n < end; ++n)
						{
							const float* sample = data.ptr<float>(n);
							distances[n] = normL2Sqr_<dims>(sample, center);
							labels[n] = 0;
						}
					}
					for (int k = 1; k < K; ++k)
					{
						const float* center = centers.ptr<float>(k);
						for (int n = begin; n < end; ++n)
						{
							const float* sample = data.ptr<float>(n);
							const float dist = normL2Sqr_<dims>(sample, center);

							if (distances[n] > dist)
							{
								distances[n] = dist;
								labels[n] = k;
							}
						}
					}
				}
			}
		}

	private:
		KMeansDistanceComputerAoS& operator=(const KMeansDistanceComputerAoS&); // = delete

		float* distances;
		int* labels;
		const Mat& data;
		const Mat& centers;
	};

	void KMeans::boxMeanCentroidAoS(Mat& data_points, const int* labels, Mat& centroids, int* counters)
	{
		const int N = data_points.rows;
		const int dims = data_points.cols;

		for (int i = 0; i < N; i++)
		{
			const float* sample = data_points.ptr<float>(i);
			int k = labels[i];
			float* center = centroids.ptr<float>(k);
			for (int j = 0; j < dims; j++)
			{
				center[j] += sample[j];
			}
			counters[k]++;
		}
	}

	double KMeans::clusteringAoS(InputArray dataInput, int K, InputOutputArray bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, int loop)
	{
		const int SPP_TRIALS = 3;
		Mat src = dataInput.getMat();
		const bool isrow = (src.rows == 1);
		const int N = isrow ? src.cols : src.rows;
		const int dims = (isrow ? 1 : src.cols) * src.channels();
		const int type = src.depth();

		attempts = std::max(attempts, 1);
		CV_Assert(src.dims <= 2 && type == CV_32F && K > 0);
		CV_CheckGE(N, K, "Number of clusters should be more than number of elements");

		// dims x N 32F
		Mat data_points(N, dims, CV_32F, src.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(src.step));

		bestLabels.create(N, 1, CV_32S, -1, true);//8U is better for small label cases
		Mat best_labels = bestLabels.getMat();

		if (flags & cv::KMEANS_USE_INITIAL_LABELS)
		{
			CV_Assert((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous());
			best_labels.reshape(1, N).copyTo(labels_internal);
			for (int i = 0; i < N; i++)
			{
				CV_Assert((unsigned)labels_internal.at<int>(i) < (unsigned)K);
			}
		}
		else
		{
			if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols * best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous()))
			{
				bestLabels.create(N, 1, CV_32S);
				best_labels = bestLabels.getMat();
			}
			labels_internal.create(best_labels.size(), best_labels.type());
		}
		int* labels = labels_internal.ptr<int>();

		Mat centroids(K, dims, type);//dims x k
		Mat old_centroids(K, dims, type);
		Mat	temp(1, dims, type);
		cv::AutoBuffer<int, 64> counters(K);
		cv::AutoBuffer<float, 64> dists(N);
		//dists.resize(N);
		RNG& rng = theRNG();

		if (criteria.type & TermCriteria::EPS) criteria.epsilon = std::max(criteria.epsilon, 0.);
		else criteria.epsilon = FLT_EPSILON;

		criteria.epsilon *= criteria.epsilon;

		if (criteria.type & TermCriteria::COUNT) criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
		else criteria.maxCount = 100;

		if (K == 1)
		{
			attempts = 1;
			criteria.maxCount = 2;
		}

		double best_compactness = DBL_MAX;
		for (int attempt_index = 0; attempt_index < attempts; attempt_index++)
		{
			double compactness = 0.0;

			//main loop
			for (int iter = 0; ;)
			{
				float max_center_shift = (iter == 0) ? FLT_MAX : 0.f;

				swap(centroids, old_centroids);

				const bool isInit = ((iter == 0) && (attempt_index > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)));//initial attemp && KMEANS_USE_INITIAL_LABELS is true
				//print_debug(isInit);
				if (isInit)
				{
					if (flags & KMEANS_PP_CENTERS)
					{
						//cout << "kmeans++ AoS" << endl;
						generateKmeansPPInitialCentroidAoS(data_points, centroids, K, rng, SPP_TRIALS);
					}
					else
					{
						generateKmeansRandomInitialCentroidAoS(data_points, centroids, K, rng);
					}
				}
				else
				{
					//update centroid 
					centroids = Scalar(0.f);
					for (int k = 0; k < K; k++) counters[k] = 0;
					if (function == MeanFunction::Harmonic)
					{
						cout << "MeanFunction::Harmonic not support" << endl;
					}
					else if (function == MeanFunction::Gauss || function == MeanFunction::GaussInv)
					{
						cout << "MeanFunction::Gauss/MeanFunction::GaussInv not support" << endl;
					}
					else if (function == MeanFunction::Mean)
					{
						boxMeanCentroidAoS(data_points, labels, centroids, counters);
					}

					for (int k = 0; k < K; k++)
					{
						if (counters[k] != 0)
							continue;

						// if some cluster appeared to be empty then:
						//   1. find the biggest cluster
						//   2. find the farthest from the center point in the biggest cluster
						//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
						int k_count_max = 0;
						for (int k1 = 1; k1 < K; k1++)
						{
							if (counters[k_count_max] < counters[k1])
								k_count_max = k1;
						}

						double max_dist = 0;
						int farthest_i = -1;
						float* base_center = centroids.ptr<float>(k_count_max);
						float* _base_center = temp.ptr<float>(); // normalized
						float scale = 1.f / counters[k_count_max];
						for (int j = 0; j < dims; j++)
							_base_center[j] = base_center[j] * scale;

						for (int i = 0; i < N; i++)
						{
							if (labels[i] != k_count_max)
								continue;
							const float* sample = data_points.ptr<float>(i);
							double dist = normL2Sqr_(sample, _base_center, dims);

							if (max_dist <= dist)
							{
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[k_count_max]--;
						counters[k]++;
						labels[farthest_i] = k;

						const float* sample = data_points.ptr<float>(farthest_i);
						float* cur_center = centroids.ptr<float>(k);
						for (int j = 0; j < dims; j++)
						{
							base_center[j] -= sample[j];
							cur_center[j] += sample[j];
						}
					}

					for (int k = 0; k < K; k++)
					{
						float* center = centroids.ptr<float>(k);
						CV_Assert(counters[k] != 0);

						float scale = 1.f / counters[k];
						for (int j = 0; j < dims; j++)
							center[j] *= scale;

						if (iter > 0)
						{
							float dist = 0.f;
							const float* old_center = old_centroids.ptr<float>(k);
							for (int j = 0; j < dims; j++)
							{
								float t = center[j] - old_center[j];
								dist += t * t;
							}
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);

				if (isLastIter)
				{
					//int parallel = (double)divUp((size_t)(dims * N * K), CV_KMEANS_PARALLEL_GRANULARITY);
					int parallel = cv::getNumThreads();
					// don't re-assign labels to avoid creation of empty clusters
					parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<true, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel);
					compactness = sum(Mat(Size(N, 1), CV_32F, &dists[0]))[0];
					break;
				}
				else
				{
					//int parallel = (double)divUp((size_t)(dims * N * K), CV_KMEANS_PARALLEL_GRANULARITY);
					int parallel = cv::getNumThreads();
					// assign labels
					if (loop == KMeansDistanceLoop::NKD)
					{
						switch (dims)
						{
						case 1:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 1, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 2:  parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 2, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 3:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 3, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 4:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 4, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 5:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 5, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 6:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 6, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 7:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 7, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 8:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 8, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 9:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 9, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 10: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 10, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 11: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 11, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 12: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 12, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 16: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 16, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 24: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 24, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 32: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 32, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 40: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 40, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 41: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 41, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 42: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 42, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 43: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 43, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						case 44: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 44, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 48: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 48, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 56: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 56, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;

						case 64: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 64, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						default: parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<false, KMeansDistanceLoop::NKD>(dists.data(), labels, data_points, centroids), parallel); break;
						}
					}
					else
					{
						switch (dims)
						{
						case 1:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 1, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 2:  parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 2, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 3:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 3, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 4:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 4, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 5:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 5, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 6:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 6, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 7:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 7, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 8:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 8, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 9:	 parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 9, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 10: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 10, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 11: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 11, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 12: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 12, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 16: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 16, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 24: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 24, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 32: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 32, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 40: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 40, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 41: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 41, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 42: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 42, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 43: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 43, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						case 44: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 44, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 48: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 48, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 56: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 56, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;

						case 64: parallel_for_(Range(0, N), KMeansDistanceComputerAoS<false, 64, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						default: parallel_for_(Range(0, N), KMeansDistanceComputerAoSDim<false, KMeansDistanceLoop::KND>(dists.data(), labels, data_points, centroids), parallel); break;
						}
					}
				}
			}

			if (compactness < best_compactness)
			{
				best_compactness = compactness;
				if (_centers.needed())
				{
					if (_centers.fixedType() && _centers.channels() == dims)
						centroids.reshape(dims).copyTo(_centers);
					else
						centroids.copyTo(_centers);
				}
				labels_internal.copyTo(best_labels);
			}
		}

		return best_compactness;
	}
#pragma endregion

#pragma region SoAoS
	double KMeans::clusteringSoAoS(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers, MeanFunction function, int loop)
	{
		cout << "not implemented clusteringSoAoS" << endl;
		return 0.0;
	}
#pragma endregion
}