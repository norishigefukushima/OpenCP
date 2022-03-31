#include "poissonDiskSampling.hpp"
#include "randomizedQueue.hpp"

using namespace cv;
using namespace std;

namespace cp
{
	void PoissonDiskSampling::set(const cv::Point pt)
	{
		cv::Point gpt = imageToGrid(pt);
		background_grid_pt.at<cv::Vec2i>(gpt) = cv::Vec2i(pt);
	}

	cv::Point PoissonDiskSampling::imageToGrid(cv::Point pt)
	{
		return cv::Point((int)(pt.x * cell_size_inv), (int)(pt.y * cell_size_inv));
		//return cv::Point((cvRound)(pt.x / cellsize), (cvRound)(pt.y / cellsize));
	}

	float PoissonDiskSampling::getDistance(const cv::Point pt1, const cv::Point pt2)
	{
		const int dx = (pt1.x - pt2.x);
		const int dy = (pt1.y - pt2.y);
		return (float)sqrt((dx) * (dx)+(dy) * (dy));
		//return float(dx * dx + dy * dy);
	}

	cv::Point PoissonDiskSampling::generateRandomPointAround(const cv::Point pt, cv::RNG& rng)
	{
		return pt + pointAroundCandidate[rng.uniform(0, (int)pointAroundCandidate.size())];
		/*
		float theta = rng.uniform(0.f, (float)CV_2PI);
		float rad = rng.uniform(min_d, 2.f * min_d);

		int x = cvRound(pt.x + rad * cos(theta));
		int y = cvRound(pt.y + rad * sin(theta));

		return Point(x,y);
		*/
	}

	bool PoissonDiskSampling::isAvailable(const cv::Point pt)
	{
		if (inImage(pt))
		{
			return inNeibourhood(pt);
		}

		return false;
	}

	cv::Point PoissonDiskSampling::initializeStart(cv::RNG& rng)
	{
		return cv::Point(rng.uniform(0, imageSize.width), rng.uniform(0, imageSize.height));
	}


	inline bool PoissonDiskSampling::inImage(const cv::Point pt)
	{
		if (pt.x < 0 || pt.x >= imageSize.width) return false;
		if (pt.y < 0 || pt.y >= imageSize.height) return false;

		return true;
	}
	inline bool PoissonDiskSampling::inNeibourhood(const cv::Point pt)
	{
		cv::Point grid_pt = imageToGrid(pt);

#if 1
		const int bb = 2;
		const int sty = (-bb + grid_pt.y < 0) ? -grid_pt.y : -bb;
		const int edy = (+bb + grid_pt.y >= grid_height) ? (grid_height - 1) - grid_pt.y : bb;
		const int stx = (-bb + grid_pt.x < 0) ? -grid_pt.x : -bb;
		const int edx = (+bb + grid_pt.x >= grid_width) ? (grid_width - 1) - grid_pt.x : bb;

		for (int y = sty; y <= edy; y++)
		{
			for (int x = stx; x <= edx; x++)
			{
				cv::Point background_pt = background_grid_pt.at<cv::Vec2i>(cv::Point(x + grid_pt.x, y + grid_pt.y));
				if (background_pt.x >= 0)
				{
					if (getDistance(background_pt, pt) < min_d) return false;
				}
			}
		}
#else
		for (int y = -2; y <= 2; y++)
		{
			if (y + grid_pt.y >= 0 && y + grid_pt.y < grid_height)
			{
				for (int x = -2; x <= 2; x++)
				{
					if (x + grid_pt.x >= 0 && x + grid_pt.x < grid_width)
					{
						Point ppt = Point(x + grid_pt.x, y + grid_pt.y);
						Point bpt = background_pt.at<Vec2i>(ppt);
						if (bpt.x >= 0)
						{
							if (calcDistance(bpt, pt) < min_d) return false;
						}
					}
				}
			}
		}
#endif

		return true;
	}

	PoissonDiskSampling::PoissonDiskSampling(const float min_d, const cv::Size imageSize)
	{
		this->min_d = min_d;
		this->min_d2 = min_d * min_d;
		const float cell_size = min_d / sqrt(2.f);
		this->cell_size_inv = 1.f / (cell_size);

		grid_width = (int)floor(imageSize.width * cell_size_inv) + 1;
		grid_height = (int)floor(imageSize.height * cell_size_inv) + 1;
		//grid_width = (int)ceil(imageSize.width / cellsize) + 1;
		//grid_height = (int)ceil(imageSize.height / cellsize) + 1;

		background_grid_pt.create(grid_height, grid_width, CV_32SC2);
		this->imageSize = imageSize;

		int rmax = cvRound(2 * min_d);
		//int rmin = cvRound(min_d);
		//float rmax = 2.f * min_d;
		float rmin = min_d;

		for (int j = -rmax; j <= rmax; j++)
		{
			for (int i = -rmax; i <= rmax; i++)
			{
				//int dist = cvRound(sqrt(i * i + j * j));
				float dist = (float)sqrt(i * i + j * j);

				if (rmin <= dist && dist <= rmax)
					pointAroundCandidate.push_back(cv::Point(i, j));
			}
		}
	}

	int PoissonDiskSampling::generate(cv::Mat& mask, cv::RNG& rng, cv::Point start, const int max_sample)
	{
		//cp::Timer t;
		mask.create(imageSize, CV_8U);
		mask.setTo(0);
		background_grid_pt.setTo(-1);

		const cv::Size kernelSize = mask.size();

		cp::RandomizedQueue proc(rng.state);
		//Queue proc(kernelSize.area()+k);

		if (start.x == -1)
		{
			start = initializeStart(rng);
		}

		proc.push(start);

		mask.at<uchar>(start) = 255;
		int ret = 1;
		set(start);

		if (max_sample < 0)
		{
			while (!proc.empty())
			{
				cv::Point pt = proc.pop();

				for (int i = 0; i < k; i++)
				{
					cv::Point newpt = generateRandomPointAround(pt, rng);
					if (isAvailable(newpt))
					{
						proc.push(newpt);
						mask.at<uchar>(newpt) = 255;
						ret++;
						set(newpt);
					}
				}
			}
		}
		else
		{
			while (!proc.empty())
			{
				cv::Point pt = proc.pop();
				for (int i = 0; i < k; i++)
				{
					cv::Point newpt = generateRandomPointAround(pt, rng);
					if (isAvailable(newpt))
					{
						proc.push(newpt);
						mask.at<uchar>(newpt) = 255;
						ret++;
						if (ret == max_sample)return ret;
						set(newpt);
					}
				}
			}
		}

		return ret;
	}
}