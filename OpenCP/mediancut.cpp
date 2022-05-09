#include "mediancut.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	struct box
	{
		uchar aucElement[3];
		uchar iTargetElement; //deviding channel 
		uchar iDifference;	//difference between min and max value in deviding channel 
		uchar iMedian;		//median value in deviding channel 
		uchar iMax;			//max value in deviding channel
		int iTop;			//top index
		int iBottom;		//bottom index
	};

	template<int NUM_INTENSITY, int channels>
	static void CalcStats(box* box_ptr, Mat& src_image, int* indices_ptr, const MedianCutMethod cmethod)
	{
		int iHist[NUM_INTENSITY];//histogram
		int iHalf = (box_ptr->iBottom - box_ptr->iTop) >> 1;  // the half number of box pixels
		box_ptr->iDifference = 0;


		// computing histogram, min and max values
		for (int i = 0; i < channels; i++)
		{
			memset(iHist, 0, sizeof(iHist));
			uchar iMin = NUM_INTENSITY - 1;
			uchar iMax = 0;
			for (int j = box_ptr->iTop; j <= box_ptr->iBottom; j++)
			{
				int index = indices_ptr[j];
				Vec3b* bgr = src_image.ptr<Vec3b>(index);
				iHist[bgr[0][i]]++;

				iMin = min(iMin, bgr[0][i]);
				iMax = max(iMax, bgr[0][i]);
			}

			// computing median
			int iMedian = 0;
			int iSum = 0;
			for (int j = 0; j < NUM_INTENSITY; j++)
			{
				iSum += iHist[j];
				if (iSum > iHalf)
				{
					iMedian = j;
					break;
				}
			}

			// get representing color
			switch (cmethod)
			{
			case MedianCutMethod::MEDIAN:
				box_ptr->aucElement[i] = (uchar)iMedian; break;
			case MedianCutMethod::MAX:
				box_ptr->aucElement[i] = (uchar)iMax; break;
			case MedianCutMethod::MIN:
				box_ptr->aucElement[i] = (uchar)iMin; break;
			default: break;
			}

			// the largest difference channel as a dividing target
			const int iDifference = iMax - iMin;
			if (iDifference > box_ptr->iDifference)
			{
				box_ptr->iTargetElement = i;
				box_ptr->iMedian = iMedian;
				box_ptr->iMax = iMax;
				box_ptr->iDifference = iDifference;
			}
		}
	}

	static void divideBox(box* new_box, box* old_box, const Mat& src_image, int* indices_ptr)
	{
		int iTop = old_box->iTop;
		int iBottom = old_box->iBottom;
		int iTarget = old_box->iTargetElement;
		int iMedian = old_box->iMedian;
		if (iMedian == old_box->iMax) iMedian--;

		while (iTop < iBottom)
		{
			int iIndex = indices_ptr[iTop];
			const Vec3b* bgr = src_image.ptr<const Vec3b>(iIndex);

			if (bgr[0][iTarget] > iMedian)
			{
				indices_ptr[iTop] = indices_ptr[iBottom];
				indices_ptr[iBottom] = iIndex;
				iBottom--;
			}
			else iTop++;
		}

		new_box->iTop = iBottom + 1;
		new_box->iBottom = old_box->iBottom;
		old_box->iBottom = iBottom;
	}

	void mediancut(InputArray src, const int K, OutputArray destLabels, OutputArray destColor, const MedianCutMethod cmethod)
	{
		const Mat& input_image = src.getMat();
		Mat	reshaped_input_image;
		int width, height, pixel_num;
		if (input_image.channels() == 1 && input_image.rows == 3)
		{
			pixel_num = input_image.cols;
			//print_matinfo(reshaped_input_image);
			reshaped_input_image = input_image.reshape(3, pixel_num);
		}
		else
		{
			width = input_image.cols;
			height = input_image.rows;
			pixel_num = width * height;

			if (input_image.cols != 3)
				reshaped_input_image = input_image.reshape(3, pixel_num);
			else
				reshaped_input_image = input_image;
		}

		Mat labels = destLabels.getMat();
		Mat centers = destColor.getMat();
		centers.create(Size(1, K), CV_8UC3);
		labels.create(Size(1, pixel_num), CV_8UC1);

		int* indices_ptr = new int[pixel_num];
		for (int i = 0; i < pixel_num; i++)
		{
			indices_ptr[i] = i;
		}

		box* boxes = new box[K];
		memset(boxes, 0, sizeof(box) * K);

		// information for the first box
		int count_box = 0;
		boxes[count_box].iTop = 0;
		boxes[count_box].iBottom = pixel_num - 1;
		// calcstat
		CalcStats<256, 3>(&boxes[count_box], reshaped_input_image, indices_ptr, cmethod);
		count_box++;

		while (count_box < K)
		{
			int iMax = -1;
			int iIndexMaxDifference = 0;   // get the largest differnece index for a box
			for (int i = 0; i < count_box; i++)
			{
				if (boxes[i].iDifference > iMax)
				{
					iMax = boxes[i].iDifference;
					iIndexMaxDifference = i;
				}
			}

			// divide as median value
			divideBox(&boxes[count_box], &boxes[iIndexMaxDifference], reshaped_input_image, indices_ptr);

			CalcStats<256, 3>(&boxes[count_box], reshaped_input_image, indices_ptr, cmethod);
			CalcStats<256, 3>(&boxes[iIndexMaxDifference], reshaped_input_image, indices_ptr, cmethod);

			count_box++;
		}

		// storing index data and representing values
		for (int i = 0; i < K; i++)
		{
			Vec3b* centers_ptr = centers.ptr<Vec3b>(i);
			centers_ptr[0][0] = boxes[i].aucElement[0];
			centers_ptr[0][1] = boxes[i].aucElement[1];
			centers_ptr[0][2] = boxes[i].aucElement[2];
			//std::cout << centers_ptr[0] << std::endl;

			for (int j = boxes[i].iTop; j <= boxes[i].iBottom; j++)
			{
				uchar* labels_ptr = labels.ptr<uchar>(indices_ptr[j]);
				labels_ptr[0] = (uchar)i;
			}
		}

		delete[] indices_ptr;
		delete[] boxes;
	}
}