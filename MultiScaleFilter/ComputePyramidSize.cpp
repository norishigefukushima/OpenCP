#include "multiscalefilter/MultiScaleFilter.hpp"

using namespace cv;
using namespace std;
namespace cp
{
	ComputePyramidSize::ComputePyramidSize(int size, int r, int level, bool isLastConv) :size(size), r(r), level(level), isLastConv(isLastConv)
	{
		borderSizeR.resize(level);
		borderSizeL.resize(level);
		imageSize.resize(level);
		borderSizeActive.resize(level);
		borderSizeRequire.resize(level);
		borderOffset.resize(level);
		computeSize(size, r, level, isLastConv);
	}

	void ComputePyramidSize::computeSize(int size, int r, int level, bool isLastConv)
	{
		if (isLastConv)
		{
			int l = level - 1;
			imageSize[l] = size / pow(2, l);
			borderSizeR[l] = imageSize[l] + r;
			borderSizeL[l] = imageSize[l] + r + 1;
			borderSizeActive[l] = (borderSizeR[l] - imageSize[l]) + (borderSizeL[l] - imageSize[l] - 1) + imageSize[l];
			for (int l = level - 2; l >= 0; l--)
			{
				imageSize[l] = size / pow(2, l);
				borderSizeR[l] = borderSizeR[l + 1] * 2 - 1 + r;
				borderSizeL[l] = borderSizeL[l + 1] * 2 - 1 + r;
				borderSizeActive[l] = (borderSizeR[l] - imageSize[l]) + (borderSizeL[l] - imageSize[l] - 1) + imageSize[l];
			}
		}
		else
		{
			int l = level - 1;
			imageSize[l] = size / pow(2, l);
			borderSizeR[l] = imageSize[l];
			borderSizeL[l] = imageSize[l] + 1;
			borderSizeActive[l] = imageSize[l];

			l = level - 2;
			imageSize[l] = size / pow(2, l);
			borderSizeR[l] = imageSize[l] + r;
			borderSizeL[l] = imageSize[l] + r + 1;
			borderSizeActive[l] = (borderSizeR[l] - imageSize[l]) + (borderSizeL[l] - imageSize[l] - 1) + imageSize[l];
			for (int l = level - 3; l >= 0; l--)
			{
				imageSize[l] = size / pow(2, l);
				borderSizeR[l] = borderSizeR[l + 1] * 2 - 1 + r;
				borderSizeL[l] = borderSizeL[l + 1] * 2 - 1 + r;
				borderSizeActive[l] = (borderSizeR[l] - imageSize[l]) + (borderSizeL[l] - imageSize[l] - 1) + imageSize[l];
			}
		}

		for (int l = 0; l < level; l++)
		{
			borderSizeR[l] -= imageSize[l];
			borderSizeL[l] -= imageSize[l] + 1;
		}
		//minsize = pow(2, level - 1);
		minsize = pow(2, level);
		borderSizeRequire[0] = get_simd_ceil(borderSizeL[0], minsize) + imageSize[0] + get_simd_ceil(borderSizeR[0], minsize);
		borderOffset[0] = get_simd_ceil(borderSizeL[0], minsize);
		for (int l = 1; l < level; l++)
		{
			borderSizeRequire[l] = borderSizeRequire[l - 1] / 2;
			borderOffset[l] = borderOffset[l - 1] / 2;
		}
	}

	void ComputePyramidSize::print()
	{
		for (int l = 0; l < level; l++)
		{
			cout << "level: " << l << endl;
			cout << "borderSizeL       " << borderSizeL[l] << endl;
			cout << "imageSize         " << imageSize[l] << endl;
			cout << "borderSizeR       " << borderSizeR[l] << endl;
			cout << "borderSizeActual  " << borderSizeActive[l] << endl;
			cout << "borderSizeRequire " << borderSizeRequire[l] << endl;
			cout << "borderOffset      " << borderOffset[l] << endl;
		}
	}
	Mat ComputePyramidSize::vizPyramid(int bs)
	{
		const int lw = 2;

		//print_debug2(pysizeL[0], pysizeR[0]);
		Mat image = Mat::zeros(bs * (level + 2), bs * (borderSizeRequire[0]), CV_8UC3);
#pragma region setBackGround
		for (int i = 0; i < image.cols / bs; i++)
		{
			if (i % 2 == 0) rectangle(image, Rect(i * bs, 0, bs, image.rows), COLOR_GRAY100, cv::FILLED);
			if (i % 2 == 1) rectangle(image, Rect(i * bs, 0, bs, image.rows), COLOR_GRAY50, cv::FILLED);
			if (i % minsize == 0) rectangle(image, Rect(i * bs, 0, bs, image.rows), COLOR_GRAY150, cv::FILLED);
		}
		for (int i = 0; i < image.cols / bs; i++)
		{
			for (int j = 0; j < image.rows / bs; j++)
			{
				rectangle(image, Rect(i * bs, j * bs, bs, bs), COLOR_GRAY150, 1);
			}
		}
#pragma endregion

		const int offx = borderOffset[0];
		const int offy = 1;
		for (int l = level - 1; l >= 0; l--)
		{
			const int h = pow(2, l);
			for (int i = 0; i < image.cols / bs; i += (int)pow(2, l))
			{
				rectangle(image, Rect(i * bs, (l + offy) * bs, bs, bs), COLOR_CYAN * 0.5, lw, LineTypes::LINE_8);
			}
			for (int i = 0; i < imageSize[l]; i++)
			{
				rectangle(image, Rect(((i * h) + offx) * bs, (l + offy) * bs, bs, bs), COLOR_GREEN, lw, LineTypes::LINE_8);
			}

			Scalar color = (l == level - 1) ? COLOR_RED : COLOR_YELLOW;
			for (int i = imageSize[l]; i < imageSize[l] + borderSizeR[l]; i++)
			{
				rectangle(image, Rect(((i * h) + offx) * bs, (l + offy) * bs, bs, bs), color, lw, LineTypes::LINE_8);
			}

			if (borderSizeL[l] != 0)
			{
				for (int i = -1; i >= -borderSizeL[l]; i--)
				{
					rectangle(image, Rect(((i * h) + offx) * bs, (l + offy) * bs, bs, bs), color, lw, LineTypes::LINE_8);
				}
			}
		}
		return image;
	}

	void ComputePyramidSize::get(const int level, int& borderSizeL, int& borderSizeR, int& imageSize, int& borderSizeActive, int& borderSizeRequre, int& borderOffset)
	{
		borderSizeR = this->borderSizeR[level];
		borderSizeL = this->borderSizeL[level];
		imageSize = this->imageSize[level];
		borderSizeActive = this->borderSizeActive[level];
		borderSizeRequre = this->borderSizeRequire[level];
		borderOffset = this->borderOffset[level];
	}	

	int ComputePyramidSize::getBorderL(int level)
	{
		return this->borderSizeL[level];
	}

	int ComputePyramidSize::getBorderR(int level)
	{
		return this->borderSizeR[level];
	}

	int ComputePyramidSize::getBorderOffset(int level)
	{
		return this->borderOffset[level];
	}

	int ComputePyramidSize::getBorderSizeRequire(int level)
	{
		return this->borderSizeRequire[level];
	}
}