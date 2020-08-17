#include "contrast.hpp"
#include "updateCheck.hpp"
using namespace std;
using namespace cv;

namespace cp
{
	cv::Mat convert(cv::Mat& src, const int depth, const double alpha, const double beta)
	{
		cv::Mat ret;
		src.convertTo(ret, depth, alpha, beta);
		return ret;
	}

	cv::Mat cenvertCentering(cv::InputArray src, int depth, double a, double b)
	{
		Mat ret;
		src.getMat().convertTo(ret, depth, a, -a * b + b);
		return ret;
	}

	template<typename T>
	void contrastSToneExp_(T* src, T* dest, const int size, const double sigma, const double a, const double b)
	{
		const double coeff = -0.5 / (sigma * sigma);
		for (int i = 0; i < size; i++)
		{
			double v = double(src[i] - b);
			dest[i] = saturate_cast<T>(double(src[i]) + a * exp(v * v * coeff) * v);
		}
	}

	//x- a*gauss(x-b)(x-b)
	void contrastSToneExp(cv::InputArray src, cv::OutputArray dest, const double sigma, const double a, const double b)
	{
		dest.create(src.size(), src.type());
		const int size = src.getMat().size().area() * src.getMat().channels();

		switch (src.depth())
		{
		case CV_8U:contrastSToneExp_<uchar>(src.getMat().ptr<uchar>(), dest.getMat().ptr<uchar>(), size, sigma, a, b); break;
		case CV_8S:contrastSToneExp_<char>(src.getMat().ptr<char>(), dest.getMat().ptr<char>(), size, sigma, a, b); break;
		case CV_16U:contrastSToneExp_<ushort>(src.getMat().ptr<ushort>(), dest.getMat().ptr<ushort>(), size, sigma, a, b); break;
		case CV_16S:contrastSToneExp_< short>(src.getMat().ptr< short>(), dest.getMat().ptr< short>(), size, sigma, a, b); break;
		case CV_32S:contrastSToneExp_<   int>(src.getMat().ptr<   int>(), dest.getMat().ptr<   int>(), size, sigma, a, b); break;
		case CV_32F:contrastSToneExp_< float>(src.getMat().ptr< float>(), dest.getMat().ptr< float>(), size, sigma, a, b); break;
		case CV_64F:contrastSToneExp_<double>(src.getMat().ptr<double>(), dest.getMat().ptr<double>(), size, sigma, a, b); break;
		}
	}

	cv::Mat guiContrast(InputArray src_, string wname)
	{
		Mat src = src_.getMat();
		namedWindow(wname);
		static int a_gui_contrast = 10;
		static int b_gui_contrast = 0;
		static int sw_gui_contrast = 0;
		static int sigma_gui_contrast = 30;
		cv::createTrackbar("sw", wname, &sw_gui_contrast, 2);
		cv::createTrackbar("a*0.1", wname, &a_gui_contrast, 100);
		cv::createTrackbar("b", wname, &b_gui_contrast, 255);
		cv::createTrackbar("sigma", wname, &sigma_gui_contrast, 255);

		int key = 0;
		cp::UpdateCheck uc(sw_gui_contrast);
		cv::Mat show;
		while (key != 'q')
		{
			if (uc.isUpdate(sw_gui_contrast))
			{
				string mes = "";
				if (sw_gui_contrast == 0)mes = "ax + b (convertTo)";
				if (sw_gui_contrast == 1)mes = "a(x-b) + b";
				if (sw_gui_contrast == 2)mes = "x - a*gauss(x-b)(x-b)";
				displayOverlay(wname, mes, 3000);
			}

			if (sw_gui_contrast == 0)
			{
				src.convertTo(show, CV_8U, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			else if (sw_gui_contrast == 1)
			{
				show = cenvertCentering(src, CV_8U, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			else if (sw_gui_contrast == 2)
			{
				contrastSToneExp(src, show, sigma_gui_contrast, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			imshow(wname, show);
			key = waitKey(1);

			if (key == 'l')
			{
				a_gui_contrast--;
				setTrackbarPos("a*0.1", wname, a_gui_contrast);
			}
			if (key == 'j')
			{
				a_gui_contrast++;
				setTrackbarPos("a*0.1", wname, a_gui_contrast);
			}
			if (key == 'i')
			{
				b_gui_contrast++;
				setTrackbarPos("b", wname, b_gui_contrast);
			}
			if (key == 'k')
			{
				b_gui_contrast--;
				setTrackbarPos("b", wname, b_gui_contrast);
			}
			if (key == 'b')
			{
				b_gui_contrast = (b_gui_contrast == 0) ? 127 : 0;
				cv::setTrackbarPos("b", wname, b_gui_contrast);
			}
			if (key == '?')
			{
				cout << "i,j,k,l: move a,b" << endl;
				cout << "q: quit" << endl;
				cout << "b: flip b->127:0 ";
			}
		}
		destroyWindow(wname);
		return show;
	}

}
