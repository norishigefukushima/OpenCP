#include "contrast.hpp"
#include "debugcp.hpp"
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

	template<typename T>
	void contrastGamma_(T* src, T* dest, const int size, const double gamma)
	{
		const double gm = 1.0 / gamma;
		for (int i = 0; i < size; i++)
		{
			dest[i] = saturate_cast<T>(pow(src[i] / 255.0, gm) * 255.0);
		}
	}

	template<>
	void contrastGamma_<uchar>(uchar* src, uchar* dest, const int size, const double gamma)
	{
		uchar LUT[256];
		for (int i = 0; i < 256; i++)
		{
			LUT[i] = saturate_cast<uchar>(pow((double)i / 255.0, 1.0 / gamma) * 255.0);
		}
		cv::Mat lut_mat = cv::Mat(256, 1, CV_8UC1, LUT);
		cv::Mat s = cv::Mat(1, size, CV_8UC1, src);
		cv::Mat d = cv::Mat(1, size, CV_8UC1, dest);
		cv::LUT(s, lut_mat, d);
	}

	// Gamma Correction
	void contrastGamma(cv::InputArray src, cv::OutputArray dest, const double gamma)
	{
		dest.create(src.size(), src.type());
		const int size = src.getMat().size().area() * src.getMat().channels();

		switch (src.depth())
		{
		case CV_8U: contrastGamma_<uchar>(src.getMat().ptr<uchar>(), dest.getMat().ptr<uchar>(), size, gamma); break;
		case CV_8S: contrastGamma_<char>(src.getMat().ptr<char>(), dest.getMat().ptr<char>(), size, gamma); break;
		case CV_16U:contrastGamma_<ushort>(src.getMat().ptr<ushort>(), dest.getMat().ptr<ushort>(), size, gamma); break;
		case CV_16S:contrastGamma_< short>(src.getMat().ptr< short>(), dest.getMat().ptr< short>(), size, gamma); break;
		case CV_32S:contrastGamma_<   int>(src.getMat().ptr<   int>(), dest.getMat().ptr<   int>(), size, gamma); break;
		case CV_32F:contrastGamma_< float>(src.getMat().ptr< float>(), dest.getMat().ptr< float>(), size, gamma); break;
		case CV_64F:contrastGamma_<double>(src.getMat().ptr<double>(), dest.getMat().ptr<double>(), size, gamma); break;
		}
	}

	template<typename T>
	void quantization_(T* src, T* dest, const int size, const int num_levels)
	{
		const double nl = 256.0 / (double)max(1, num_levels);
		const double nlstep = 255.0 / (double)max(1, num_levels - 1);

		for (int i = 0; i < size; i++)
		{
			dest[i] = saturate_cast<T>(saturate_cast<int>(src[i] / nl) * nlstep);
		}
	}

	void quantization(cv::InputArray src, cv::OutputArray dest, const int num_levels)
	{
		dest.create(src.size(), src.type());
		const int size = src.getMat().size().area() * src.getMat().channels();

		switch (src.depth())
		{
		case CV_8U: quantization_<uchar>(src.getMat().ptr<uchar>(), dest.getMat().ptr<uchar>(), size, num_levels); break;
		case CV_8S: quantization_<char>(src.getMat().ptr<char>(), dest.getMat().ptr<char>(), size, num_levels); break;
		case CV_16U:quantization_<ushort>(src.getMat().ptr<ushort>(), dest.getMat().ptr<ushort>(), size, num_levels); break;
		case CV_16S:quantization_< short>(src.getMat().ptr< short>(), dest.getMat().ptr< short>(), size, num_levels); break;
		case CV_32S:quantization_<   int>(src.getMat().ptr<   int>(), dest.getMat().ptr<   int>(), size, num_levels); break;
		case CV_32F:quantization_< float>(src.getMat().ptr< float>(), dest.getMat().ptr< float>(), size, num_levels); break;
		case CV_64F:quantization_<double>(src.getMat().ptr<double>(), dest.getMat().ptr<double>(), size, num_levels); break;
		}
	}

	cv::Mat guiContrast(InputArray src_, string wname)
	{
		cp::Plot p(Size(300, 300));
		p.setKey(cp::Plot::NOKEY);
		p.setPlotSymbol(0, cp::Plot::NOPOINT);
		//p.setPlotThickness
		p.setIsDrawMousePosition(false);
		p.setXYRange(0, 255, 0, 255);
		p.setGrid(2);

		Mat src = src_.getMat();
		namedWindow(wname);
		static int a_gui_contrast = 10;
		static int b_gui_contrast = 0;
		static int sw_gui_contrast = 0;
		static int sigma_gui_contrast = 30;
		static int gamma_gui_contrast = 100;
		static int quantization_gui_contrast = 8;
		cv::createTrackbar("sw", wname, &sw_gui_contrast, 4);
		cv::createTrackbar("a*0.1", wname, &a_gui_contrast, 100);
		cv::setTrackbarMin("a*0.1", wname, -100);
		cv::createTrackbar("b", wname, &b_gui_contrast, 255);
		cv::setTrackbarMin("b", wname, -255);
		cv::createTrackbar("sigma", wname, &sigma_gui_contrast, 255);
		cv::createTrackbar("gamma*0.01", wname, &gamma_gui_contrast, 1000);
		cv::createTrackbar("quantization", wname, &quantization_gui_contrast, 255);

		int key = 0;
		cp::UpdateCheck uc(sw_gui_contrast);
		cv::Mat show;
		while (key != 'q')
		{
			if (uc.isUpdate(sw_gui_contrast))
			{
				string mes = "";
				if (sw_gui_contrast == 0)mes = "ax + b (convertTo)";
				else if (sw_gui_contrast == 1)mes = "a(x-b) + b";
				else if (sw_gui_contrast == 2)mes = "x - a*gauss(x-b)(x-b)";
				else if (sw_gui_contrast == 3)mes = "gamma: pow(x / 255, 1/gamma) * 255.0";
				else if (sw_gui_contrast == 4)mes = "quantization: int(x/(255.0/level))*level";
				else mes = "not supported";
				displayOverlay(wname, mes, 3000);
			}

			if (sw_gui_contrast == 0)
			{
				for (int i = 0; i < 256; i++)
				{
					//ax+b
					p.push_back(i, 0.1 * a_gui_contrast * i + b_gui_contrast);
				}
				src.convertTo(show, CV_8U, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			else if (sw_gui_contrast == 1)
			{
				for (int i = 0; i < 256; i++)
				{
					//a(x-b)+b=a*x -a * b + b
					p.push_back(i, 0.1 * a_gui_contrast * i + b_gui_contrast - 0.1 * a_gui_contrast * b_gui_contrast);
				}
				show = cenvertCentering(src, CV_8U, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			else if (sw_gui_contrast == 2)
			{
				const double coeff = -0.5 / (sigma_gui_contrast * sigma_gui_contrast);
				for (int i = 0; i < 256; i++)
				{
					double v = (i - b_gui_contrast);
					p.push_back(i, 0.1 * a_gui_contrast * exp(v * v * coeff) * v + i);
				}
				contrastSToneExp(src, show, sigma_gui_contrast, 0.1 * a_gui_contrast, b_gui_contrast);
			}
			else if (sw_gui_contrast == 3)
			{
				for (int i = 0; i < 256; i++)
				{
					p.push_back(i, pow((double)i / 255.0, 1.0 / (gamma_gui_contrast * 0.01)) * 255.0);
				}
				contrastGamma(src, show, 0.01 * gamma_gui_contrast);
			}
			else if (sw_gui_contrast == 4)
			{
				const double nl = 256.0 / (double)max(1, quantization_gui_contrast);
				const double nlstep = 255.0 / (double)max(1, quantization_gui_contrast - 1);

				for (int i = 0; i < 256; i++)
				{
					p.push_back(i, saturate_cast<uchar>(int(i / nl) * nlstep));
				}
				quantization(src, show, quantization_gui_contrast);
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

			p.plot("tone curve", false);
			p.clear();
		}
		destroyWindow(wname);
		return show;
	}

}
