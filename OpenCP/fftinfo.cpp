#include "fftinfo.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	void imshowFFT(string wname, InputArray src__)
	{
		Mat src_ = src__.getMat();
		Mat src;
		if (src_.channels() == 3)cvtColor(src_, src, COLOR_BGR2GRAY);
		else src = src_;
		Mat padded;                            //expand input image to optimal size
		int m = getOptimalDFTSize(src.rows);
		int n = getOptimalDFTSize(src.cols); // on the border add zero values
		copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

		Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
		Mat complexI;
		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros


		{
			//CalcTime t("fft");
			dft(complexI, complexI);            // this way the result may fit in the source matrix
		}

		// compute the magnitude and switch to logarithmic scale
		// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
		split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
		Mat magI = planes[0];

		magI += Scalar::all(1);                    // switch to logarithmic scale
		log(magI, magI);

		// crop the spectrum, if it has an odd number of rows or columns
		magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

		// rearrange the quadrants of Fourier image  so that the origin is at the image center
		int cx = magI.cols / 2;
		int cy = magI.rows / 2;

		Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
		Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
		Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

		Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);

		//showMatInfo(magI);
		normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
		// viewable image form (float between values 0 and 1).

		Mat b, show;
		magI.convertTo(b, CV_8U, 255);
		applyColorMap(b, show, 2);
		imshow(wname, show);
	}
}