
#include "imq.h"
#include <math.h>

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

const double K1 = 0.01;
const double K2 = 0.03;
const _INT32 lpf_width = 9;
const double luminance_exponent[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.1333 };
const double contrast_exponent[] = { 1.0, 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };
const double structure_exponent[] = { 1.0, 0.0448, 0.2856, 0.3001, 0.2363, 0.1333 };
const double lod[] = { 0.0378, -0.0238, -0.1106, 0.3774, 0.8527, 0.3774, -0.1106, -0.0238, 0.0378 };
const _INT32 filter_width = 11;
const bool do_ssim_cut = true;
const bool interpolationMethod = true;
const _INT32 dowsampled_add = 1;

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double getScale(float* kernel, _INT32 length) {
	double scale = 1.0;
	bool normalize = true;
	if (normalize) {
		double sum = 0.0;
		for (_INT32 i = 0; i < length; ++i)
			sum += kernel[i];
		if (sum != 0.0)
			scale = (float)(1.0 / sum);
	}
	return scale;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

float getPixel(_INT32 x, _INT32 y, float* pixels, _INT32 width, _INT32 height) {
	if (x <= 0) x = 0;
	if (x >= width) x = width - 1;
	if (y <= 0) y = 0;
	if (y >= height) y = height - 1;
	return pixels[x + y * width];
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double getInterpolatedPixel2(double x, double y, float* pixels, _INT32 width, _INT32 height) {
	_INT32 xbase = (_INT32)x;
	_INT32 ybase = (_INT32)y;
	double xFraction = x - double(xbase);
	double yFraction = y - double(ybase);
	_INT32 offset = (_INT32)(ybase * width + xbase);
	_INT32 lowerLeft = (_INT32)pixels[offset];
	//if ((xbase>=(width-1))||(ybase>=(height-1)))
	//	return lowerLeft;
	_INT32 lowerRight = (_INT32)pixels[offset + 1];
	_INT32 upperRight = (_INT32)pixels[offset + width + 1];
	_INT32 upperLeft = (_INT32)pixels[offset + width];
	double upperAverage = upperLeft + xFraction * (upperRight - upperLeft);
	double lowerAverage = lowerLeft + xFraction * (lowerRight - lowerLeft);
	return lowerAverage + yFraction * (upperAverage - lowerAverage);
}


double getInterpolatedPixel(double x, double y, float* pixels, _INT32 width, _INT32 height) {

	if (x < 0.0) x = 0.0;
	if (x >= double(width) - 1.0) x = double(width) - 1.001;
	if (y < 0.0) y = 0.0;
	if (y >= double(height) - 1.0) y = double(height) - 1.001;
	return getInterpolatedPixel2(x, y, pixels, width, height);

}

_INT32 getPixelInterpolated(double x, double y, float* pixels, _INT32 width, _INT32 height) {

	if (x < 0.0 || y < 0.0 || x >= width - 1 || y >= height - 1)
		return 0;
	else
		return (_INT32)getInterpolatedPixel2(x, y, pixels, width, height);

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

bool resize(float *mas, _INT32 dstWidth, _INT32 *WW, _INT32 *HH, bool modify, bool interpolationMethod)
{

	_INT32 Height = *HH;
	_INT32 Width = *WW;
	_INT32 width = Width;
	_INT32 height = Height;
	_INT32 dstHeight = (_INT32)(dstWidth*((double)Height / Width));
	double srcCenterX = 0.0 + ((double)Width) / 2.0;
	double srcCenterY = 0.0 + ((double)Height) / 2.0;
	double dstCenterX = ((double)dstWidth) / 2.0;
	double dstCenterY = ((double)dstHeight) / 2.0;
	double xScale = (double)(double(dstWidth) / Width);
	double yScale = (double)(double(dstHeight) / Height);

	if (interpolationMethod) {
		dstCenterX += xScale / 2.0;
		dstCenterY += yScale / 2.0;
	}

	float* pixels = mas;
	float* pixels2 = NULL;
	try {
		pixels2 = new float[dstWidth * dstHeight];
	}
	catch (...) { return false; }

	double xs, ys;
	_INT32 index1, index2;
	double xlimit = double(width) - 1.0, xlimit2 = double(width) - 1.001;
	double ylimit = double(height) - 1.0, ylimit2 = double(height) - 1.001;
	for (_INT32 y = 0; y <= dstHeight - 1; ++y) {
		ys = (double(y) - dstCenterY) / yScale + srcCenterY;
		if (interpolationMethod) {
			if (ys < 0.0) ys = 0.0;
			if (ys >= ylimit) ys = ylimit2;
		}
		index1 = width * ((_INT32)ys);
		index2 = y * dstWidth;
		for (_INT32 x = 0; x <= dstWidth - 1; ++x) {
			xs = (double(x) - dstCenterX) / xScale + srcCenterX;
			if (interpolationMethod) {
				if (xs < 0.0) xs = 0.0;
				if (xs >= xlimit) xs = xlimit2;
				pixels2[index2++] = (float)((_INT32)(getInterpolatedPixel2(xs, ys, pixels, width, height) + 0.5));
			}
			else
				pixels2[index2++] = pixels[index1 + (_INT32)xs];
		}
	}

	for (_INT32 x = 0; x < dstWidth * dstHeight; ++x) pixels[x] = pixels2[x];
	delete[] pixels2;
	if (modify) {
		*WW = dstWidth;
		*HH = dstHeight;
	}
	return true;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

bool convolve(float *image, _INT32 width, _INT32 height, float* kernel, _INT32 kernel_len, _INT32 kw, _INT32 kh, bool fast = false) {

	_INT32 i;
	_INT32 offset;
	_INT32 uc = kw / 2;
	_INT32 vc = kh / 2;
	_INT32 x1 = 0;
	_INT32 y1 = 0;
	_INT32 x2 = x1 + width;
	_INT32 y2 = y1 + height;
	float* pixels = image;
	float* pixels2 = NULL;
	_INT32 size = width * height;

	try {
		pixels2 = new float[size];
	}
	catch (...) { return false; }

	for (i = 0; i < size; ++i) pixels2[i] = pixels[i];

	double scale = getScale(kernel, kernel_len);

	double sum;
	bool edgePixel;
	_INT32 xedge = width - uc;
	_INT32 yedge = height - vc;

	if (!fast) {

		for (_INT32 y = y1; y < y2; ++y) {
			for (_INT32 x = x1; x < x2; ++x) {


				sum = 0.0;
				i = 0;
				edgePixel = y < vc || y >= yedge || x < uc || x >= xedge;

				for (_INT32 v = -vc; v <= vc; ++v) {

					offset = x + (y + v) * width;
					for (_INT32 u = -uc; u <= uc; ++u) {

						if (edgePixel) {
							//if (i>=kernel_len) // work around for JIT compiler bug on Linux IJ.log("kernel index error: "+i);
							sum += getPixel(x + u, y + v, pixels2, width, height) * kernel[i++];
						}
						else
							sum += pixels2[offset + u] * kernel[i++];
					} // for u = 
				} // for v =

				pixels[x + y * width] = float(sum*scale);

			}
		}
	}
	else // if fast
	{


		for (_INT32 y = y1; y < y2; ++y) {
			for (_INT32 x = x1; x < x2; ++x) {

				if ((y % kh == 0) || (x % kw == 0)) {
					sum = 0.0;
					i = 0;
					edgePixel = y < vc || y >= yedge || x < uc || x >= xedge;

					for (_INT32 v = -vc; v <= vc; ++v) {

						offset = x + (y + v) * width;
						for (_INT32 u = -uc; u <= uc; ++u) {

							if (edgePixel) {
								//if (i>=kernel_len) // work around for JIT compiler bug on Linux IJ.log("kernel index error: "+i);
								sum += getPixel(x + u, y + v, pixels2, width, height) * kernel[i++];
							}
							else
								sum += pixels2[offset + u] * kernel[i++];
						} // for u = 
					} // for v =

					pixels[x + y * width] = float(sum*scale);
				}
				else // if x % and y %
					pixels[x + y * width] = 0.0f;

			}
		}



	}

	delete[] pixels2;
	return true;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double MS_SSIMF(float *forig_img, float *fcomp_img, _INT32 PX, _INT32 PY, bool algorithm_selection_Wang, bool SSIM, _INT32 bits_per_pixel_1, bool fast,
	float aa, float bb, float gg) {

	double result = 0.0;
	double luminance_comparison = 1.0;
	double contrast_comparison = 1.0;
	double structure_comparison = 1.0;
	double sigma_gauss = 1.5;
	_INT32  pointer, filter_length, image_height, image_width, image_dimension, a, b;
	float *filter_weights = NULL;
	const _INT32 number_of_levels = 5;
	bool gaussian_window = true;
	double ms_ssim_index = 0.0;
	_INT32 size = PX * PY;

	double C1 = (pow(2.0, double(bits_per_pixel_1)) - 1.0) * K1;
	double C2 = (pow(2.0, double(bits_per_pixel_1)) - 1.0) * K2;
	C1 = C1 * C1;
	C2 = C2 * C2;


	//
	// NOW, WE CREATE THE FILTER, GAUSSIAN OR MEDIA FILTER, ACCORDING TO THE VALUE OF boolean "gaussian_window"
	//
	filter_length = filter_width * filter_width;
	float *window_weights = NULL;
	double *array_gauss_window = NULL;

	try {
		window_weights = new float[filter_length];
		array_gauss_window = new double[filter_length];
	}
	catch (...) { if (array_gauss_window) delete[] array_gauss_window; if (window_weights) delete[] window_weights; return result; }

	if (gaussian_window) {

		double distance = 0.0;
		_INT32 center = filter_width / 2;
		double total = 0.0;
		double sigma_sq = sigma_gauss * sigma_gauss;

		for (_INT32 y = 0; y < filter_width; ++y) {
			for (_INT32 x = 0; x < filter_width; ++x) {
				distance = abs(x - center) * abs(x - center) + abs(y - center) * abs(y - center);
				pointer = y * filter_width + x;
				array_gauss_window[pointer] = (float)exp(-0.5 * distance / sigma_sq);
				total = total + array_gauss_window[pointer];
			}
		}
		for (pointer = 0; pointer < filter_length; ++pointer) {
			array_gauss_window[pointer] = array_gauss_window[pointer] / total;
			window_weights[pointer] = (float)array_gauss_window[pointer];
		}
	}
	else { 								// NO WEIGHTS. ALL THE PIXELS IN THE EVALUATION WINDOW HAVE THE SAME WEIGHT
		for (pointer = 0; pointer < filter_length; ++pointer) {
			array_gauss_window[pointer] = 1.0 / double(filter_length);
			window_weights[pointer] = (float)array_gauss_window[pointer];
		}
	}
	//
	// END OF FILTER SELECTION							
	//

	//
	// THE VALUE OF THE LOW PASS FILTER
	//
	float *lpf = NULL;
	try {

		lpf = new float[lpf_width * lpf_width];

	}
	catch (...) { if (array_gauss_window) delete[] array_gauss_window; if (window_weights) delete[] window_weights; return result; }

	for (a = 0; a < lpf_width; ++a) {
		for (b = 0; b < lpf_width; ++b) {
			lpf[a * lpf_width + b] = float(lod[a] * lod[b]);
		}
	}

	float suma_lpf = 0.0f;
	_INT32 cont = 0;

	for (cont = 0; cont < lpf_width * lpf_width; ++cont) {
		suma_lpf += lpf[cont];
	}

	for (cont = 0; cont < lpf_width * lpf_width; ++cont) {
		lpf[cont] /= suma_lpf;
	}

	//
	// MAIN ALGORITHM
	//

	_INT32 level = 1;

	image_width = PX;
	image_height = PY;

	double contrast[number_of_levels + 1];
	double structure[number_of_levels + 1];
	double luminance[number_of_levels + 1];
	float *array_mu1_ip = NULL;
	float *array_mu2_ip = NULL;
	float *mu1_mu2 = NULL;
	float *array_soporte_1 = NULL;
	float *array_soporte_2 = NULL;
	float *array_soporte_3 = NULL;
	bool stop = false;

	//
	// WE ARE GOING TO USE ARRAYS OF 6 LEVELS INSTEAD OF 5.
	// WE WANT TO FORCE THAT THE INDEX OVER THE LEVEL WERE THE SAME THAN THE INDEX OVER THE ARRAY. 
	// REMEMBER THAT IN JAVA THE FIRST INDEX OF AN ARRAY IS THE "0" POSITION. WE WILL NEVER USE THIS POSITION IN THE FOLLOWING THREE ARRAYS.
	//
	if (SSIM && do_ssim_cut && (image_width >= 256) && (image_height >= 256)) {

		double downsampled = dowsampled_add + (_INT32)image_height / 256;
		if (downsampled > 1.0) {
			stop = false;
			_INT32 new_image_width = (_INT32)(image_width / downsampled);
			if (!resize(fcomp_img, new_image_width, &image_width, &image_height, false, interpolationMethod)) stop = true;
			if (!resize(forig_img, new_image_width, &image_width, &image_height, true, interpolationMethod)) stop = true;
			if (stop)
			{
				if (array_gauss_window) delete[] array_gauss_window;
				if (window_weights) delete[] window_weights;
				if (lpf) delete[] lpf;
				return result;
			}
		}


	}

	//
	// WE ARE GOING TO USE ARRAYS OF 6 LEVELS INSTEAD OF 5.
	//

	for (level = 1; level <= number_of_levels; ++level) {	// THIS LOOP CALCULATES, FOR EACH ITERATION, THE VALUES OF L, C AND S

		if (level != 1)
		{
			stop = false;
			if (!convolve(fcomp_img, image_width, image_height, lpf, lpf_width*lpf_width, lpf_width, lpf_width, fast)) stop = true;
			if (!convolve(forig_img, image_width, image_height, lpf, lpf_width*lpf_width, lpf_width, lpf_width, fast)) stop = true;
			if (!resize(fcomp_img, image_width / 2, &image_width, &image_height, false)) stop = true;
			if (!resize(forig_img, image_width / 2, &image_width, &image_height, true)) stop = true;
			if (stop)
			{
				if (array_gauss_window) delete[] array_gauss_window;
				if (window_weights) delete[] window_weights;
				if (lpf) delete[] lpf;
				if (array_mu1_ip) delete[] array_mu1_ip;
				if (array_mu2_ip) delete[] array_mu2_ip;
				if (mu1_mu2) delete[] mu1_mu2;
				if (array_soporte_1) delete[] array_soporte_1;
				if (array_soporte_2) delete[] array_soporte_2;
				if (array_soporte_3) delete[] array_soporte_3;
				return result;
			}

		}

		image_dimension = image_width * image_height;

		array_mu1_ip = NULL;
		array_mu2_ip = NULL;

		try {

			array_mu1_ip = new float[image_dimension];
			array_mu2_ip = new float[image_dimension];

		}
		catch (...)
		{
			if (array_gauss_window) delete[] array_gauss_window;
			if (window_weights) delete[] window_weights;
			if (lpf) delete[] lpf;
			if (array_mu1_ip) delete[] array_mu1_ip;
			if (array_mu2_ip) delete[] array_mu2_ip;
			return result;
		}

		for (a = b = pointer = 0; pointer < image_dimension; ++pointer) {

			array_mu1_ip[pointer] = forig_img[pointer];
			array_mu2_ip[pointer] = fcomp_img[pointer];

		}

		stop = false;
		if (!convolve(array_mu1_ip, image_width, image_height, window_weights, filter_width*filter_width, filter_width, filter_width, fast)) stop = true;
		if (!convolve(array_mu2_ip, image_width, image_height, window_weights, filter_width*filter_width, filter_width, filter_width, fast)) stop = true;
		if (stop)
		{
			if (array_gauss_window) delete[] array_gauss_window;
			if (window_weights) delete[] window_weights;
			if (lpf) delete[] lpf;
			if (array_mu1_ip) delete[] array_mu1_ip;
			if (array_mu2_ip) delete[] array_mu2_ip;
			return result;
		}

		float *mu1_sq = array_mu1_ip;
		float *mu2_sq = array_mu2_ip;
		mu1_mu2 = NULL;

		try {
			mu1_mu2 = new float[image_dimension];
		}
		catch (...)
		{
			if (array_gauss_window) delete[] array_gauss_window;
			if (window_weights) delete[] window_weights;
			if (lpf) delete[] lpf;
			if (array_mu1_ip) delete[] array_mu1_ip;
			if (array_mu2_ip) delete[] array_mu2_ip;
			if (mu1_mu2) delete[] mu1_mu2;
			return result;
		}

		for (pointer = 0; pointer < image_dimension; ++pointer) {

			mu1_mu2[pointer] = array_mu1_ip[pointer] * array_mu2_ip[pointer];
			mu1_sq[pointer] = array_mu1_ip[pointer] * array_mu1_ip[pointer];
			mu2_sq[pointer] = array_mu2_ip[pointer] * array_mu2_ip[pointer];

		}


		//	
		//THERE IS A METHOD IN IMAGEJ THAT CONVOLVES ANY ARRAY, BUT IT ONLY WORKS WITH IMAGE PROCESSORS. THIS IS THE REASON BECAUSE I CREATE THE FOLLOWING PROCESSORS
		//
		array_soporte_1 = NULL;
		array_soporte_2 = NULL;
		array_soporte_3 = NULL;

		try {

			array_soporte_1 = new float[image_dimension];
			array_soporte_2 = new float[image_dimension];
			array_soporte_3 = new float[image_dimension];

		}
		catch (...)
		{
			if (array_gauss_window) delete[] array_gauss_window;
			if (window_weights) delete[] window_weights;
			if (lpf) delete[] lpf;
			if (array_mu1_ip) delete[] array_mu1_ip;
			if (array_mu2_ip) delete[] array_mu2_ip;
			if (mu1_mu2) delete[] mu1_mu2;
			if (array_soporte_1) delete[] array_soporte_1;
			if (array_soporte_2) delete[] array_soporte_2;
			if (array_soporte_3) delete[] array_soporte_3;
			return result;
		}


		float *sigma1_sq = array_soporte_1;
		float *sigma2_sq = array_soporte_2;
		float *sigma12 = array_soporte_3;

		for (pointer = 0; pointer < image_dimension; ++pointer) {

			array_soporte_1[pointer] = forig_img[pointer] * forig_img[pointer];
			array_soporte_2[pointer] = fcomp_img[pointer] * fcomp_img[pointer];
			array_soporte_3[pointer] = forig_img[pointer] * fcomp_img[pointer];

		}

		stop = false;
		if (!convolve(array_soporte_1, image_width, image_height, window_weights, filter_width*filter_width, filter_width, filter_width, fast)) stop = true;
		if (!convolve(array_soporte_2, image_width, image_height, window_weights, filter_width*filter_width, filter_width, filter_width, fast)) stop = true;
		if (!convolve(array_soporte_3, image_width, image_height, window_weights, filter_width*filter_width, filter_width, filter_width, fast)) stop = true;
		if (stop)
		{
			if (array_gauss_window) delete[] array_gauss_window;
			if (window_weights) delete[] window_weights;
			if (lpf) delete[] lpf;
			if (array_mu1_ip) delete[] array_mu1_ip;
			if (array_mu2_ip) delete[] array_mu2_ip;
			if (mu1_mu2) delete[] mu1_mu2;
			if (array_soporte_1) delete[] array_soporte_1;
			if (array_soporte_2) delete[] array_soporte_2;
			if (array_soporte_3) delete[] array_soporte_3;
			return result;
		}

		for (pointer = 0; pointer < image_dimension; ++pointer) {

			sigma1_sq[pointer] -= mu1_sq[pointer];
			sigma2_sq[pointer] -= mu2_sq[pointer];
			sigma12[pointer] -= mu1_mu2[pointer];
			//
			// THE FOLLOWING SENTENCES ARE VERY AD-HOC. SOMETIMES, FOR INTERNAL REASONS OF PRECISION OF CALCULATIONS AROUND THE BORDERS, SIGMA_SQ
			// CAN BE NEGATIVE. THE VALUE CAN BE AROUND 0.001 IN SOME POINTS (A FEW). THE PROBLEM IS THAT, FOR SIMPICITY I CALCULATE SIGMA1 AS SQUARE ROOT OF SIGMA1_SQ
			// OF COURSE, IF THE ALGORITHM FINDS NEGATIVE VALUES, YOU GET THE MESSAGE  "IS NOT A NUMBER" IN RUN TIME.
			// 
			if (sigma1_sq[pointer] < 0.0f) {
				sigma1_sq[pointer] = 0.0f;
			}
			if (sigma2_sq[pointer] < 0.0f) {
				sigma2_sq[pointer] = 0.0f;
			}
		}


		//
		// WE HAVE GOT ALL THE VALUES TO CALCULATE LUMINANCE, CONTRAST AND STRUCTURE
		//
		double luminance_point = 1.0;
		double contrast_point = 0.0;
		double structure_point = 0.0;
		double suma = 0.0;
		luminance[level] = 0.0;
		contrast[level] = 0.0;
		structure[level] = 0.0;

		//
		// IF SSIM INDEX SELECTED, NO OTHER LEVELS NEEDED
		//

		if (SSIM)
		{
			_INT32 i, j, count = 0;
			suma = 0.0;
			if ((aa == 1.0f) && (bb == 1.0f) && (gg == 1.0f))
			{
				if (fast)
				{
					for (i = 0; i < image_height; ++i)
					{
						for (j = 0; j < image_width; ++j) if ((i % filter_width == 0) || (j % filter_width == 0))
						{
							pointer = j + i * image_width;
							++count;
							suma += ((2.0 * mu1_mu2[pointer] + C1) * (2.0 * sigma12[pointer] + C2)) /
								((mu1_sq[pointer] + mu2_sq[pointer] + C1) * (sigma1_sq[pointer] + sigma2_sq[pointer] + C2));
						}
					}
				}
				else // if not fast
				{
					for (pointer = 0; pointer < image_dimension; ++pointer)
					{
						suma += ((2.0 * mu1_mu2[pointer] + C1) * (2.0 * sigma12[pointer] + C2)) /
							((mu1_sq[pointer] + mu2_sq[pointer] + C1) * (sigma1_sq[pointer] + sigma2_sq[pointer] + C2));
					}
					count = image_dimension;
				}
			}
			else // setting ABG weights
			{
				if (fast) 
				{
					for (i = 0; i < image_height; ++i)
						for (j = 0; j < image_width; ++j) if ((i % filter_width == 0) || (j % filter_width == 0))
						{
							pointer = j + i * image_width;
							luminance_point = fabs(((2.0 * mu1_mu2[pointer] + C1) / (mu1_sq[pointer] + mu2_sq[pointer] + C1)));
							contrast_point = fabs(((2.0 *sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2) / (sigma1_sq[pointer] + sigma2_sq[pointer] + C2)));
							structure_point = fabs(((sigma12[pointer] + C2 / 2.0) / (sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2 / 2.0)));
							++count;
							suma += pow(luminance_point, double(aa)) * pow(contrast_point, double(bb)) * pow(structure_point, double(gg));



						}
				}
				else // if not fast
				{

					for (pointer = 0; pointer < image_dimension; ++pointer) {
						luminance_point = fabs(((2.0 * mu1_mu2[pointer] + C1) / (mu1_sq[pointer] + mu2_sq[pointer] + C1)));
						contrast_point = fabs(((2.0 *sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2) / (sigma1_sq[pointer] + sigma2_sq[pointer] + C2)));
						structure_point = fabs(((sigma12[pointer] + C2 / 2.0) / (sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2 / 2.0)));
						suma += pow(luminance_point, double(aa)) * pow(contrast_point, double(bb)) * pow(structure_point, double(gg));
					}
					count = image_dimension;

				}


			}

			result = suma / double(count);

			delete[] array_soporte_1;
			delete[] array_soporte_2;
			delete[] array_soporte_3;
			delete[] array_mu1_ip;
			delete[] array_mu2_ip;
			delete[] mu1_mu2;
			delete[] window_weights;
			delete[] array_gauss_window;
			delete[] lpf;

			return result;
		}

		if (algorithm_selection_Wang) 
		{
			for (pointer = 0; pointer < image_dimension; ++pointer) 
			{
				luminance_point = ((2.0 * mu1_mu2[pointer] + C1) / (mu1_sq[pointer] + mu2_sq[pointer] + C1));
				luminance[level] += luminance_point;

				contrast_point = ((2.0 *sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2) / (sigma1_sq[pointer] + sigma2_sq[pointer] + C2));
				contrast[level] += contrast_point;

				structure_point = ((sigma12[pointer] + C2 / 2.0) / (sqrt(sigma1_sq[pointer] * sigma2_sq[pointer]) + C2 / 2.0));
				structure[level] += structure_point;
			}
		}
		else 
		{   // ROUSE/HEMAMI
			for (pointer = 0; pointer < image_dimension; ++pointer)
			{

				if ((mu1_sq[pointer] + mu2_sq[pointer]) == 0.0f)
					luminance_point = 1.0;
				else
					luminance_point = ((2.0 * mu1_mu2[pointer]) / (mu1_sq[pointer] + mu2_sq[pointer]));

				luminance[level] += luminance_point;

				if ((sigma1_sq[pointer] + sigma2_sq[pointer]) == 0.0f)
					contrast_point = 1.0;
				else
					contrast_point = ((2.0 * sqrt(sigma1_sq[pointer] * sigma2_sq[pointer])) / (sigma1_sq[pointer] + sigma2_sq[pointer]));

				contrast[level] += contrast_point;

				if (((sigma1_sq[pointer] == 0.0f) | (sigma2_sq[pointer] == 0.0f)) & (fabs(sigma1_sq[pointer]) != fabs(sigma2_sq[pointer])))
					structure_point = 0.0;
				else
					if ((sqrt(sigma1_sq[pointer]) == 0.0f) & (sqrt(sigma2_sq[pointer]) == 0.0f))
						structure_point = 1.0;
					else
						structure_point = ((sigma12[pointer]) / (sqrt(sigma1_sq[pointer] * sigma2_sq[pointer])));

				structure[level] += structure_point;

			}
		}	// END WANG - ROUSE/HEMAMI IF-ELSE

		contrast[level] /= double(image_dimension);
		structure[level] /= double(image_dimension);
		if (level == number_of_levels)
			luminance[level] /= double(image_dimension);
		else
			luminance[level] = 1.0;


		//
		// END-FOR OF OUTER LOOP OVER THE DIFFERENT VIEWING LEVELS
		//

		delete[] array_soporte_1;
		delete[] array_soporte_2;
		delete[] array_soporte_3;
		delete[] array_mu1_ip;
		delete[] array_mu2_ip;
		delete[] mu1_mu2;

		array_soporte_1 = NULL;
		array_soporte_2 = NULL;
		array_soporte_3 = NULL;
		array_mu1_ip = NULL;
		array_mu2_ip = NULL;
		mu1_mu2 = NULL;
	}

	for (level = 1; level <= number_of_levels; ++level) {

		if (structure[level] < 0.0) structure[level] = -1.0 * structure[level];
		luminance_comparison = pow(luminance[level], luminance_exponent[level]) * luminance_comparison;
		contrast_comparison = pow(contrast[level], contrast_exponent[level]) * contrast_comparison;
		structure_comparison = pow(structure[level], structure_exponent[level]) * structure_comparison;
	}

	result = ms_ssim_index = luminance_comparison * contrast_comparison * structure_comparison;


	delete[] window_weights;
	delete[] array_gauss_window;
	delete[] lpf;

	return result;
}


//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoMS_SSIM(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast) {

	_INT32 size = PX * PY;
	double result = 0.0;

	float *orig_imgb = NULL;
	float *comp_imgb = NULL;
	try {
		orig_imgb = new float[size];
		comp_imgb = new float[size];
	}
	catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete[] comp_imgb; return result; }


	switch (BPP) {


	case 8: {


		for (_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i] & _MBYTE); comp_imgb[i] = (float)(comp_img[i] & _MBYTE); }
		result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 8, fast, 0.0f, 0.0f, 0.0f);


	};
			break;

	case 16: {

		for (_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i] & _MINT16); comp_imgb[i] = (float)(comp_img[i] & _MINT16); }
		result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 16, fast, 0.0f, 0.0f, 0.0f);


	};
			 break;


	default: break;


	}

	delete[] orig_imgb;
	delete[] comp_imgb;
	return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double MS_SSIM8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, bool Wang, bool fast) {

	_INT32 size = PX * PY;
	double result = 0.0;

	float *orig_imgb = NULL;
	float *comp_imgb = NULL;
	try {
		orig_imgb = new float[size];
		comp_imgb = new float[size];
	}
	catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete[] comp_imgb; return result; }


	for (_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i] & _MBYTE); comp_imgb[i] = (float)(comp_img[i] & _MBYTE); }
	result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 8, fast, 0.0f, 0.0f, 0.0f);

	delete[] orig_imgb;
	delete[] comp_imgb;
	return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double MS_SSIM16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY, bool Wang, bool fast)
{

	_INT32 size = PX * PY;
	double result = 0.0;

	float *orig_imgb = NULL;
	float *comp_imgb = NULL;
	try {
		orig_imgb = new float[size];
		comp_imgb = new float[size];
	}
	catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete[] comp_imgb; return result; }


	for (_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (float)(orig_img[i] & _MINT16); comp_imgb[i] = (float)(comp_img[i] & _MINT16); }
	result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 16, fast, 0.0f, 0.0f, 0.0f);

	delete[] orig_imgb;
	delete[] comp_imgb;
	return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------


double DoMS_SSIMY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast)
{

	_INT32 size = PX * PY;
	double result = 0.0;

	switch (BPP) {

	case 8: case 16:
		result = DoMS_SSIM(orig_img, comp_img, PX, PY, BPP, Wang, fast);
		break;

	case 24: {

		float *orig_imgb = NULL;
		float *comp_imgb = NULL;
		try {
			orig_imgb = new float[size];
			comp_imgb = new float[size];
		}
		catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete[] comp_imgb; return result; }

		for (_INT32 i = 0; i < size; ++i) {
			double Y1 = rgB * (double)((orig_img[i] >> 16)&_MBYTE) + rGb * (double)((orig_img[i] >> 8)&_MBYTE) + Rgb * (double)(orig_img[i] & _MBYTE) + Crgb;
			double Y2 = rgB * (double)((comp_img[i] >> 16)&_MBYTE) + rGb * (double)((comp_img[i] >> 8)&_MBYTE) + Rgb * (double)(comp_img[i] & _MBYTE) + Crgb;
			comp_imgb[i] = (float)Y2;
			orig_imgb[i] = (float)Y1;
		}

		result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 8, fast, 0.0f, 0.0f, 0.0f);
		delete[] orig_imgb;
		delete[] comp_imgb;
	};
			 break;

	default: break;

	}

	return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoMS_SSIMY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP, bool Wang, bool fast)
{

	_INT32 size = PX * PY;
	_INT32 bsize = size * 3;
	double result = 0.0;

	switch (BPP) {

	case 24: {

		float *orig_imgb = NULL;
		float *comp_imgb = NULL;
		try {
			orig_imgb = new float[size];
			comp_imgb = new float[size];
		}
		catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete[] comp_imgb; return result; }

		for (_INT32 i = 0, j = 0; i < bsize; i += 3, ++j) {
			if ((i < bsize) && (i + 1 < bsize) && (i + 2 < bsize) && (j < size)) {
				double Y1 = rgB * (double)(orig_img[i]) + rGb * (double)(orig_img[i + 1]) + Rgb * (double)(orig_img[i + 2]) + Crgb;
				double Y2 = rgB * (double)(comp_img[i]) + rGb * (double)(comp_img[i + 1]) + Rgb * (double)(comp_img[i + 2]) + Crgb;
				comp_imgb[j] = (float)Y2;
				orig_imgb[j] = (float)Y1;
			}
		}

		result = MS_SSIMF(orig_imgb, comp_imgb, PX, PY, Wang, false, 8, fast, 0.0f, 0.0f, 0.0f);
		delete[] orig_imgb;
		delete[] comp_imgb;
	};
			 break;

	default: break;

	}

	return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
