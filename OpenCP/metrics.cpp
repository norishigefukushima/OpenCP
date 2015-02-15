#include "opencp.hpp"

double YPSNR(const Mat& src1, const Mat& src2)
{
	Mat g1,g2;
	cvtColor(src1,g1,COLOR_BGR2GRAY);
	cvtColor(src2,g2,COLOR_BGR2GRAY);
	return PSNR(g1,g2);
}

double calcBadPixel(const Mat& src, const Mat& ref, int threshold)
{
	Mat g1,g2;
	if(src.channels()==3)
	{
		cvtColor(src,g1,CV_BGR2GRAY);
		cvtColor(ref,g2,CV_BGR2GRAY);
	}
	else
	{
		g1=src;
		g2=ref;
	}
	Mat temp;
	absdiff(g1,g2,temp);
	Mat mask;
	compare(temp,threshold,mask,CMP_GE);
	return 100.0*countNonZero(mask)/src.size().area();
}

Scalar getMSSIM( const Mat& i1, const Mat& i2, double sigma=1.5)
{
	int r = cvRound(sigma*3.0);
	Size kernel = Size(2*r+1,2*r+1);
	
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
	int d     = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, kernel,sigma);
    GaussianBlur(I2, mu2, kernel,sigma);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, kernel,sigma);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, kernel,sigma);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, kernel,sigma);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
	imshow("ssim",ssim_map);
    return mssim;
}

double SSIM(Mat& src, Mat& ref, double sigma)
{
	Mat gray1,gray2;
	cvtColor(src,gray1,CV_BGR2GRAY);
	cvtColor(ref,gray2,CV_BGR2GRAY);

	Scalar v = getMSSIM(gray1,gray2,sigma);
	return v.val[0];
}

inline int norm_l(int a, int b, int norm)
{
	if(norm==0)
	{
		int v = (a==b) ? 0 : 1;
		return v;
	}
	else if(norm==1)
	{
		return abs(a-b);
	}
	else
	{
		return 0;
	}
}

double calcTV(Mat& src)
{
	Mat gray;
	cvtColor(src,gray,CV_BGR2GRAY);
	Mat bb;
	copyMakeBorder(gray,bb,0,1,0,1,BORDER_REFLECT);

	int sum = 0;
	int count=0;

	int NRM = 0;
	for(int j=0;j<src.rows;j++)
	{
		uchar* pb = bb.ptr(j);
		uchar* b = bb.ptr(j+1);
		for(int i=0;i<src.rows;i++)
		{
			sum+=norm_l(pb[i],b[i],NRM);
			sum+=norm_l(b[i],b[i+1],NRM);
			count++;
		}
	}
	return (double)sum/(double)count;
}