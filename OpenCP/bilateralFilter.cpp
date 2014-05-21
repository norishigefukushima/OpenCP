#include "opencp.hpp"

void bilateralFilterSlowest(const Mat& src, Mat& dest, int d, double sigma_color, double sigma_space)
{
	Mat srcd;src.convertTo(srcd,CV_64F);
	Mat destd = Mat::zeros(src.size(),CV_MAKETYPE(CV_64F,src.channels()));
	const int r = d/2;
	int channels = src.channels();
	if(channels==1)
	{
		for(int j=0;j<src.rows;j++)
		{
			for(int i=0;i<src.cols;i++)
			{
				double sum = 0.0;
				double coeff = 0.0;
				for(int l=-r;l<=r;l++)
				{
					for(int k=-r;k<=r;k++)
					{
						if(sqrt(l*l+k*k)<=r && i+k>=0 && i+k<src.cols && j+l>=0 && j+l<src.rows )
						{
							double c = exp(-0.5*((srcd.at<double>(j+l,i+k)-srcd.at<double>(j,i))*(srcd.at<double>(j+l,i+k)-srcd.at<double>(j,i)))/(sigma_color*sigma_color));
							double s = exp(-0.5*(l*l+k*k)/(sigma_space*sigma_space));
							coeff+=c*s;
							sum+=srcd.at<double>(j+l,i+k)*c*s;
						}
					}
				}
				destd.at<double>(j,i)=sum/coeff;
			}
		}
	}
	else if(channels ==3)
	{
		for(int j=0;j<src.rows;j++)
		{
			for(int i=0;i<src.cols;i++)
			{
				double sumb = 0.0;
				double sumg = 0.0;
				double sumr = 0.0;
				double coeff = 0.0;
				for(int l=-r;l<=r;l++)
				{
					for(int k=-r;k<=r;k++)
					{
						if(sqrt(l*l+k*k)<=r && i+k>=0 && i+k<src.cols && j+l>=0 && j+l<src.rows )
						{
							double c = exp(-0.5*(
								(srcd.at<double>(j+l,3*(i+k)+0)-srcd.at<double>(j,3*i+0))*(srcd.at<double>(j+l,3*(i+k)+0)-srcd.at<double>(j,3*i+0))+
								(srcd.at<double>(j+l,3*(i+k)+1)-srcd.at<double>(j,3*i+1))*(srcd.at<double>(j+l,3*(i+k)+1)-srcd.at<double>(j,3*i+1))+
								(srcd.at<double>(j+l,3*(i+k)+2)-srcd.at<double>(j,3*i+2))*(srcd.at<double>(j+l,3*(i+k)+2)-srcd.at<double>(j,3*i+2))
								)/(sigma_color*sigma_color));
							double s = exp(-0.5*(l*l+k*k)/(sigma_space*sigma_space));
							coeff+=c*s;
							sumb+=srcd.at<double>(j+l,3*(i+k)+0)*c*s;
							sumg+=srcd.at<double>(j+l,3*(i+k)+1)*c*s;
							sumr+=srcd.at<double>(j+l,3*(i+k)+2)*c*s;
						}
					}
				}
				destd.at<double>(j,3*i+0)=sumb/coeff;
				destd.at<double>(j,3*i+1)=sumg/coeff;
				destd.at<double>(j,3*i+2)=sumr/coeff;
			}
		}
	}
	destd.convertTo(dest,src.type());
}