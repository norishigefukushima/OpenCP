#include "filterCore.h"

void set1DSpaceKernel45(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=j) continue;

			//printf("45: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void set1DSpaceKernel135(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=-j) continue;
			//printf("135: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;

			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;

			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}

void set1DSpaceKernel45(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
	{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=j) continue;

			//printf("45: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}

void set1DSpaceKernel135(float* space_weight, int* space_ofs, int* space_guide_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep1, const int imstep2, const bool isRectangle)
	{
	const int maxr = std::max(radiusV,radiusH);
	for(int i = -radiusV; i <= radiusV; i++ )
	{
		for(int j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > maxr && !isRectangle) continue;
			if( i!=-j) continue;

			//printf("135: %d %d\n", i,j);
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk] = (int)(i*imstep1 + j);
			space_guide_ofs[maxk++] = (int)(i*imstep2 + j);
		}
	}
}
