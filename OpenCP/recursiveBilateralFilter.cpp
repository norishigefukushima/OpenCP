#include "opencp.hpp"
#include <opencv2/core/internal.hpp>

//Qingxiong Yang, Recursive Bilateral Filtering, European Conference on Computer Vision (ECCV) 2012, 399-413.
//http://www.cs.cityu.edu.hk/~qiyang/publications/eccv-12/
using namespace std;

#define QX_DEF_PADDING					10

/*memory*/
inline double *** qx_allocd_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
	double *a,**p,***pp;
	int rc=r*c;
	int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*n*r);
	pp=(double***) malloc(sizeof(double**)*n);
	for(i=0;i<n;i++) 
		for(j=0;j<r;j++) 
			p[i*r+j]=&a[i*rc+j*c];
	for(i=0;i<n;i++) 
		pp[i]=&p[i*r];
	return(pp);
}
inline void qx_freed_3(double ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char** qx_allocu(int r,int c,int padding=QX_DEF_PADDING)
{
	unsigned char *a,**p;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(r*c+padding));
	if(a==NULL) {printf("qx_allocu() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void qx_freeu(unsigned char **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char *** qx_allocu_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
	unsigned char *a,**p,***pp;
	int rc=r*c;
	int i,j;
	a=(unsigned char*) malloc(sizeof(unsigned char )*(n*rc+padding));
	if(a==NULL) {printf("qx_allocu_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*n*r);
	pp=(unsigned char***) malloc(sizeof(unsigned char**)*n);
	for(i=0;i<n;i++) 
		for(j=0;j<r;j++) 
			p[i*r+j]=&a[i*rc+j*c];
	for(i=0;i<n;i++) 
		pp[i]=&p[i*r];
	return(pp);
}
inline void qx_freeu_3(unsigned char ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void qx_freeu_1(unsigned char*p)
{
	if(p!=NULL)
	{
		delete [] p;
		p=NULL;
	}
}
inline float** qx_allocf(int r,int c,int padding=QX_DEF_PADDING)
{
	float *a,**p;
	a=(float*) malloc(sizeof(float)*(r*c+padding));
	if(a==NULL) {printf("qx_allocf() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(float**) malloc(sizeof(float*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void qx_freef(float **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline float *** qx_allocf_3(int n,int r,int c,int padding=QX_DEF_PADDING)
{
	float *a,**p,***pp;
	int rc=r*c;
	int i,j;
	a=(float*) malloc(sizeof(float)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocf_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(float**) malloc(sizeof(float*)*n*r);
	pp=(float***) malloc(sizeof(float**)*n);
	for(i=0;i<n;i++) 
		for(j=0;j<r;j++) 
			p[i*r+j]=&a[i*rc+j*c];
	for(i=0;i<n;i++) 
		pp[i]=&p[i*r];
	return(pp);
}
inline void qx_freef_3(float ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline int** qx_alloci(int r,int c,int padding=QX_DEF_PADDING)
{
	int *a,**p;
	a=(int*) malloc(sizeof(int)*(r*c+padding));
	if(a==NULL) {printf("qx_alloci() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(int**) malloc(sizeof(int*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void qx_freei(int **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void qx_freei_1(int*p)
{
	if(p!=NULL)
	{
		delete [] p;
		p=NULL;
	}
}
inline double** qx_allocd(int r,int c,int padding=QX_DEF_PADDING)
{
	double *a,**p;
	a=(double*) malloc(sizeof(double)*(r*c+padding));
	if(a==NULL) {printf("qx_allocd() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void qx_freed(double **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char**** qx_allocu_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
	unsigned char *a,**p,***pp,****ppp;
	int nrc=n*r*c,nr=n*r,rc=r*c;
	int i,j,k;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(t*nrc+padding));
	if(a==NULL) {printf("qx_allocu_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*t*nr);
	pp=(unsigned char***) malloc(sizeof(unsigned char**)*t*n);
	ppp=(unsigned char****) malloc(sizeof(unsigned char***)*t);
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			for(j=0;j<r;j++)
				p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			pp[k*n+i]=&p[k*nr+i*r];
	for(k=0;k<t;k++)
		ppp[k]=&pp[k*n];
	return(ppp);
}
inline void qx_freeu_4(unsigned char ****p)
{
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline double**** qx_allocd_4(int t,int n,int r,int c,int padding=QX_DEF_PADDING)
{
	double *a,**p,***pp,****ppp;
	int nrc=n*r*c,nr=n*r,rc=r*c;
	int i,j,k;
	a=(double*) malloc(sizeof(double)*(t*nrc+padding));
	if(a==NULL) {printf("qx_allocd_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*t*nr);
	pp=(double***) malloc(sizeof(double**)*t*n);
	ppp=(double****) malloc(sizeof(double***)*t);
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			for(j=0;j<r;j++)
				p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			pp[k*n+i]=&p[k*nr+i*r];
	for(k=0;k<t;k++)
		ppp[k]=&pp[k*n];
	return(ppp);
}
inline void qx_freed_4(double ****p)
{

	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}


void gradient_domain_recursive_bilateral_filter_(double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,double***temp,double***temp_2w)
{
	double range_table[UCHAR_MAX+1];//compute a lookup table

	//double inv_sigma_range=-1.0/(sigma_range);
	double inv_sigma_range=-1.0/(2.0*sigma_range*sigma_range);

	for(int i=0;i<=UCHAR_MAX;i++)
	{
		//range_table[i]=exp(i*inv_sigma_range);
		range_table[i]=exp(i*i*inv_sigma_range);
	}

	double alpha=exp(-sqrt(2.0)/(sigma_spatial));//filter kernel size
	double***in_=in;
	unsigned char tpr,tpg,tpb,tcr,tcg,tcb;
	double ypr,ypg,ypb,ycr,ycg,ycb;
	for(int y=0;y<h;y++)//horizontal filtering
	{
		double*temp_x=temp[y][0];
		double*in_x=in_[y][0];
		unsigned char*texture_x=texture[y][0];
		*temp_x++=ypr=*in_x++; *temp_x++=ypg=*in_x++; *temp_x++=ypb=*in_x++;
		tpr=*texture_x++; tpg=*texture_x++; tpb=*texture_x++;
		for(int x=1;x<w;x++) //from left to right
		{
			tcr=*texture_x++; tcg=*texture_x++; tcb=*texture_x++;
			unsigned char dr=abs(tcr-tpr);
			unsigned char dg=abs(tcg-tpg);
			unsigned char db=abs(tcb-tpb);
			int range_dist=(((dr<<1)+dg+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;
			double inv_alpha_=1-alpha_;
			*temp_x++=ycr=inv_alpha_*(*in_x++)+alpha_*ypr; *temp_x++=ycg=inv_alpha_*(*in_x++)+alpha_*ypg; *temp_x++=ycb=inv_alpha_*(*in_x++)+alpha_*ypb;//update temp buffer
			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;
		}
		int w1=w-1;
		*--temp_x; *temp_x=0.5*((*temp_x)+(*--in_x)); 
		*--temp_x; *temp_x=0.5*((*temp_x)+(*--in_x)); 
		*--temp_x; *temp_x=0.5*((*temp_x)+(*--in_x));

		ypr=*in_x; ypg=*in_x; ypb=*in_x;
		tpr=*--texture_x; tpg=*--texture_x; tpb=*--texture_x;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			tcr=*--texture_x; tcg=*--texture_x; tcb=*--texture_x;
			unsigned char dr=abs(tcr-tpr);
			unsigned char dg=abs(tcg-tpg);
			unsigned char db=abs(tcb-tpb);
			int range_dist=(((dr<<1)+dg+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;
			double inv_alpha_=1-alpha_;

			ycr=inv_alpha_*(*--in_x)+alpha_*ypr; ycg=inv_alpha_*(*--in_x)+alpha_*ypg; ycb=inv_alpha_*(*--in_x)+alpha_*ypb;
			*--temp_x; *temp_x=0.5*((*temp_x)+ycr);
			*--temp_x; *temp_x=0.5*((*temp_x)+ycg);
			*--temp_x; *temp_x=0.5*((*temp_x)+ycb);
			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;
		}
	}
	alpha=exp(-sqrt(2.0)/(sigma_spatial));//filter kernel size
	in_=temp;//vertical filtering
	double*ycy,*ypy,*xcy;
	unsigned char*tcy,*tpy;
	memcpy(out[0][0],temp[0][0],sizeof(double)*w*3);
	for(int y=1;y<h;y++)
	{
		tpy=texture[y-1][0];
		tcy=texture[y][0];
		xcy=in_[y][0];
		ypy=out[y-1][0];
		ycy=out[y][0];
		for(int x=0;x<w;x++)
		{
			unsigned char dr=abs((*tcy++)-(*tpy++));
			unsigned char dg=abs((*tcy++)-(*tpy++));
			unsigned char db=abs((*tcy++)-(*tpy++));
			int range_dist=(((dr<<1)+dg+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;
			double inv_alpha_=1-alpha_;
			for(int c=0;c<3;c++) *ycy++=inv_alpha_*(*xcy++)+alpha_*(*ypy++);
		}
	}
	int h1=h-1;
	ycy=temp_2w[0][0];
	ypy=temp_2w[1][0];
	memcpy(ypy,in_[h1][0],sizeof(double)*w*3);
	int k=0; for(int x=0;x<w;x++) for(int c=0;c<3;c++) out[h1][x][c]=0.5*(out[h1][x][c]+ypy[k++]);
	for(int y=h1-1;y>=0;y--)
	{
		tpy=texture[y+1][0];
		tcy=texture[y][0];
		xcy=in_[y][0];
		double*ycy_=ycy;
		double*ypy_=ypy;
		double*out_=out[y][0];
		for(int x=0;x<w;x++)
		{
			unsigned char dr=abs((*tcy++)-(*tpy++));
			unsigned char dg=abs((*tcy++)-(*tpy++));
			unsigned char db=abs((*tcy++)-(*tpy++));
			int range_dist=(((dr<<1)+dg+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;
			double inv_alpha_=1-alpha_;
			for(int c=0;c<3;c++) 
			{
				//ycy[x][c]=inv_alpha_*xcy[x][c]+alpha_*ypy[x][c];
				//out[y][x][c]=0.5*(out[y][x][c]+ycy[x][c]);
				double ycc=inv_alpha_*(*xcy++)+alpha_*(*ypy_++);
				*ycy_++=ycc;
				*out_=0.5*(*out_+ycc); *out_++;
			}
		}
		memcpy(ypy,ycy,sizeof(double)*w*3);
	}
}

void recursive_bilateral_filter_(double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,double***temp,double***temp_2w,double**factor,double**temp_factor,double**temp_factor_2w)
{
	double range_table[UCHAR_MAX+1];//compute a lookup table
	//double inv_sigma_range=1.0/(sigma_range*UCHAR_MAX);
	//double inv_sigma_range=-1.0/(sigma_range);//raplacian
	double inv_sigma_range=-1.0/(2.0*sigma_range*sigma_range);//raplacian
	for(int i=0;i<=UCHAR_MAX;i++)
	{
		//range_table[i]=exp(i*inv_sigma_range);
		range_table[i]=exp((i*i)*inv_sigma_range);
	}

	double alpha=exp(-sqrt(2.0)/(sigma_spatial));//filter kernel size

	unsigned char tpr,tpg,tpb,tcr,tcg,tcb;
	double ypr,ypg,ypb,ycr,ycg,ycb;
	double fp,fc;//factor

	double inv_alpha_=1-alpha;

	double***in_=in;
	for(int y=0;y<h;y++)//horizontal filtering
	{
		double*temp_x=temp[y][0];
		double*in_x=in_[y][0];
		//y previous
		temp_x[0]=ypr=in_x[0]; 
		temp_x[1]=ypg=in_x[1]; 
		temp_x[2]=ypb=in_x[2];

		unsigned char*texture_x=texture[y][0];
		//texture previous
		tpr=texture_x[0];
		tpg=texture_x[1];
		tpb=texture_x[2];

		double*temp_factor_x=temp_factor[y];//factor
		temp_factor_x[0]=fp=1.0; 

		for(int x=1;x<w;x++) //from left to right
		{
			tcr = texture_x[3*x+0];
			tcg = texture_x[3*x+1];
			tcb = texture_x[3*x+2];
			const unsigned char dr=abs(tcr-tpr);
			const unsigned char dg=abs(tcg-tpg);
			const unsigned char db=abs(tcb-tpb);
			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			//int range_dist=(((dr<<1)+dg+db)>>2);
			int range_dist=((dr+(dg<<1)+db)>>2);
			double weight=range_table[range_dist];

			double alpha_=weight*alpha;
			temp_x[3*x+0] = ycr = inv_alpha_*(in_x[3*x+0])+alpha_*ypr;
			temp_x[3*x+1] = ycg = inv_alpha_*(in_x[3*x+1])+alpha_*ypg; 
			temp_x[3*x+2] = ycb = inv_alpha_*(in_x[3*x+2])+alpha_*ypb;//update temp buffer
			ypr=ycr; ypg=ycg; ypb=ycb;// y update

			temp_factor_x[x]=fc=inv_alpha_+alpha_*fp;//factor
			fp=fc;
		}

		temp_x[3*(w-1)+0]=0.5*((temp_x[3*(w-1)+0])+(in_x[3*(w-1)+0])); 
		temp_x[3*(w-1)+1]=0.5*((temp_x[3*(w-1)+1])+(in_x[3*(w-1)+1])); 
		temp_x[3*(w-1)+2]=0.5*((temp_x[3*(w-1)+2])+(in_x[3*(w-1)+2]));

		tpr=texture_x[3*(w-1)+0]; tpg=texture_x[3*(w-1)+1]; tpb=texture_x[3*(w-1)+2];
		ypr=in_x[3*(w-1)+0]; ypg=in_x[3*(w-1)+1]; ypb=in_x[3*(w-1)+2];

		temp_factor_x[w-1]=0.5*((temp_factor_x[w-1])+1.0);//factor
		fp=1;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			tcr = texture_x[3*x+0];
			tcg = texture_x[3*x+1];
			tcb = texture_x[3*x+2];
			const unsigned char dr=abs(tcr-tpr);
			const unsigned char dg=abs(tcg-tpg);
			const unsigned char db=abs(tcb-tpb);

			//int range_dist=(((dr<<1)+dg+db)>>2);
			int range_dist=((dr+(dg<<1)+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;

			ycr=inv_alpha_*(in_x[3*x+0])+alpha_*ypr; 
			ycg=inv_alpha_*(in_x[3*x+1])+alpha_*ypg; 
			ycb=inv_alpha_*(in_x[3*x+2])+alpha_*ypb;
			fc =inv_alpha_              +alpha_*fp;//factor

			temp_x[3*x+0]   =0.5*((   temp_x[3*x+0])+ycr);
			temp_x[3*x+1]   =0.5*((   temp_x[3*x+1])+ycg);
			temp_x[3*x+2]   =0.5*((   temp_x[3*x+2])+ycb);
			temp_factor_x[x]=0.5*((temp_factor_x[x])+fc);

			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;
			fp=fc;
		}
	}

	alpha=exp(-sqrt(2.0)/(sigma_spatial));//filter kernel size
	inv_alpha_=1-alpha;

	in_=temp;//vertical filtering
	double*ycy,*ypy,*xcy;
	unsigned char*tcy,*tpy;
	memcpy(out[0][0],temp[0][0],sizeof(double)*w*3);

	double**in_factor=temp_factor;//factor
	double*ycf,*ypf,*xcf;
	memcpy(factor[0],in_factor[0],sizeof(double)*w);
	for(int y=1;y<h;y++)
	{
		tpy=texture[y-1][0];
		tcy=texture[y][0];
		xcy=in_[y][0];
		ypy=out[y-1][0];
		ycy=out[y][0];

		xcf=&in_factor[y][0];//factor
		ypf=&factor[y-1][0];
		ycf=&factor[y][0];
		for(int x=0;x<w;x++)
		{
			unsigned char dr=abs((tcy[3*x+0])-(tpy[3*x+0]));
			unsigned char dg=abs((tcy[3*x+1])-(tpy[3*x+1]));
			unsigned char db=abs((tcy[3*x+2])-(tpy[3*x+2]));
			//int range_dist=(((dr<<1)+dg+db)>>2);
			int range_dist=((dr+(dg<<1)+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;

			ycy[3*x+0]=inv_alpha_*(xcy[3*x+0])+alpha_*(ypy[3*x+0]);
			ycy[3*x+1]=inv_alpha_*(xcy[3*x+1])+alpha_*(ypy[3*x+1]);
			ycy[3*x+2]=inv_alpha_*(xcy[3*x+2])+alpha_*(ypy[3*x+2]);

			ycf[x]=inv_alpha_*(xcf[x])+alpha_*(ypf[x]);
		}
	}

	int h1=h-1;
	ycf=&temp_factor_2w[0][0];//factor
	ypf=&temp_factor_2w[1][0];
	memcpy(ypf,in_factor[h1],sizeof(double)*w);
	for(int x=0;x<w;x++) factor[h1][x]=0.5*(factor[h1][x]+ypf[x]);

	ycy=temp_2w[0][0];
	ypy=temp_2w[1][0];
	memcpy(ypy,in_[h1][0],sizeof(double)*w*3);
	int k=0; 
	for(int x=0;x<w;x++)
	{
		for(int c=0;c<3;c++)
		{
			out[h1][x][c]=0.5*(out[h1][x][c]+ypy[k++])/factor[h1][x];
		}
	}

	for(int y=h1-1;y>=0;y--)
	{
		tpy=texture[y+1][0];
		tcy=texture[y][0];
		xcy=in_[y][0];
		double*ycy_=ycy;
		double*ypy_=ypy;
		double*out_=out[y][0];

		xcf=&in_factor[y][0];//factor
		double*ycf_=ycf;
		double*ypf_=ypf;
		double*factor_=&factor[y][0];
		for(int x=0;x<w;x++)
		{
			unsigned char dr=abs((tcy[3*x+0])-(tpy[3*x+0]));
			unsigned char dg=abs((tcy[3*x+1])-(tpy[3*x+1]));
			unsigned char db=abs((tcy[3*x+2])-(tpy[3*x+2]));
			//int range_dist=(((dr<<1)+dg+db)>>2);
			int range_dist=((dr+(dg<<1)+db)>>2);
			double weight=range_table[range_dist];
			double alpha_=weight*alpha;

			double fcc=inv_alpha_*(xcf[x])+alpha_*(ypf_[x]);//factor
			ycf_[x]=fcc;
			factor_[x]=0.5*(factor_[x]+fcc); 

			double ycc=inv_alpha_*(xcy[3*x+0])+alpha_*(ypy_[3*x+0]);
			ycy_[3*x+0]=ycc;
			out_[3*x+0]=0.5*(out_[3*x+0]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy[3*x+1])+alpha_*(ypy_[3*x+1]);
			ycy_[3*x+1]=ycc;
			out_[3*x+1]=0.5*(out_[3*x+1]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy[3*x+2])+alpha_*(ypy_[3*x+2]);
			ycy_[3*x+2]=ycc;
			out_[3*x+2]=0.5*(out_[3*x+2]+ycc)/(factor_[x]); 
		}
		memcpy(ypy,ycy,sizeof(double)*w*3);
		memcpy(ypf,ycf,sizeof(double)*w);//factor
	}	
}


void recursive_bilateral_filter_base(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
	if(dest.empty())dest.create(src.size(),src.type());
	int w = src.cols;
	int h = src.rows;

	unsigned char***texture=qx_allocu_3(h,w,3);//allocate memory

	double***image=qx_allocd_3(h,w,3);
	double***image_filtered=qx_allocd_3(h,w,3);

	double***temp=qx_allocd_3(h,w,3);

	double***temp_2=qx_allocd_3(2,w,3);
	double**temp_factor=qx_allocd(h*2+2,w);

	for(int y=0;y<h;y++)
	{
		for(int x=0;x<w;x++) 
		{
			for(int c=0;c<3;c++) 
			{
				image[y][x][c]=(double)src.at<uchar>(y,3*x+c);
				texture[y][x][c]=src.at<uchar>(y,3*x+c);
			}
		}
	}

	recursive_bilateral_filter_(image_filtered, image, texture, sigma_spatial,sigma_range,h,w,temp,temp_2,temp_factor,&(temp_factor[h]),&(temp_factor[h+h]));
	//gradient_domain_recursive_bilateral_filter_(image_filtered, image, texture, sigma_spatial,sigma_range,h,w,temp,temp_2);

	for(int y=0;y<h;y++)
	{
		for(int x=0;x<w;x++) 
		{
			for(int c=0;c<3;c++) 
			{
				dest.at<uchar>(y,3*x+c) = saturate_cast<uchar>(image_filtered[y][x][c]+0.5);
			}
		}
	}

	qx_freed(temp_factor); temp_factor=NULL;
	qx_freeu_3(texture);
	qx_freed_3(image);
	qx_freed_3(image_filtered);
	qx_freed_3(temp);
	qx_freed_3(temp_2);
}

void setColorLUTGaussian(float* lut, float sigma)
{
	const float inv_sigma_range=-1.f/(2.f*sigma*sigma);
	for(int i=0;i<=UCHAR_MAX;i++)
	{
		lut[i]=exp((i*i)*inv_sigma_range);
	}
}
void setColorLUTLaplacian(float* lut, float sigma)
{
	float inv_sigma_range=-1.f/(sigma);//raplacian
	for(int i=0;i<=UCHAR_MAX;i++)
	{
		lut[i]=exp(i*inv_sigma_range);
	}
}


void recursive_bilateral_filter_interleave_sse_before(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
#define distance ((abs(tcr-tpr)+abs(tcg-tpg)+abs(tcb-tpb))*0.333f+0.5f)
	const int w = src.cols;
	const int h = src.rows;

	Mat texture;//texture is joint signal
	Mat bgra;
	cvtColor(src, bgra, COLOR_BGR2BGRA);

	Mat destf; bgra.convertTo(destf, CV_MAKETYPE(CV_32F,bgra.channels()));
	destf.copyTo(texture);

	Mat temp = Mat::zeros(src.size(), CV_MAKETYPE(CV_32F,bgra.channels()));
	Mat tempw = Mat::zeros(Size(w,2), CV_MAKETYPE(CV_32F,bgra.channels()));

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table
	setColorLUTGaussian(range_table,sigma_range);

	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;

	for(int y=0;y<h;y++)//horizontal filtering
	{
		float ypr,ypg,ypb,ycr,ycg,ycb;
		float fp,fc;//factor

		float* in_x = destf.ptr<float>(y);//dest is copy (float) of src;
		float* temp_x=temp.ptr<float>(y);

		//y previous
		temp_x[0]=ypr=in_x[0]; 
		temp_x[1]=ypg=in_x[1]; 
		temp_x[2]=ypb=in_x[2];
		temp_x[3]= fp=1.f; 

		float* texture_x=texture.ptr<float>(y);
		//texture previous
		float tpr=texture_x[0];
		float tpg=texture_x[1];
		float tpb=texture_x[2];

		for(int x=1;x<w;x++) //from left to right
		{				
			float tcr = texture_x[4*x+0];
			float tcg = texture_x[4*x+1];
			float tcb = texture_x[4*x+2];

			float weight=range_table[(int)(distance)];
			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;
			temp_x[4*x+0] = ycr = inv_alpha_*(in_x[4*x+0])+alpha_*ypr;
			temp_x[4*x+1] = ycg = inv_alpha_*(in_x[4*x+1])+alpha_*ypg; 
			temp_x[4*x+2] = ycb = inv_alpha_*(in_x[4*x+2])+alpha_*ypb;//update temp buffer
			temp_x[4*x+3] =  fc = inv_alpha_*(1.f         )+alpha_*fp;//factor

			ypr=ycr; ypg=ycg; ypb=ycb;// y update
			fp=fc;
		}

		temp_x[4*(w-1)+0]=0.5f*((temp_x[4*(w-1)+0])+(in_x[4*(w-1)+0])); 
		temp_x[4*(w-1)+1]=0.5f*((temp_x[4*(w-1)+1])+(in_x[4*(w-1)+1])); 
		temp_x[4*(w-1)+2]=0.5f*((temp_x[4*(w-1)+2])+(in_x[4*(w-1)+2]));

		tpr=texture_x[4*(w-1)+0]; tpg=texture_x[4*(w-1)+1]; tpb=texture_x[4*(w-1)+2];
		ypr=in_x[4*(w-1)+0]; ypg=in_x[4*(w-1)+1]; ypb=in_x[4*(w-1)+2];

		temp_x[4*(w-1)+3]=0.5f*((temp_x[4*(w-1)+3])+1.f);//factor
		fp=1;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			float tcr = texture_x[4*x+0];
			float tcg = texture_x[4*x+1];
			float tcb = texture_x[4*x+2];

			float weight=range_table[(int)(distance)];
			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;

			ycr=inv_alpha_*(in_x[4*x+0])+alpha_*ypr; 
			ycg=inv_alpha_*(in_x[4*x+1])+alpha_*ypg; 
			ycb=inv_alpha_*(in_x[4*x+2])+alpha_*ypb;

			temp_x[4*x+0]=0.5f*((temp_x[4*x+0])+ycr);
			temp_x[4*x+1]=0.5f*((temp_x[4*x+1])+ycg);
			temp_x[4*x+2]=0.5f*((temp_x[4*x+2])+ycb);
			ypr=ycr; ypg=ycg; ypb=ycb;
			fc=inv_alpha_+alpha_*fp;//factor

			temp_x[4*x+3]=0.5f*((temp_x[4*x+3])+fc);
			fp=fc;
		}
	}

	//horizontal filter is end.

	//now, filtering target is Mat temp and dividing factor is temp_factor

	//in = temp
	//in_factor temp_factor
	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;

	float*ycy,*ypy,*xcy;
	float*tcy,*tpy;

	memcpy_float_sse(destf.ptr<float>(0),temp.ptr<float>(0),w*4);//copy from top buffer
	for(int y=1;y<h;y++)
	{
		tpy=texture.ptr<float>(y-1);
		tcy=texture.ptr<float>(y);

		xcy=temp.ptr<float>(y);
		ypy=destf.ptr<float>(y-1);
		ycy=destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			float tcb = tcy[4*x+0];
			float tcg = tcy[4*x+1];
			float tcr = tcy[4*x+2];
			float tpb = tpy[4*x+0];
			float tpg = tpy[4*x+1];
			float tpr = tpy[4*x+2];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycy[4*x+0]=inv_alpha_*(xcy[4*x+0])+alpha_*(ypy[4*x+0]);
			ycy[4*x+1]=inv_alpha_*(xcy[4*x+1])+alpha_*(ypy[4*x+1]);
			ycy[4*x+2]=inv_alpha_*(xcy[4*x+2])+alpha_*(ypy[4*x+2]);
			ycy[4*x+3]=inv_alpha_*(xcy[4*x+3])+alpha_*(ypy[4*x+3]);
		}
	}

	int h1=h-1;


	float* ph1 = temp.ptr<float>(h1);
	float* factor_h1 = destf.ptr<float>(h1);
	for(int i=0;i<w;i++)
	{
		factor_h1[4*i+3]=0.5f*(factor_h1[4*i+3]+ph1[4*i+3]);
	}

	ycy=tempw.ptr<float>(0);
	ypy=tempw.ptr<float>(1);

	memcpy_float_sse(ypy,temp.ptr<float>(h1),w*4);
	//memcpy(ypy,temp.ptr<float>(h1),sizeof(float)*w*3);

	//output final line
	{
		float* out_h1 = destf.ptr<float>(h1);
		for(int x=0;x<w;x++)
		{
			out_h1[4*x+0]=0.5f*(out_h1[4*x+0]+ypy[4*x+0])/factor_h1[x];
			out_h1[4*x+1]=0.5f*(out_h1[4*x+1]+ypy[4*x+1])/factor_h1[x];
			out_h1[4*x+2]=0.5f*(out_h1[4*x+2]+ypy[4*x+2])/factor_h1[x];	
		}
	}

	//
	for(int y=h1-1;y>=0;y--)
	{
		tpy=texture.ptr<float>(y+1);
		tcy=texture.ptr<float>(y);
		xcy=temp.ptr<float>(y);
		float*ycy_=ycy;
		float*ypy_=ypy;
		float*out_= destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			float tcb = tcy[4*x+0];
			float tcg = tcy[4*x+1];
			float tcr = tcy[4*x+2];
			float tpb = tpy[4*x+0];
			float tpg = tpy[4*x+1];
			float tpr = tpy[4*x+2];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			float fcc=inv_alpha_*(xcy[4*x+3])+alpha_*(ypy_[4*x+3]);//factor
			ycy_[4*x+3]=fcc;
			out_[4*x+3]=0.5f*(out_[4*x+3]+fcc); 

			float ycc=inv_alpha_*(xcy[4*x+0])+alpha_*(ypy_[4*x+0]);
			ycy_[4*x+0]=ycc;
			out_[4*x+0]=0.5f*(out_[4*x+0]+ycc)/(out_[4*x+3]); 

			ycc=inv_alpha_*(xcy[4*x+1])+alpha_*(ypy_[4*x+1]);
			ycy_[4*x+1]=ycc;
			out_[4*x+1]=0.5f*(out_[4*x+1]+ycc)/(out_[4*x+3]); 

			ycc=inv_alpha_*(xcy[4*x+2])+alpha_*(ypy_[4*x+2]);
			ycy_[4*x+2]=ycc;
			out_[4*x+2]=0.5f*(out_[4*x+2]+ycc)/(out_[4*x+3]); 
		}
		memcpy_float_sse(ypy,ycy,w*4);
	}	

	destf.convertTo(bgra,src.type(),1.0,0.5);
	cvtColor(bgra,dest,COLOR_BGRA2BGR);
}

void set_one_skip4(Mat& src)
{
	int size = src.size().area();
	float* s = src.ptr<float>(0);
	float* sv = &s[3];
	for(int i=0;i<size;i++)
	{
		*sv =1.0f;
		sv+=4;
	}
}

void recursive_bilateral_filter_interleave_sse(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
	const int w = src.cols;
	const int h = src.rows;

	Mat texture;//texture is joint signal
	Mat bgra;

	cvtColorBGR2BGRA(src, bgra,1);

	Mat destf; bgra.convertTo(destf,  CV_MAKETYPE(CV_32F,bgra.channels()));
	Mat temp = Mat::zeros(src.size(), CV_MAKETYPE(CV_32F,bgra.channels()));
	Mat tempw = Mat::zeros(Size(w,2), CV_MAKETYPE(CV_32F,bgra.channels()));

	destf.copyTo(texture);

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table
	setColorLUTGaussian(range_table,sigma_range);

	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;

	const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

	const __m128 color_normal_factor = _mm_set1_ps(0.33333f);//0.3333f
	const __m128 m05mul = _mm_set1_ps(0.5f);//0.5f
	const __m128 ones = _mm_set1_ps(1.f);//1.f
	const __m128 mspace = _mm_set1_ps(alpha);
	const __m128 minvalpha = _mm_set1_ps(inv_alpha_);

	for(int y=0;y<h;y++)//horizontal filtering
	{
		float* in_x = destf.ptr<float>(y);//destf is now copy (float) of src;
		float* dest_x=temp.ptr<float>(y);//dest

		__m128 myp = _mm_set_ps(1.f,in_x[2],in_x[1],in_x[0]);//y previous
		_mm_storeu_ps(dest_x, myp);//set first pixe;

		float* texture_x=texture.ptr<float>(y);

		__m128 mtp = _mm_load_ps(texture_x);//texture previous
		for(int x=1;x<w;x++) //from left to right
		{
			__m128 mtc = _mm_loadu_ps(texture_x +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtc, mtp),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			mtp = mtc;

			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];

			__m128 malpha = _mm_set1_ps(alpha_);

			__m128 myc = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(in_x+4*x)), _mm_mul_ps(malpha, myp));
			_mm_storeu_ps(dest_x+ 4*x ,myc);
			myp = myc;
		}
		_mm_storeu_ps(dest_x+4*(w-1), _mm_mul_ps(m05mul,_mm_add_ps(_mm_loadu_ps(dest_x+4*(w-1)), _mm_loadu_ps(in_x+4*(w-1)))));

		mtp =_mm_load_ps(texture_x+4*(w-1));
		myp = _mm_load_ps(in_x+4*(w-1)); 

		for(int x=w-2;x>=0;x--) //from right to left
		{
			__m128 mtc = _mm_loadu_ps(texture_x +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtc, mtp),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			mtp = mtc;

			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];
			__m128 malpha = _mm_set1_ps(alpha_);
			__m128 myc = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(in_x+4*x)), _mm_mul_ps(malpha, myp));

			_mm_storeu_ps(dest_x+ 4*x, _mm_mul_ps(m05mul,_mm_add_ps(_mm_load_ps(dest_x + 4*x),myc)));
			myp = myc;//update value
		}
	}

	//horizontal filter is end.
	//now, filtering target is Mat temp and dividing factor is temp_factor

	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;


	memcpy_float_sse(destf.ptr<float>(0),temp.ptr<float>(0),w*4);//copy from top buffer
	for(int y=1;y<h;y++)
	{
		float* tpy=texture.ptr<float>(y-1);
		float* tcy=texture.ptr<float>(y);

		float* xcy=temp.ptr<float>(y);
		float* ypy=destf.ptr<float>(y-1);
		float* ycy=destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			__m128 mtcy = _mm_loadu_ps(tcy +4*x);
			__m128 mtpy = _mm_loadu_ps(tpy +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtcy, mtpy),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];
			__m128 malpha = _mm_set1_ps(alpha_);

			_mm_store_ps(ycy+ 4*x, _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(xcy+4*x)), _mm_mul_ps(malpha, _mm_loadu_ps(ypy+4*x))));
		}
	}

	int h1=h-1;

	float* ycy=tempw.ptr<float>(0);
	float* ypy=tempw.ptr<float>(1);

	//output final line
	memcpy_float_sse(ypy,temp.ptr<float>(h1),w*4);	
	{
		float* out_h1 = destf.ptr<float>(h1);
		for(int x=0;x<w;x++)
		{
			__m128 mv = _mm_mul_ps(m05mul,_mm_add_ps( _mm_loadu_ps(out_h1+4*x), _mm_loadu_ps(ypy+4*x) ));
			__m128  mdiv = _mm_shuffle_ps(mv, mv, 0xFF);

			_mm_storeu_ps(out_h1+4*x, _mm_div_ps(mv, mdiv));	
		}
	}
	//
	for(int y=h1-1;y>=0;y--)
	{
		float* tpy=texture.ptr<float>(y+1);
		float* tcy=texture.ptr<float>(y);
		float* xcy=temp.ptr<float>(y);
		float* ycy_=ycy;
		float* ypy_=ypy;
		float* out_= destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			__m128 mtcy = _mm_loadu_ps(tcy +4*x);
			__m128 mtpy = _mm_loadu_ps(tpy +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtcy, mtpy),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];

			__m128 malpha = _mm_set1_ps(alpha_);

			__m128  mv = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(xcy+4*x)), _mm_mul_ps(malpha,_mm_loadu_ps(ypy_+4*x)));
			_mm_storeu_ps(ycy_+4*x,mv);
			mv = _mm_mul_ps(m05mul,_mm_add_ps(mv,_mm_loadu_ps(out_+4*x)));
			__m128  mdiv = _mm_shuffle_ps(mv, mv, 0xFF);
			_mm_storeu_ps(out_+4*x,_mm_div_ps(mv,mdiv));
		}
		memcpy_float_sse(ypy,ycy,w*4);
	}	

	destf.convertTo(bgra,src.type(),1.0,0.5);
	cvtColorBGRA2BGR(bgra,dest);
}

void recursive_bilateral_filter_interleave_non_sse(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
#define distance ((abs(tcr-tpr)+abs(tcg-tpg)+abs(tcb-tpb))*0.333f+0.5f)
	const int w = src.cols;
	const int h = src.rows;

	Mat texture;//texture is joint signal

	Mat destf; src.convertTo(destf, CV_MAKETYPE(CV_32F,src.channels()));
	destf.copyTo(texture);
	Mat temp = Mat::zeros(src.size(), CV_MAKETYPE(CV_32F,src.channels()));

	Mat temp_factor = Mat::zeros(src.size(), CV_32F);
	Mat dest_factor  = Mat::zeros(src.size(), CV_32F);

	Mat tempw = Mat::zeros(Size(w,2), CV_MAKETYPE(CV_32F,src.channels()));
	Mat temp_factor_buffw = Mat::zeros(Size(w,2),CV_32F);

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table
	setColorLUTGaussian(range_table,sigma_range);

	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;

	for(int y=0;y<h;y++)//horizontal filtering
	{
		float ypr,ypg,ypb,ycr,ycg,ycb;
		float fp,fc;//factor
		float* temp_x=temp.ptr<float>(y);
		float* in_x = destf.ptr<float>(y);//dest is copy (float) of src;

		//y previous
		temp_x[0]=ypr=in_x[0]; 
		temp_x[1]=ypg=in_x[1]; 
		temp_x[2]=ypb=in_x[2];

		float* texture_x=texture.ptr<float>(y);
		//texture previous
		float tpr=texture_x[0];
		float tpg=texture_x[1];
		float tpb=texture_x[2];

		float*temp_factor_x=temp_factor.ptr<float>(y);//factor
		temp_factor_x[0]=fp=1.f; 

		for(int x=1;x<w;x++) //from left to right
		{
			float tcr = texture_x[3*x+0];
			float tcg = texture_x[3*x+1];
			float tcb = texture_x[3*x+2];

			float weight=range_table[(int)(distance)];
			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;
			temp_x[3*x+0] = ycr = inv_alpha_*(in_x[3*x+0])+alpha_*ypr;
			temp_x[3*x+1] = ycg = inv_alpha_*(in_x[3*x+1])+alpha_*ypg; 
			temp_x[3*x+2] = ycb = inv_alpha_*(in_x[3*x+2])+alpha_*ypb;//update temp buffer
			ypr=ycr; ypg=ycg; ypb=ycb;// y update

			temp_factor_x[x]=fc=inv_alpha_+alpha_*fp;//factor
			fp=fc;
		}

		temp_x[3*(w-1)+0]=0.5f*((temp_x[3*(w-1)+0])+(in_x[3*(w-1)+0])); 
		temp_x[3*(w-1)+1]=0.5f*((temp_x[3*(w-1)+1])+(in_x[3*(w-1)+1])); 
		temp_x[3*(w-1)+2]=0.5f*((temp_x[3*(w-1)+2])+(in_x[3*(w-1)+2]));

		tpr=texture_x[3*(w-1)+0]; tpg=texture_x[3*(w-1)+1]; tpb=texture_x[3*(w-1)+2];
		ypr=in_x[3*(w-1)+0]; ypg=in_x[3*(w-1)+1]; ypb=in_x[3*(w-1)+2];

		temp_factor_x[w-1]=0.5f*((temp_factor_x[w-1])+1.f);//factor
		fp=1;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			float tcr = texture_x[3*x+0];
			float tcg = texture_x[3*x+1];
			float tcb = texture_x[3*x+2];

			float weight=range_table[(int)(distance)];
			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;

			ycr=inv_alpha_*(in_x[3*x+0])+alpha_*ypr; 
			ycg=inv_alpha_*(in_x[3*x+1])+alpha_*ypg; 
			ycb=inv_alpha_*(in_x[3*x+2])+alpha_*ypb;

			temp_x[3*x+0]=0.5f*((temp_x[3*x+0])+ycr);
			temp_x[3*x+1]=0.5f*((temp_x[3*x+1])+ycg);
			temp_x[3*x+2]=0.5f*((temp_x[3*x+2])+ycb);
			ypr=ycr; ypg=ycg; ypb=ycb;

			fc=inv_alpha_+alpha_*fp;//factor
			temp_factor_x[x]=0.5f*((temp_factor_x[x])+fc);
			fp=fc;
		}
	}

	//horizontal filter is end.
	//now, filtering target is Mat temp and dividing factor is temp_factor

	//in = temp
	//in_factor temp_factor
	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;

	float*ycy,*ypy,*xcy;
	float*tcy,*tpy;
	float *ycf,*ypf,*xcf;


	memcpy_float_sse(destf.ptr<float>(0),temp.ptr<float>(0),w*3);//copy from top buffer
	memcpy_float_sse(dest_factor.ptr<float>(0),temp_factor.ptr<float>(0),w);//copy from top buffer
	//memcpy(destf.ptr<float>(0),temp.ptr<float>(0),sizeof(float)*w*3);//copy from top buffer
	//memcpy(dest_factor.ptr<float>(0),temp_factor.ptr<float>(0),sizeof(float)*w);//copy from top buffer
	for(int y=1;y<h;y++)
	{
		tpy=texture.ptr<float>(y-1);
		tcy=texture.ptr<float>(y);

		xcy=temp.ptr<float>(y);
		ypy=destf.ptr<float>(y-1);
		ycy=destf.ptr<float>(y);

		xcf=temp_factor.ptr<float>(y);
		ypf=dest_factor.ptr<float>(y-1);
		ycf=dest_factor.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			float tcb = tcy[3*x+0];
			float tcg = tcy[3*x+1];
			float tcr = tcy[3*x+2];
			float tpb = tpy[3*x+0];
			float tpg = tpy[3*x+1];
			float tpr = tpy[3*x+2];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycy[3*x+0]=inv_alpha_*(xcy[3*x+0])+alpha_*(ypy[3*x+0]);
			ycy[3*x+1]=inv_alpha_*(xcy[3*x+1])+alpha_*(ypy[3*x+1]);
			ycy[3*x+2]=inv_alpha_*(xcy[3*x+2])+alpha_*(ypy[3*x+2]);

			ycf[x]=inv_alpha_*(xcf[x])+alpha_*(ypf[x]);
		}
	}

	int h1=h-1;
	ycf=temp_factor_buffw.ptr<float>(0);
	ypf=temp_factor_buffw.ptr<float>(1);

	memcpy_float_sse(ypf,temp_factor.ptr<float>(h1),w);// copy from bottom line.
	//memcpy(ypf,temp_factor.ptr<float>(h1),sizeof(float)*w);// copy from bottom line.

	float* factor_h1 = dest_factor.ptr<float>(h1);
	for(int x=0;x<w;x++)
	{
		factor_h1[x]=0.5f*(factor_h1[x]+ypf[x]);
	}

	ycy=tempw.ptr<float>(0);
	ypy=tempw.ptr<float>(1);

	memcpy_float_sse(ypy,temp.ptr<float>(h1),w*3);
	//memcpy(ypy,temp.ptr<float>(h1),sizeof(float)*w*3);

	//output final line
	{
		float* out_h1 = destf.ptr<float>(h1);
		for(int x=0;x<w;x++)
		{
			out_h1[3*x+0]=0.5f*(out_h1[3*x+0]+ypy[3*x+0])/factor_h1[x];
			out_h1[3*x+1]=0.5f*(out_h1[3*x+1]+ypy[3*x+1])/factor_h1[x];
			out_h1[3*x+2]=0.5f*(out_h1[3*x+2]+ypy[3*x+2])/factor_h1[x];	
		}
	}

	//
	for(int y=h1-1;y>=0;y--)
	{
		tpy=texture.ptr<float>(y+1);
		tcy=texture.ptr<float>(y);
		xcy=temp.ptr<float>(y);
		float*ycy_=ycy;
		float*ypy_=ypy;
		float*out_= destf.ptr<float>(y);

		xcf=temp_factor.ptr<float>(y);//factor
		float* ycf_=ycf;
		float* ypf_=ypf;

		float* factor_=dest_factor.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			float tcb = tcy[3*x+0];
			float tcg = tcy[3*x+1];
			float tcr = tcy[3*x+2];
			float tpb = tpy[3*x+0];
			float tpg = tpy[3*x+1];
			float tpr = tpy[3*x+2];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			float fcc=inv_alpha_*(xcf[x])+alpha_*(ypf_[x]);//factor
			ycf_[x]=fcc;
			factor_[x]=0.5f*(factor_[x]+fcc); 

			float ycc=inv_alpha_*(xcy[3*x+0])+alpha_*(ypy_[3*x+0]);
			ycy_[3*x+0]=ycc;
			out_[3*x+0]=0.5f*(out_[3*x+0]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy[3*x+1])+alpha_*(ypy_[3*x+1]);
			ycy_[3*x+1]=ycc;
			out_[3*x+1]=0.5f*(out_[3*x+1]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy[3*x+2])+alpha_*(ypy_[3*x+2]);
			ycy_[3*x+2]=ycc;
			out_[3*x+2]=0.5f*(out_[3*x+2]+ycc)/(factor_[x]); 
		}
		memcpy_float_sse(ypy,ycy,w*3);
		memcpy_float_sse(ypf,ycf,w);//factor
		//memcpy(ypy,ycy,sizeof(float)*w*3);
		//memcpy(ypf,ycf,sizeof(float)*w);//factor
	}	

	destf.convertTo(dest,src.type(),1.0,0.5);
}

void recursive_bilateral_filter_non_sse(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
#define distance ((abs(tcr-tpr)+abs(tcg-tpg)+abs(tcb-tpb))*0.333f+0.5f)

	Mat plane;
	cvtColorBGR2PLANE(src,plane);
	const int w = src.cols;
	const int h = src.rows;

	const int bptr = 0*h;
	const int gptr = 1*h;
	const int rptr = 2*h;

	Mat srcf; plane.convertTo(srcf, CV_32F);
	Mat texture = srcf;//texture is joint signal

	Mat destf; plane.convertTo(destf, CV_32F);
	Mat temp = Mat::zeros(plane.size(), CV_32F);

	Mat temp_factor = Mat::zeros(src.size(), CV_32F);
	Mat dest_factor  = Mat::zeros(src.size(), CV_32F);

	Mat tempw = Mat::zeros(Size(w,6), CV_32F);
	Mat temp_factor_buffw = Mat::zeros(Size(w,2),CV_32F);

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table
	setColorLUTGaussian(range_table,sigma_range);

	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;

	for(int y=0;y<h;y++)//horizontal filtering
	{	
		float ypr,ypg,ypb,ycr,ycg,ycb;

		float* temp_x_b= temp.ptr<float>(y+bptr);
		float* temp_x_g= temp.ptr<float>(y+gptr);
		float* temp_x_r= temp.ptr<float>(y+rptr);
		float* in_x_b  = destf.ptr<float>(y+bptr);//dest is copy (float) of src;
		float* in_x_g  = destf.ptr<float>(y+gptr);//dest is copy (float) of src;
		float* in_x_r  = destf.ptr<float>(y+rptr);//dest is copy (float) of src;

		//y previous
		temp_x_b[0]=ypb=in_x_b[0]; 
		temp_x_g[0]=ypg=in_x_g[0]; 
		temp_x_r[0]=ypr=in_x_r[0];

		float* texture_x_b=texture.ptr<float>(y+bptr);
		float* texture_x_g=texture.ptr<float>(y+gptr);
		float* texture_x_r=texture.ptr<float>(y+rptr);

		//texture previous
		float tpb= texture_x_b[0];
		float tpg= texture_x_g[0];
		float tpr= texture_x_r[0];

		float fp,fc;//factor
		float*temp_factor_x=temp_factor.ptr<float>(y);//factor
		temp_factor_x[0]=fp=1.f; 

		for(int x=1;x<w;x++) //from left to right
		{
			float tcb = texture_x_b[x];
			float tcg = texture_x_g[x];
			float tcr = texture_x_r[x];

			float weight=range_table[(int)(distance)];

			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;
			temp_x_b[x] = ycb = inv_alpha_*(in_x_b[x])+alpha_*ypb;
			temp_x_g[x] = ycg = inv_alpha_*(in_x_g[x])+alpha_*ypg; 
			temp_x_r[x] = ycr = inv_alpha_*(in_x_r[x])+alpha_*ypr;//update temp buffer
			ypr=ycr; ypg=ycg; ypb=ycb;// y update

			temp_factor_x[x]=fc=inv_alpha_+alpha_*fp;//factor
			fp=fc;
		}

		temp_x_b[w-1]=0.5f*((temp_x_b[w-1])+(in_x_b[w-1])); 
		temp_x_g[w-1]=0.5f*((temp_x_g[w-1])+(in_x_g[w-1])); 
		temp_x_r[w-1]=0.5f*((temp_x_r[w-1])+(in_x_r[w-1]));

		tpb=texture_x_b[w-1]; 
		tpg=texture_x_g[w-1]; 
		tpr=texture_x_r[w-1];

		ypb=in_x_b[w-1]; 
		ypg=in_x_g[w-1]; 
		ypr=in_x_r[w-1];

		temp_factor_x[w-1]=0.5f*((temp_factor_x[w-1])+1.f);//factor
		fp=1.f;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			//texture weight
			float tcb = texture_x_b[x];
			float tcg = texture_x_g[x];
			float tcr = texture_x_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycb=inv_alpha_*(in_x_b[x])+alpha_*ypb; 
			ycg=inv_alpha_*(in_x_g[x])+alpha_*ypg; 
			ycr=inv_alpha_*(in_x_r[x])+alpha_*ypr;

			temp_x_b[x]=0.5f*((temp_x_b[x])+ycb);
			temp_x_g[x]=0.5f*((temp_x_g[x])+ycg);
			temp_x_r[x]=0.5f*((temp_x_r[x])+ycr);
			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;

			//factor
			fc=inv_alpha_+alpha_*fp;//factor
			temp_factor_x[x]=0.5f*((temp_factor_x[x])+fc);
			fp=fc;
		}
	}

	//horizontal filter is end.
	//now, filtering target is Mat temp and dividing factor is temp_factor

	//in = temp
	//in_factor temp_factor
	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;

	float *tcy_b, *tcy_g, *tcy_r;
	float *tpy_b, *tpy_g, *tpy_r;


	float *ycy_b, *ycy_g, *ycy_r;
	float *ypy_b, *ypy_g, *ypy_r; 
	float *xcy_b, *xcy_g, *xcy_r;

	float *ycf,*ypf,*xcf;

	memcpy(destf.ptr<float>(bptr),temp.ptr<float>(bptr),sizeof(float)*w);//copy from top buffer
	memcpy(destf.ptr<float>(gptr),temp.ptr<float>(gptr),sizeof(float)*w);//copy from top buffer
	memcpy(destf.ptr<float>(rptr),temp.ptr<float>(rptr),sizeof(float)*w);//copy from top buffer
	memcpy(dest_factor.ptr<float>(0),temp_factor.ptr<float>(0),sizeof(float)*w);//copy from top buffer

	for(int y=1;y<h;y++)
	{
		tpy_b = texture.ptr<float>(y-1+bptr);
		tpy_g = texture.ptr<float>(y-1+gptr);
		tpy_r = texture.ptr<float>(y-1+rptr);
		tcy_b = texture.ptr<float>(y+bptr);
		tcy_g = texture.ptr<float>(y+gptr);
		tcy_r = texture.ptr<float>(y+rptr);

		xcy_b=temp.ptr<float>(y+bptr);
		xcy_g=temp.ptr<float>(y+gptr);
		xcy_r=temp.ptr<float>(y+rptr);

		ypy_b=destf.ptr<float>(y-1+bptr);
		ypy_g=destf.ptr<float>(y-1+gptr);
		ypy_r=destf.ptr<float>(y-1+rptr);

		ycy_b=destf.ptr<float>(y+bptr);
		ycy_g=destf.ptr<float>(y+gptr);
		ycy_r=destf.ptr<float>(y+rptr);

		xcf=temp_factor.ptr<float>(y);
		ypf=dest_factor.ptr<float>(y-1);
		ycf=dest_factor.ptr<float>(y);


		for(int x=0;x<w;x++)
		{
			float tcb = tcy_b[x];
			float tcg = tcy_g[x];
			float tcr = tcy_r[x];
			float tpb = tpy_b[x];
			float tpg = tpy_g[x];
			float tpr = tpy_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycy_b[x]=inv_alpha_*(xcy_b[x])+alpha_*(ypy_b[x]);
			ycy_g[x]=inv_alpha_*(xcy_g[x])+alpha_*(ypy_g[x]);
			ycy_r[x]=inv_alpha_*(xcy_r[x])+alpha_*(ypy_r[x]);

			ycf[x]=inv_alpha_*(xcf[x])+alpha_*(ypf[x]);
		}
	}

	int h1=h-1;
	ycf=temp_factor_buffw.ptr<float>(0);
	ypf=temp_factor_buffw.ptr<float>(1);

	memcpy(ypf,temp_factor.ptr<float>(h1),sizeof(float)*w);// copy from bottom line.

	float* factor_h1 = dest_factor.ptr<float>(h1);
	for(int x=0;x<w;x++)
	{
		factor_h1[x]=0.5f*(factor_h1[x]+ypf[x]);
	}

	ycy_b=tempw.ptr<float>(0);
	ycy_g=tempw.ptr<float>(1);
	ycy_r=tempw.ptr<float>(2);

	ypy_b=tempw.ptr<float>(3);
	ypy_g=tempw.ptr<float>(4);
	ypy_r=tempw.ptr<float>(5);

	memcpy(ypy_b,temp.ptr<float>(h1+bptr),sizeof(float)*w);
	memcpy(ypy_g,temp.ptr<float>(h1+gptr),sizeof(float)*w);
	memcpy(ypy_r,temp.ptr<float>(h1+rptr),sizeof(float)*w);

	//output final line
	{
		float* out_h1_b = destf.ptr<float>(h1+bptr);
		float* out_h1_g = destf.ptr<float>(h1+gptr);
		float* out_h1_r = destf.ptr<float>(h1+rptr);
		for(int x=0;x<w;x++)
		{
			out_h1_b[x+0]=0.5f*(out_h1_b[x]+ypy_b[x])/factor_h1[x];
			out_h1_g[x+0]=0.5f*(out_h1_g[x]+ypy_g[x])/factor_h1[x];
			out_h1_r[x+0]=0.5f*(out_h1_r[x]+ypy_r[x])/factor_h1[x];
		}
	}

	for(int y=h1-1;y>=0;y--)
	{
		tcy_b = texture.ptr<float>(y+bptr);
		tcy_g = texture.ptr<float>(y+gptr);
		tcy_r = texture.ptr<float>(y+rptr);
		tpy_b = texture.ptr<float>(y+1+bptr);
		tpy_g = texture.ptr<float>(y+1+gptr);
		tpy_r = texture.ptr<float>(y+1+rptr);

		xcy_b=temp.ptr<float>(y+bptr);
		xcy_g=temp.ptr<float>(y+gptr);
		xcy_r=temp.ptr<float>(y+rptr);

		float*ycy_b_=ycy_b;
		float*ycy_g_=ycy_g;
		float*ycy_r_=ycy_r;

		float*ypy_b_=ypy_b;
		float*ypy_g_=ypy_g;
		float*ypy_r_=ypy_r;

		float*out_b_= destf.ptr<float>(y+bptr);
		float*out_g_= destf.ptr<float>(y+gptr);
		float*out_r_= destf.ptr<float>(y+rptr);

		xcf=temp_factor.ptr<float>(y);//factor

		float* ycf_=ycf;
		float* ypf_=ypf;

		float* factor_=dest_factor.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			float tcb = tcy_b[x];
			float tcg = tcy_g[x];
			float tcr = tcy_r[x];
			float tpb = tpy_b[x];
			float tpg = tpy_g[x];
			float tpr = tpy_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			float fcc=inv_alpha_*(xcf[x])+alpha_*(ypf_[x]);//factor
			ycf_[x]=fcc;
			factor_[x]=0.5f*(factor_[x]+fcc); 

			float ycc=inv_alpha_*(xcy_b[x])+alpha_*(ypy_b_[x]);
			ycy_b_[x]=ycc;
			out_b_[x]=0.5f*(out_b_[x]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy_g[x])+alpha_*(ypy_g_[x]);
			ycy_g_[x]=ycc;
			out_g_[x]=0.5f*(out_g_[x]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy_r[x])+alpha_*(ypy_r_[x]);
			ycy_r_[x]=ycc;
			out_r_[x]=0.5f*(out_r_[x]+ycc)/(factor_[x]); 
		}
		memcpy(ypy_b,ycy_b,sizeof(float)*w);
		memcpy(ypy_g,ycy_g,sizeof(float)*w);
		memcpy(ypy_r,ycy_r,sizeof(float)*w);
		memcpy(ypf,ycf,sizeof(float)*w);//factor
	}	

	Mat temp8u;
	destf.convertTo(temp8u,src.type(),1.0,0.5);
	cvtColorPLANE2BGR(temp8u,dest);
}

void recursive_bilateral_filter_sse_vonly(Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
#define distance ((abs(tcr-tpr)+abs(tcg-tpg)+abs(tcb-tpb))*0.333f+0.5f)

	Mat plane;
	cvtColorBGR2PLANE(src,plane);

	const int w = src.cols;
	const int h = src.rows;

	const int bptr = 0*h;
	const int gptr = 1*h;
	const int rptr = 2*h;

	Mat srcf; plane.convertTo(srcf, CV_32F);
	Mat texture = srcf;//texture is joint signal

	Mat destf; plane.convertTo(destf, CV_32F);
	Mat temp = Mat::zeros(plane.size(), CV_32F);

	Mat temp_factor = Mat::zeros(src.size(), CV_32F);
	Mat dest_factor  = Mat::zeros(src.size(), CV_32F);

	Mat tempw = Mat::zeros(Size(w,6), CV_32F);
	Mat temp_factor_buffw = Mat::zeros(Size(w,2),CV_32F);

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table

	//set space and color
	setColorLUTGaussian(range_table,sigma_range);
	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;


	for(int y=0;y<h;y++)//horizontal filtering
	{	
		float ypr,ypg,ypb,ycr,ycg,ycb;

		float* temp_x_b= temp.ptr<float>(y+bptr);
		float* temp_x_g= temp.ptr<float>(y+gptr);
		float* temp_x_r= temp.ptr<float>(y+rptr);
		float* in_x_b  = destf.ptr<float>(y+bptr);//dest is copy (float) of src;
		float* in_x_g  = destf.ptr<float>(y+gptr);//dest is copy (float) of src;
		float* in_x_r  = destf.ptr<float>(y+rptr);//dest is copy (float) of src;

		//y previous
		temp_x_b[0]=ypb=in_x_b[0]; 
		temp_x_g[0]=ypg=in_x_g[0]; 
		temp_x_r[0]=ypr=in_x_r[0];

		float* texture_x_b=texture.ptr<float>(y+bptr);
		float* texture_x_g=texture.ptr<float>(y+gptr);
		float* texture_x_r=texture.ptr<float>(y+rptr);

		//texture previous
		float tpb= texture_x_b[0];
		float tpg= texture_x_g[0];
		float tpr= texture_x_r[0];

		float fp,fc;//factor
		float*temp_factor_x=temp_factor.ptr<float>(y);//factor
		temp_factor_x[0]=fp=1.f; 

		for(int x=1;x<w;x++) //from left to right
		{
			float tcb = texture_x_b[x];
			float tcg = texture_x_g[x];
			float tcr = texture_x_r[x];

			float weight=range_table[(int)(distance)];

			tpr=tcr; tpg=tcg; tpb=tcb;//texture update

			float alpha_=weight*alpha;
			temp_x_b[x] = ycb = inv_alpha_*(in_x_b[x])+alpha_*ypb;
			temp_x_g[x] = ycg = inv_alpha_*(in_x_g[x])+alpha_*ypg; 
			temp_x_r[x] = ycr = inv_alpha_*(in_x_r[x])+alpha_*ypr;//update temp buffer
			ypr=ycr; ypg=ycg; ypb=ycb;// y update

			temp_factor_x[x]=fc=inv_alpha_+alpha_*fp;//factor
			fp=fc;
		}

		temp_x_b[w-1]=0.5f*((temp_x_b[w-1])+(in_x_b[w-1])); 
		temp_x_g[w-1]=0.5f*((temp_x_g[w-1])+(in_x_g[w-1])); 
		temp_x_r[w-1]=0.5f*((temp_x_r[w-1])+(in_x_r[w-1]));

		tpb=texture_x_b[w-1]; 
		tpg=texture_x_g[w-1]; 
		tpr=texture_x_r[w-1];

		ypb=in_x_b[w-1]; 
		ypg=in_x_g[w-1]; 
		ypr=in_x_r[w-1];

		temp_factor_x[w-1]=0.5f*((temp_factor_x[w-1])+1.f);//factor
		fp=1.f;

		for(int x=w-2;x>=0;x--) //from right to left
		{
			//texture weight
			float tcb = texture_x_b[x];
			float tcg = texture_x_g[x];
			float tcr = texture_x_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycb=inv_alpha_*(in_x_b[x])+alpha_*ypb; 
			ycg=inv_alpha_*(in_x_g[x])+alpha_*ypg; 
			ycr=inv_alpha_*(in_x_r[x])+alpha_*ypr;

			temp_x_b[x]=0.5f*((temp_x_b[x])+ycb);
			temp_x_g[x]=0.5f*((temp_x_g[x])+ycg);
			temp_x_r[x]=0.5f*((temp_x_r[x])+ycr);
			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;

			//factor
			fc=inv_alpha_+alpha_*fp;//factor
			temp_factor_x[x]=0.5f*((temp_factor_x[x])+fc);
			fp=fc;
		}
	}

	//horizontal filter is end.
	//now, filtering target is Mat temp and dividing factor is temp_factor

	//in = temp
	//in_factor temp_factor
	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;

	float *tcy_b, *tcy_g, *tcy_r;
	float *tpy_b, *tpy_g, *tpy_r;

	float *ycy_b, *ycy_g, *ycy_r;
	float *ypy_b, *ypy_g, *ypy_r; 
	float *xcy_b, *xcy_g, *xcy_r;

	float *ycf,*ypf,*xcf;

	memcpy_float_sse(destf.ptr<float>(bptr),temp.ptr<float>(bptr),w);//copy from top buffer
	memcpy_float_sse(destf.ptr<float>(gptr),temp.ptr<float>(gptr),w);//copy from top buffer
	memcpy_float_sse(destf.ptr<float>(rptr),temp.ptr<float>(rptr),w);//copy from top buffer
	memcpy_float_sse(dest_factor.ptr<float>(0),temp_factor.ptr<float>(0),w);//copy from top buffer

	for(int y=1;y<h;y++)
	{
		tpy_b = texture.ptr<float>(y-1+bptr);
		tpy_g = texture.ptr<float>(y-1+gptr);
		tpy_r = texture.ptr<float>(y-1+rptr);
		tcy_b = texture.ptr<float>(y+bptr);
		tcy_g = texture.ptr<float>(y+gptr);
		tcy_r = texture.ptr<float>(y+rptr);

		xcy_b=temp.ptr<float>(y+bptr);
		xcy_g=temp.ptr<float>(y+gptr);
		xcy_r=temp.ptr<float>(y+rptr);

		ypy_b=destf.ptr<float>(y-1+bptr);
		ypy_g=destf.ptr<float>(y-1+gptr);
		ypy_r=destf.ptr<float>(y-1+rptr);

		ycy_b=destf.ptr<float>(y+bptr);
		ycy_g=destf.ptr<float>(y+gptr);
		ycy_r=destf.ptr<float>(y+rptr);

		xcf=temp_factor.ptr<float>(y);
		ypf=dest_factor.ptr<float>(y-1);
		ycf=dest_factor.ptr<float>(y);

		int x;
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		int CV_DECL_ALIGNED(16) int_buf[4];
		const __m128 color_normal_factor = _mm_set1_ps(0.33333f);
		//0.5f
		const __m128 m05mul = _mm_set1_ps(0.5f);
		//1.f
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 mspace = _mm_set1_ps(alpha);
		const __m128 minvalpha = _mm_set1_ps(inv_alpha_);
		for(x=0;x<w-4;x+=4)
		{		
			__m128 mtb = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_b+x), _mm_loadu_ps(tpy_b+x)),*(const __m128*)v32f_absmask);
			__m128 mtg = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_g+x), _mm_loadu_ps(tpy_g+x)),*(const __m128*)v32f_absmask);
			__m128 mtr = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_r+x), _mm_loadu_ps(tpy_r+x)),*(const __m128*)v32f_absmask);
			_mm_store_si128((__m128i*)int_buf,_mm_cvtps_epi32(_mm_mul_ps(_mm_add_ps(_mm_add_ps(mtb,mtg),mtr),color_normal_factor)));

			__m128 malpha = _mm_set_ps(range_table[int_buf[3]],range_table[int_buf[2]],range_table[int_buf[1]],range_table[int_buf[0]]);
			malpha = _mm_mul_ps(malpha,mspace);

			_mm_storeu_ps( ycy_b + x, _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(  xcy_b  +x)), _mm_mul_ps(malpha,_mm_loadu_ps(  ypy_b  +x))));
			_mm_storeu_ps( ycy_g + x, _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(  xcy_g  +x)), _mm_mul_ps(malpha,_mm_loadu_ps(  ypy_g  +x))));
			_mm_storeu_ps( ycy_r + x, _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(  xcy_r  +x)), _mm_mul_ps(malpha,_mm_loadu_ps(  ypy_r  +x))));

			_mm_storeu_ps( ycf +x,_mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(  xcf  +x)), _mm_mul_ps(malpha,_mm_loadu_ps(  ypf  +x))));
		}
		for(;x<w;x++)
		{
			float tcb = tcy_b[x];
			float tcg = tcy_g[x];
			float tcr = tcy_r[x];
			float tpb = tpy_b[x];
			float tpg = tpy_g[x];
			float tpr = tpy_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			ycy_b[x]=inv_alpha_*(xcy_b[x])+alpha_*(ypy_b[x]);
			ycy_g[x]=inv_alpha_*(xcy_g[x])+alpha_*(ypy_g[x]);
			ycy_r[x]=inv_alpha_*(xcy_r[x])+alpha_*(ypy_r[x]);

			ycf[x]=inv_alpha_*(xcf[x])+alpha_*(ypf[x]);
		}
	}

	int h1=h-1;
	ycf=temp_factor_buffw.ptr<float>(0);
	ypf=temp_factor_buffw.ptr<float>(1);

	memcpy_float_sse(ypf,temp_factor.ptr<float>(h1),w);// copy from bottom line.

	float* factor_h1 = dest_factor.ptr<float>(h1);
	{
		const __m128 m05mul = _mm_set1_ps(0.5f);
		int x;
		for(x=0;x<=w-4;x+=4)
		{
			_mm_store_ps(factor_h1+x, _mm_mul_ps(m05mul,_mm_add_ps(_mm_loadu_ps(factor_h1+x), _mm_loadu_ps(ypf+x))));
		}
		for(;x<w;x++)
		{
			factor_h1[x]=0.5f*(factor_h1[x]+ypf[x]);
		}
	}

	ycy_b=tempw.ptr<float>(0);
	ycy_g=tempw.ptr<float>(1);
	ycy_r=tempw.ptr<float>(2);

	ypy_b=tempw.ptr<float>(3);
	ypy_g=tempw.ptr<float>(4);
	ypy_r=tempw.ptr<float>(5);

	memcpy_float_sse(ypy_b,temp.ptr<float>(h1+bptr),w);
	memcpy_float_sse(ypy_g,temp.ptr<float>(h1+gptr),w);
	memcpy_float_sse(ypy_r,temp.ptr<float>(h1+rptr),w);

	//output final line
	{
		int x;
		//0.5f
		const __m128 m05mul = _mm_set1_ps(0.5f);

		float* out_h1_b = destf.ptr<float>(h1+bptr);
		float* out_h1_g = destf.ptr<float>(h1+gptr);
		float* out_h1_r = destf.ptr<float>(h1+rptr);
		for(x=0;x<=w-4;x+=4)
		{
			__m128 mdiv = _mm_mul_ps(m05mul,_mm_rcp_ps(_mm_loadu_ps(factor_h1+x)));

			_mm_store_ps(out_h1_b+x, _mm_mul_ps(mdiv,_mm_add_ps(_mm_loadu_ps(out_h1_b+x), _mm_loadu_ps(ypy_b+x))));
			_mm_store_ps(out_h1_g+x, _mm_mul_ps(mdiv,_mm_add_ps(_mm_loadu_ps(out_h1_g+x), _mm_loadu_ps(ypy_g+x))));
			_mm_store_ps(out_h1_r+x, _mm_mul_ps(mdiv,_mm_add_ps(_mm_loadu_ps(out_h1_r+x), _mm_loadu_ps(ypy_r+x))));
		}
		for(;x<w;x++)
		{
			out_h1_b[x]=0.5f*(out_h1_b[x]+ypy_b[x])/factor_h1[x];
			out_h1_g[x]=0.5f*(out_h1_g[x]+ypy_g[x])/factor_h1[x];
			out_h1_r[x]=0.5f*(out_h1_r[x]+ypy_r[x])/factor_h1[x];
		}
	}

	for(int y=h1-1;y>=0;y--)
	{
		tcy_b = texture.ptr<float>(y+bptr);
		tcy_g = texture.ptr<float>(y+gptr);
		tcy_r = texture.ptr<float>(y+rptr);
		tpy_b = texture.ptr<float>(y+1+bptr);
		tpy_g = texture.ptr<float>(y+1+gptr);
		tpy_r = texture.ptr<float>(y+1+rptr);

		xcy_b=temp.ptr<float>(y+bptr);
		xcy_g=temp.ptr<float>(y+gptr);
		xcy_r=temp.ptr<float>(y+rptr);

		float*ycy_b_=ycy_b;
		float*ycy_g_=ycy_g;
		float*ycy_r_=ycy_r;

		float*ypy_b_=ypy_b;
		float*ypy_g_=ypy_g;
		float*ypy_r_=ypy_r;

		float*out_b_= destf.ptr<float>(y+bptr);
		float*out_g_= destf.ptr<float>(y+gptr);
		float*out_r_= destf.ptr<float>(y+rptr);

		xcf=temp_factor.ptr<float>(y);//factor

		float* ycf_=ycf;
		float* ypf_=ypf;

		float* factor_=dest_factor.ptr<float>(y);
		int x;
		const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		int CV_DECL_ALIGNED(16) int_buf[4];
		const __m128 color_normal_factor = _mm_set1_ps(0.33333f);
		//0.5f
		const __m128 m05mul = _mm_set1_ps(0.5f);
		//1.f
		const __m128 ones = _mm_set1_ps(1.f);
		const __m128 mspace = _mm_set1_ps(alpha);
		const __m128 minvalpha = _mm_set1_ps(inv_alpha_);
		for(x=0;x<=w-4;x+=4)
		{		
			__m128 mtb = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_b+x), _mm_loadu_ps(tpy_b+x)),*(const __m128*)v32f_absmask);
			__m128 mtg = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_g+x), _mm_loadu_ps(tpy_g+x)),*(const __m128*)v32f_absmask);
			__m128 mtr = _mm_and_ps(_mm_sub_ps(_mm_loadu_ps(tcy_r+x), _mm_loadu_ps(tpy_r+x)),*(const __m128*)v32f_absmask);
			_mm_store_si128((__m128i*)int_buf,_mm_cvtps_epi32(_mm_mul_ps(_mm_add_ps(_mm_add_ps(mtb,mtg),mtr),color_normal_factor)));

			__m128 malpha = _mm_set_ps(range_table[int_buf[3]],range_table[int_buf[2]],range_table[int_buf[1]],range_table[int_buf[0]]);
			malpha = _mm_mul_ps(malpha,mspace);

			__m128 mtemp = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(  xcf  +x)), _mm_mul_ps(malpha,_mm_loadu_ps(  ypf  +x)));
			_mm_storeu_ps( ycf +x,mtemp);
			mtemp=_mm_mul_ps(_mm_add_ps(_mm_loadu_ps( factor_ +x),mtemp),m05mul);
			_mm_storeu_ps( factor_ +x,mtemp);
			//__m128 mdiv = _mm_mul_ps(_mm_rcp_ps(mtemp),m05mul);
			__m128 mdiv = _mm_mul_ps(_mm_div_ps(ones,mtemp),m05mul);

			mtemp = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(   xcy_b   +x)), _mm_mul_ps(malpha,_mm_loadu_ps(   ypy_b   +x)));
			_mm_storeu_ps(   ycy_b   +x,mtemp);
			_mm_storeu_ps(   out_b_  +x,_mm_mul_ps(_mm_add_ps(_mm_loadu_ps(   out_b_    +x),mtemp),mdiv));

			mtemp = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(   xcy_g   +x)), _mm_mul_ps(malpha,_mm_loadu_ps(   ypy_g   +x)));
			_mm_storeu_ps(   ycy_g   +x,mtemp);
			_mm_storeu_ps(   out_g_  +x,_mm_mul_ps(_mm_add_ps(_mm_loadu_ps(   out_g_    +x),mtemp),mdiv));

			mtemp = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(   xcy_r   +x)), _mm_mul_ps(malpha,_mm_loadu_ps(   ypy_r   +x)));
			_mm_storeu_ps(   ycy_r   +x,mtemp);
			_mm_storeu_ps(   out_r_  +x,_mm_mul_ps(_mm_add_ps(_mm_loadu_ps(   out_r_    +x),mtemp),mdiv));
		}
		for(;x<w;x++)
		{
			float tcb = tcy_b[x];
			float tcg = tcy_g[x];
			float tcr = tcy_r[x];
			float tpb = tpy_b[x];
			float tpg = tpy_g[x];
			float tpr = tpy_r[x];
			float weight=range_table[(int)(distance)];

			float alpha_=weight*alpha;

			float fcc=inv_alpha_*(xcf[x])+alpha_*(ypf_[x]);//factor
			ycf_[x]=fcc;
			factor_[x]=0.5f*(factor_[x]+fcc); 

			float ycc=inv_alpha_*(xcy_b[x])+alpha_*(ypy_b_[x]);
			ycy_b_[x]=ycc;
			out_b_[x]=0.5f*(out_b_[x]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy_g[x])+alpha_*(ypy_g_[x]);
			ycy_g_[x]=ycc;
			out_g_[x]=0.5f*(out_g_[x]+ycc)/(factor_[x]); 

			ycc=inv_alpha_*(xcy_r[x])+alpha_*(ypy_r_[x]);
			ycy_r_[x]=ycc;
			out_r_[x]=0.5f*(out_r_[x]+ycc)/(factor_[x]); 
		}
		memcpy_float_sse(ypy_b,ycy_b,w);
		memcpy_float_sse(ypy_g,ycy_g,w);
		memcpy_float_sse(ypy_r,ycy_r,w);
		memcpy_float_sse(ypf,ycf,w);//factor
	}	

	Mat temp8u;
	destf.convertTo(temp8u,src.type(),1.0,0.5);
	cvtColorPLANE2BGR(temp8u,dest);
}

void recursiveBilateralFilter(Mat& src, Mat& dest, float sigma_range, float sigma_spatial, int method)
{
	//sse for vertical parallerization
	if(method==0) recursive_bilateral_filter_sse_vonly(src, dest, sigma_range, sigma_spatial);
	//sse for color image vectorization (homoginious coordinate)
	if(method==1) recursive_bilateral_filter_interleave_sse(src, dest, sigma_range, sigma_spatial);
	//base for 0
	if(method==2) recursive_bilateral_filter_non_sse(src, dest, sigma_range, sigma_spatial);
	//base for 1
	if(method==3) recursive_bilateral_filter_interleave_non_sse(src, dest, sigma_range, sigma_spatial);
	//base code of this file
	if(method==4) recursive_bilateral_filter_base(src, dest, sigma_range, sigma_spatial);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// for class RecursiveBilateralFilter

void  RecursiveBilateralFilter::setColorLUTGaussian(float* lut, float sigma)
{
	const float inv_sigma_range=-1.f/(2.f*sigma*sigma);
	for(int i=0;i<=UCHAR_MAX;i++)
	{
		lut[i]=exp((i*i)*inv_sigma_range);
	}
}

void  RecursiveBilateralFilter::setColorLUTLaplacian(float* lut, float sigma)
{
	float inv_sigma_range=-1.f/(sigma);//raplacian
	for(int i=0;i<=UCHAR_MAX;i++)
	{
		lut[i]=exp(i*inv_sigma_range);
	}
}

void  RecursiveBilateralFilter::init(Size size_)
{
	size = size_;
	bgra = Mat::zeros(size,CV_8U);

	texture= Mat::zeros(size,CV_32FC4);
	destf = Mat::zeros(size,CV_32FC4);
	temp = Mat::zeros(size,CV_32FC4);
	tempw = Mat::zeros(Size(size.width,2),CV_32FC4);
}

RecursiveBilateralFilter::RecursiveBilateralFilter(Size size)
{
	init(size);
}

RecursiveBilateralFilter::RecursiveBilateralFilter()
{
	;
}

RecursiveBilateralFilter::~RecursiveBilateralFilter()
{
	;
}

void RecursiveBilateralFilter::operator()(const Mat& src, const Mat& guide, Mat& dest, float sigma_range, float sigma_spatial)
{
	if(src.size().area()!=size.area())init(src.size());

	const int w = src.cols;
	const int h = src.rows;

	Mat texture;//texture is joint signal
	Mat bgra;

	cvtColorBGR2BGRA(src, bgra,1);
	bgra.convertTo(destf,CV_32F);
	if(src.data==guide.data)
	{
		destf.copyTo(texture);
	}
	else
	{
		cvtColorBGR2BGRA(guide, bgra,1);
		bgra.convertTo(texture,CV_32F);
	}

	float CV_DECL_ALIGNED(16) range_table[UCHAR_MAX+1];//compute a lookup table
	setColorLUTGaussian(range_table,sigma_range);

	float alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size
	float inv_alpha_=1.f-alpha;

	const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

	const __m128 color_normal_factor = _mm_set1_ps(0.33333f);//0.3333f
	const __m128 m05mul = _mm_set1_ps(0.5f);//0.5f
	const __m128 ones = _mm_set1_ps(1.f);//1.f
	const __m128 mspace = _mm_set1_ps(alpha);
	const __m128 minvalpha = _mm_set1_ps(inv_alpha_);

	for(int y=0;y<h;y++)//horizontal filtering
	{
		float* in_x = destf.ptr<float>(y);//destf is now copy (float) of src;
		float* dest_x=temp.ptr<float>(y);//dest

		__m128 myp = _mm_set_ps(1.f,in_x[2],in_x[1],in_x[0]);//y previous
		_mm_storeu_ps(dest_x, myp);//set first pixe;

		float* texture_x=texture.ptr<float>(y);

		__m128 mtp = _mm_load_ps(texture_x);//texture previous
		for(int x=1;x<w;x++) //from left to right
		{
			__m128 mtc = _mm_loadu_ps(texture_x +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtc, mtp),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			mtp = mtc;

			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];

			__m128 malpha = _mm_set1_ps(alpha_);

			__m128 myc = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(in_x+4*x)), _mm_mul_ps(malpha, myp));
			_mm_storeu_ps(dest_x+ 4*x ,myc);
			myp = myc;
		}
		_mm_storeu_ps(dest_x+4*(w-1), _mm_mul_ps(m05mul,_mm_add_ps(_mm_loadu_ps(dest_x+4*(w-1)), _mm_loadu_ps(in_x+4*(w-1)))));

		mtp =_mm_load_ps(texture_x+4*(w-1));
		myp = _mm_load_ps(in_x+4*(w-1)); 

		for(int x=w-2;x>=0;x--) //from right to left
		{
			__m128 mtc = _mm_loadu_ps(texture_x +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtc, mtp),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			mtp = mtc;

			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];
			__m128 malpha = _mm_set1_ps(alpha_);
			__m128 myc = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(in_x+4*x)), _mm_mul_ps(malpha, myp));

			_mm_storeu_ps(dest_x+ 4*x, _mm_mul_ps(m05mul,_mm_add_ps(_mm_load_ps(dest_x + 4*x),myc)));
			myp = myc;//update value
		}
	}

	//horizontal filter is end.
	//now, filtering target is Mat temp and dividing factor is temp_factor

	alpha=exp(-sqrt(2.f)/(sigma_spatial));//filter kernel size virtical
	inv_alpha_=1.f-alpha;

	memcpy_float_sse(destf.ptr<float>(0),temp.ptr<float>(0),w*4);//copy from top buffer
	for(int y=1;y<h;y++)
	{
		float* tpy=texture.ptr<float>(y-1);
		float* tcy=texture.ptr<float>(y);

		float* xcy=temp.ptr<float>(y);
		float* ypy=destf.ptr<float>(y-1);
		float* ycy=destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			__m128 mtcy = _mm_loadu_ps(tcy +4*x);
			__m128 mtpy = _mm_loadu_ps(tpy +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtcy, mtpy),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];
			__m128 malpha = _mm_set1_ps(alpha_);

			_mm_store_ps(ycy+ 4*x, _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(xcy+4*x)), _mm_mul_ps(malpha, _mm_loadu_ps(ypy+4*x))));
		}
	}

	int h1=h-1;

	float* ycy=tempw.ptr<float>(0);
	float* ypy=tempw.ptr<float>(1);

	//output final line
	memcpy_float_sse(ypy,temp.ptr<float>(h1),w*4);	
	{
		float* out_h1 = destf.ptr<float>(h1);
		for(int x=0;x<w;x++)
		{
			__m128 mv = _mm_mul_ps(m05mul,_mm_add_ps( _mm_loadu_ps(out_h1+4*x), _mm_loadu_ps(ypy+4*x) ));
			__m128  mdiv = _mm_shuffle_ps(mv, mv, 0xFF);

			_mm_storeu_ps(out_h1+4*x, _mm_div_ps(mv, mdiv));	
		}
	}
	//
	for(int y=h1-1;y>=0;y--)
	{
		float* tpy=texture.ptr<float>(y+1);
		float* tcy=texture.ptr<float>(y);
		float* xcy=temp.ptr<float>(y);
		float* ycy_=ycy;
		float* ypy_=ypy;
		float* out_= destf.ptr<float>(y);

		for(int x=0;x<w;x++)
		{
			__m128 mtcy = _mm_loadu_ps(tcy +4*x);
			__m128 mtpy = _mm_loadu_ps(tpy +4*x);
			__m128 a = _mm_and_ps(_mm_sub_ps(mtcy, mtpy),*(const __m128*)v32f_absmask);
			a = _mm_hadd_ps(a, a);
			a = _mm_hadd_ps(a, a);
			float alpha_= alpha*range_table[(int)(a.m128_f32[0]*0.3333f+0.5f)];

			__m128 malpha = _mm_set1_ps(alpha_);

			__m128  mv = _mm_add_ps(_mm_mul_ps(minvalpha,_mm_loadu_ps(xcy+4*x)), _mm_mul_ps(malpha,_mm_loadu_ps(ypy_+4*x)));
			_mm_storeu_ps(ycy_+4*x,mv);
			mv = _mm_mul_ps(m05mul,_mm_add_ps(mv,_mm_loadu_ps(out_+4*x)));
			__m128  mdiv = _mm_shuffle_ps(mv, mv, 0xFF);
			_mm_storeu_ps(out_+4*x,_mm_div_ps(mv,mdiv));
		}
		memcpy_float_sse(ypy,ycy,w*4);
	}	

	destf.convertTo(bgra,src.type(),1.0,0.5);
	cvtColorBGRA2BGR(bgra,dest);
}

void RecursiveBilateralFilter::operator()(const Mat& src, Mat& dest, float sigma_range, float sigma_spatial)
{
	operator()(src,src,dest,sigma_range,sigma_spatial);
}
