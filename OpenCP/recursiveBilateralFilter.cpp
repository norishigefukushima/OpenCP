#include "opencp.hpp"

//Qingxiong Yang, Recursive Bilateral Filtering, European Conference on Computer Vision (ECCV) 2012, 399-413.
//http://www.cs.cityu.edu.hk/~qiyang/publications/eccv-12/


#define QX_DEF_PADDING					10


inline void qx_sort_increase_using_histogram(int*id,unsigned char*image,int len)
{
	int histogram[UCHAR_MAX+1];
	int nr_bin=UCHAR_MAX+1;
	memset(histogram,0,sizeof(int)*nr_bin);
	for(int i=0;i<len;i++)
	{
		histogram[image[i]]++;
	}
	int nr_hitted_prev=histogram[0];
	histogram[0]=0;
	for(int k=1;k<nr_bin;k++)
	{
		int nr_hitted=histogram[k];
		histogram[k]=nr_hitted_prev+histogram[k-1];
		nr_hitted_prev=nr_hitted;
	}
	for(int i=0;i<len;i++)
	{
		unsigned char dist=image[i];
		int index=histogram[dist]++;
		id[index]=i;
	}
}
inline double *get_color_weighted_table(double sigma_range,int len)
{
	double *table_color,*color_table_x; int y;
	table_color=new double [len];
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*sigma_range*sigma_range));
	return(table_color);
}
inline void color_weighted_table_update(double *table_color,double dist_color,int len)
{
	double *color_table_x; int y;
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*dist_color*dist_color));
}

inline void vec_min_val(float &min_val,float *in,int len)
{
	min_val=in[0];
	for(int i=1;i<len;i++) if(in[i]<min_val) min_val=in[i];	
}
inline void vec_min_val(unsigned char &min_val,unsigned char *in,int len)
{
	min_val=in[0];
	for(int i=1;i<len;i++) if(in[i]<min_val) min_val=in[i];	
}
inline void vec_max_val(float &max_val,float *in,int len)
{
	max_val=in[0];
	for(int i=1;i<len;i++) if(in[i]>max_val) max_val=in[i];	
}
inline void vec_max_val(unsigned char &max_val,unsigned char *in,int len)
{
	max_val=in[0];
	for(int i=1;i<len;i++) if(in[i]>max_val) max_val=in[i];	
}
inline void down_sample_1(unsigned char **out,unsigned char **in,int h,int w,int scale_exp)
{
	int y,x; int ho,wo; unsigned char *out_y,*in_x;
	ho=(h>>scale_exp); wo=(w>>scale_exp); 
	for(y=0;y<ho;y++)
	{
		out_y=&out[y][0]; in_x=in[y<<scale_exp];
		for(x=0;x<wo;x++) *out_y++=in_x[x<<scale_exp];
	}
}
inline void down_sample_1(float**out,float**in,int h,int w,int scale_exp)
{
	int y,x; int ho,wo; float *out_y,*in_x;
	ho=(h>>scale_exp); wo=(w>>scale_exp); 
	for(y=0;y<ho;y++)
	{
		out_y=&out[y][0]; in_x=in[y<<scale_exp];
		for(x=0;x<wo;x++) *out_y++=in_x[x<<scale_exp];
	}
}
inline double qx_linear_interpolate_xy(double **image,double x,double y,int h,int w)
{
	int x0,xt,y0,yt; double dx,dy,dx1,dy1,d00,d0t,dt0,dtt;
	x0=int(x); xt=min(x0+1,w-1); y0=int(y); yt=min(y0+1,h-1);
	dx=x-x0; dy=y-y0; dx1=1-dx; dy1=1-dy; d00=dx1*dy1; d0t=dx*dy1; dt0=dx1*dy; dtt=dx*dy;
	return(d00*image[y0][x0]+d0t*image[y0][xt]+dt0*image[yt][x0]+dtt*image[yt][xt]);
}
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

/*
inline int qx_compute_color_difference(unsigned char a[3],unsigned char b[3])
{
	unsigned char dr=abs(a[0]-b[0]);
	unsigned char dg=abs(a[1]-b[1]);
	unsigned char db=abs(a[2]-b[2]);
	return(((dr<<1)+dg+db)>>2);
}*/


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
	double*ycy,*ypy,*xcy,*xpy;
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

			temp_x[3*x+0]=0.5*((temp_x[3*x+0])+ycr);
			temp_x[3*x+1]=0.5*((temp_x[3*x+1])+ycg);
			temp_x[3*x+2]=0.5*((temp_x[3*x+2])+ycb);
			tpr=tcr; tpg=tcg; tpb=tcb;
			ypr=ycr; ypg=ycg; ypb=ycb;

			fc=inv_alpha_+alpha_*fp;//factor
			temp_factor_x[x]=0.5*((temp_factor_x[x])+fc);
			fp=fc;
		}
	}

	alpha=exp(-sqrt(2.0)/(sigma_spatial));//filter kernel size
	inv_alpha_=1-alpha;

	in_=temp;//vertical filtering
	double*ycy,*ypy,*xcy,*xpy;
	unsigned char*tcy,*tpy;
	memcpy(out[0][0],temp[0][0],sizeof(double)*w*3);
	
	double**in_factor=temp_factor;//factor
	double*ycf,*ypf,*xcf,*xpf;
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


void recursiveBilateralFilter(Mat& src, Mat& dest, double sigma_range, double sigma_spatial)
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