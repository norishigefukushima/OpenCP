#include <stdio.h>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>

#include "imq.h"

const double MM_PI=3.14159265358979323846;

int my_pow(int a, int b){

int rez = 1;
for(int i = 0; i < b; i++) rez *= a;
return rez;

}
 
int my_lgamma(int k){

	int rez = 1;
	if (k < 0) return 0;
	switch(k) {

		case 0: return 0;
		case 1: return 1;
		case 2: return 2;
		default: for(int i=2; i<=k; ++i) rez *=i;
				 break;
	}
return rez;
}

float steera(float theta, float *B1, float *B2, float *B3, float *B4, int w, int h, int x, int y)
  {
     double k1a, k1b, k1c, k1d, outsteera;
     int index;
     k1a = (2.0 * cos((theta - 0.0 * MM_PI / 4.0)) + 2.0 * cos(3.0 * (theta - 0.0 * MM_PI / 4.0))) / 4.0;
     k1b = (2.0 * cos((theta - 1.0 * MM_PI / 4.0)) + 2.0 * cos(3.0 * (theta - 1.0 * MM_PI / 4.0))) / 4.0;
     k1c = (2.0 * cos((theta - 2.0 * MM_PI / 4.0)) + 2.0 * cos(3.0 * (theta - 2.0 * MM_PI / 4.0))) / 4.0;
     k1d = (2.0 * cos((theta - 3.0 * MM_PI / 4.0)) + 2.0 * cos(3.0 * (theta - 3.0 * MM_PI / 4.0))) / 4.0;
     index = x + w * y;
     outsteera = k1a * B1[index] + k1b * B2[index] + k1c * B3[index] + k1d * B4[index];
     return (float)outsteera;
  }

float steerb(float theta, float *hB1, float *hB2, float *hB3, float *hB4, float *hB5, 
	int w, int h, int x, int y)
  {
     double k1a, k1b, k1c, k1d, k1e, outsteerb;
     int index;
     k1a = (1.0 + 2.0 * cos(2.0 * (theta - 0.0 * MM_PI / 5.0)) + 2.0 * cos(4.0 * (theta - 0.0 * MM_PI / 5.0))) / 5.0;
     k1b = (1.0 + 2.0 * cos(2.0 * (theta - 1.0 * MM_PI / 5.0)) + 2.0 * cos(4.0 * (theta - 1.0 * MM_PI / 5.0))) / 5.0;
     k1c = (1.0 + 2.0 * cos(2.0 * (theta - 2.0 * MM_PI / 5.0)) + 2.0 * cos(4.0 * (theta - 2.0 * MM_PI / 5.0))) / 5.0;
     k1d = (1.0 + 2.0 * cos(2.0 * (theta - 3.0 * MM_PI / 5.0)) + 2.0 * cos(4.0 * (theta - 3.0 * MM_PI / 5.0))) / 5.0;
     k1e = (1.0 + 2.0 * cos(2.0 * (theta - 4.0 * MM_PI / 5.0)) + 2.0 * cos(4.0 * (theta - 4.0 * MM_PI / 5.0))) / 5.0;
     index = x + w * y;
     outsteerb = k1a * hB1[index] + k1b * hB2[index] + k1c * hB3[index] + k1d * hB4[index] + k1e * hB5[index];
     return (float)outsteerb;
  }	
  	
//Generate bandpass image from FT with one transfer function mask, real output
//============================================================================
  
int fourier2spatialband1(int w, int h, float *otf1, float *BP, fftwf_complex *conv, fftwf_complex *fim, fftwf_complex *ftmp)  
   {
      fftwf_plan p;	
      int i, j;
      for (j = 0; j < h; j++)
        {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j;
              conv[index][0] = fim[index][0] / (w * h) * otf1[index];
	      conv[index][1] = fim[index][1] / (w * h) * otf1[index];
            }
	}       
      p = fftwf_plan_dft_2d(h, w, conv, ftmp, 1, FFTW_ESTIMATE);
      fftwf_execute(p); 
      fftwf_destroy_plan(p);
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BP[i + w * j] = ftmp[i + w * j][0];}}
      return 1;
   }
   
//Generate bandpass image from FT with two transfer function masks, complex output
//================================================================================
   
int fourier2spatialband2(int w, int h, float *otf1, float *otf2, float *BPr, float *BPc, fftwf_complex *conv, fftwf_complex *fim, fftwf_complex *ftmp)  
   {
      fftwf_plan p;	
      int i, j;
      for (j = 0; j < h; j++)
        {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j;
              conv[index][0] = fim[index][0] / (w * h) * otf1[index] * otf2[index];
	      conv[index][1] = fim[index][1] / (w * h) * otf1[index] * otf2[index];
            }
	}       
      p = fftwf_plan_dft_2d(h, w, conv, ftmp, 1, FFTW_ESTIMATE);
      fftwf_execute(p);  
      fftwf_destroy_plan(p);
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BPr[i + w * j] = ftmp[i + w * j][0];}}
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BPc[i + w * j] = ftmp[i + w * j][1];}}
      return 1;
   }   
//Generate bandpass image from FT with two transfer function masks, real output
//=============================================================================
  
int fourier2spatialband2a(int w, int h, float *otf1, float *otf2, float *BP, fftwf_complex *conv, fftwf_complex *fim, fftwf_complex *ftmp)  
   {
      fftwf_plan p;	
      int i, j;
      float maxr, minr, maxc, minc; 
      maxr = 0.0; minr = 1000; maxc = 0.0; minc = 1000.0;
      for (j = 0; j < h; j++)
        {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j;
              conv[index][0] = fim[index][0] / (w * h) * otf1[index] * otf2[index];
	      conv[index][1] = fim[index][1] / (w * h) * otf1[index] * otf2[index];
            }
	}       
      p = fftwf_plan_dft_2d(h, w, conv, ftmp, 1, FFTW_ESTIMATE);
      fftwf_execute(p);  
      fftwf_destroy_plan(p);
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BP[i + w * j] = ftmp[i + w * j][0];}}
      return 1;
   }  
    
//Generate bandpass image from FT with three transfer function masks, complex output
//==================================================================================
   
int fourier2spatialband3(int w, int h, float *otf1, float *otf2, float *otf3, float *BPr, float *BPc, fftwf_complex *conv, fftwf_complex *fim, fftwf_complex *ftmp)  
   {
      fftwf_plan p;	
      int i, j;
      for (j = 0; j < h; j++)
        {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j;
              conv[index][0] = fim[index][0] / (w * h) * otf1[index] * otf2[index] * otf3[index];
	      conv[index][1] = fim[index][1] / (w * h) * otf1[index] * otf2[index] * otf3[index];
            }
	}       
      p = fftwf_plan_dft_2d(h, w, conv, ftmp, 1, FFTW_ESTIMATE);
      fftwf_execute(p);  
      fftwf_destroy_plan(p);
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BPr[i + w * j] = ftmp[i + w * j][0];}}
      for (j = 0; j < h; j++) {for (i = 0; i < w; i++) {BPc[i + w * j] = ftmp[i + w * j][1];}}
      return 1;
   }
   
//Reconstruction substep, input real and complex part of oriented 
//subbands for fourier transform for adding the different subbands
//================================================================
   
int reconststep(fftwf_complex *fourtmp, fftwf_complex *fourBP, float *BPr, float *BPc, int w, int h) 
  {
    fftwf_plan p;
    int i, j;	
    for (j = 0; j < h; j ++) 
      {
        for (i = 0; i < w; i++) 
	  {
	    fourtmp[i + w * j][0] = BPr[i + w * j]; 
	    fourtmp[i + w * j][1] = BPc[i + w * j];
	  }
      }
    p = fftwf_plan_dft_2d(h, w, fourtmp, fourBP, -1, FFTW_ESTIMATE);  
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    return 1;
  }
   
//Reconstruction substep, input with only real part subbands (LP
//and HP) subbands for fourier transform for adding the different subbands
//========================================================================
  
int reconststepa(fftwf_complex *fourtmp, fftwf_complex *fourBP, float *BP, int w, int h) 
  {
    fftwf_plan p;
    int i, j;	
    for (j = 0; j < h; j ++) 
      {
        for (i = 0; i < w; i++) 
	  {
	    fourtmp[i + w * j][0] = BP[i + w * j]; 
	    fourtmp[i + w * j][1] = 0.0;
	  }
      }
    p = fftwf_plan_dft_2d(h, w, fourtmp, fourBP, -1, FFTW_ESTIMATE);  
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    return 1;
  }
  
int genHPfilter(float *HP, int w, int h, double x1, double x2)
  {
     double r, x;
     int i, j, k, l;
     
     for (j = 0; j < h; j++)
       {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j;
	      
	      if (i <= w/2) {k = i;}
	      if (i >  w/2) {k = i - w;}
	       
	      if (j <= h/2) {l = j;}
	      if (j >  h/2) {l = j - h;}
	      
	      r = sqrt((float)(k*k + l*l));
				
              if (r < x1) {HP[index] = 0.0;}
	      if (r > x2) {HP[index] = 1.0;}
	      if ((r >= x1) && (r <= x2)) 
	        {
		  x = MM_PI / 4.0 * (1.0 + (r - x1) / (x2 - x1));
		  HP[index] = (float)(cos(MM_PI / 2.0 * (log (x * 2.0 / MM_PI) / log(2.0))));
		}
          }
      }
    return 1;
  }  
	
    
int genLPfilter(float *LP0, int w, int h, double x1, double x2)
  {
     double r, x;
     int i, j, k, l;

     for (j = 0; j < h; j++)
       {
          for (i = 0; i < w; i++)
	    {		        
	      int index = i + w * j; 
	      
	      if (i <= w/2) {k = i;}
	      if (i >  w/2) {k = i - w;}
	       
	      if (j <= h/2) {l = j;}
	      if (j >  h/2) {l = j - h;}
	      
	      r = sqrt((float)(k*k + l*l));
	      
              if (r < x1) {LP0[index] = 1.0;}
	      if (r > x2) {LP0[index] = 0.0;}
	      
	      if ((r >= x1) && (r <= x2)) 
	        {
		  x = MM_PI / 4.0 * (1.0 + (r - x1) / (x2 - x1));
		  LP0[index] = (float)(cos(MM_PI / 2.0 * (log(x * 4.0 / MM_PI)/ log(2.0))));
		}
            }
      }
    return 1;
  }    



bool decompose(int maxscale, int K, float *kanaal, int w, int h, float **L1, 
		float ***Br, float ***Bc, float  **Ar, float  **Ac)

  {
	fftwf_plan p;	
	fftwf_complex *fim = NULL, *conv = NULL, *ftmp = NULL, *finput = NULL;

	int i, j, k, t, w1, h1, w2, h2, scale;
	float *fLP0 = NULL, *fHP0= NULL, *fLPtmp= NULL, *fHPtmp= NULL; 
	float **fB = NULL;
	//float **tmpar, *tmparL, *tmparH;
	double a, kfact, kfact2, normfactor;
	double normC;
	normC = sqrt(2.0 * K - 1.0);

	fim    = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * w  * h); 
	conv   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * w  * h); 
	ftmp   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * w  * h); 
	finput = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * w  * h); 

	//initialize input
	//================
	
	for (j = 0; j < h; j++) 
	  {
	    for (i = 0; i < w; i ++) 
	      {
	        L1[0][i + w * j] = kanaal[i + w * j];
		finput[i + w * j][0] = kanaal[i + w * j];
		finput[i + w * j][1] = 0.0;
	      }
	  }

	//scale = 0
	//=========
	
    a = w / 2;
		
	p = fftwf_plan_dft_2d(h, w, finput, fim, -1, FFTW_ESTIMATE); 
	fftwf_execute(p);

try {

	fLP0   = new float[w * h];	
	fHP0   = new float[w * h];	
	fLPtmp = new float[w * h];	
	fHPtmp = new float[w * h];	
	fB    = new float*[K];

	for (k = 0; k < K; k ++)
	  {
	    fB[k] = new float[w * h];
	 //   tmpar[k] = (float*)malloc(sizeof(float) * w * h);
	  }	

   } 

catch (...) {
	
     	fftwf_free(fim);	
		fftwf_free(conv);
		fftwf_free(ftmp);
		fftwf_free(finput);	
	for (k = 0; k < K; k ++) delete [] fB[k];
	delete [] fB;
	delete [] fLP0;
	delete [] fHP0;
	delete [] fLPtmp;
	delete [] fHPtmp;
		return false;	 
	 
   }


//	tmparL = (float*)malloc(sizeof(float) * w * h);
//	tmparH = (float*)malloc(sizeof(float) * w * h);
	
//	tmpar = (float**)malloc(sizeof(float *) * K);
	
	
	
//	tmparL = (float*)malloc(sizeof(float) * w * h);
//	tmparH = (float*)malloc(sizeof(float) * w * h);	
	
	//exp(my_lgamma(x)) = (x-1)!
	//=======================
	
	//kfact  = exp(my_lgamma(K));
	kfact  = my_lgamma(K - 1);
	//kfact2 = exp(my_lgamma(2.0 * K));
	kfact2 = my_lgamma(2 * K - 1);
	normfactor = normC * kfact / sqrt(K * (kfact2));
	
	genLPfilter(fLP0,   w, h, a/2, a);     
	genHPfilter(fHP0,   w, h, a/2, a); 
	genHPfilter(fHPtmp, w, h, a/4, a/2); 	    
	genLPfilter(fLPtmp, w, h, a/4, a/2);
	
	for (j = 0; j < h; j++) 
	  {
	    for (i = 0; i < w; i++) 
	      {
	        int index = i + w * j;
		int k, l, m, n, t, index2, index3;
		float theta;
		if (i <  w/2) {k =     i;}
		if (i >= w/2) {k = i - w;}
		if (j <  h/2) {l =     j;}
		if (j >= h/2) {l = j - h;}
		
		if (i <  w/2) {m = i + w/2;}
		if (i >= w/2) {m = i - w/2;}
		if (j <  h/2) {n = j + h/2;}
		if (j >= h/2) {n = j - h/2;}
		
		index2 = k + w * l;
		index3 = m + w * n;
		
		for (t = 0; t < K; t ++)
		  {
		    theta = (float)(atan2((float)l, (float)k) - t * MM_PI / K);		
		    if ((fabs(theta) < MM_PI / 2.0) || (fabs(theta) > 3.0 * MM_PI / 2.0)) 
		      {
		        fB[t][index] = (float)(normfactor * pow((double)(2.0 * cos(theta)), (double)(K-1)));
		      } 
		    else 
		      {
		        fB[t][index] = 0.0;
		      }
		  //  tmpar[t][index3] = fB[t][index] * fLP0[index] * fHPtmp[index];
		  }
             }
          }	  

//	            output (tmpar[0], w, h, id, im, "fB1new.ppm", 1);  		    
//	if (K > 1) {output (tmpar[1], w, h, id, im, "fB2new.ppm", 1);}  		    
//	if (K > 2) {output (tmpar[2], w, h, id, im, "fB3new.ppm", 1);}  		    
//	if (K > 3) {output (tmpar[3], w, h, id, im, "fB4new.ppm", 1);}  		    
	//output (tmparL, w, h, id, im, "fLnew.ppm", 1);  		    
	//output (tmparH, w, h, id, im, "fHnew.ppm", 1);  
	
//	free(tmpar);  	free(tmparL);	free(tmparH);


	for (t = 0; t < K; t ++)
	  {
	     fourier2spatialband2(w, h, fHP0, fB[t],  Ar[t], Ac[t], conv, fim, ftmp);	
	     fourier2spatialband3(w, h, fLP0, fHPtmp, fB[t], Br[t][0], Bc[t][0], conv, fim, ftmp);
	  }
			    	
	fourier2spatialband2a(w, h, fLP0, fLPtmp, L1[0], conv, fim, ftmp);
	
	//subsampling for the next scale
	//==============================

	w2 = w / 2;
	h2 = h / 2;

	for (j = 0; j < h2; j ++)
	  {
	    for (i = 0; i < w2; i ++)
	      {
		 L1[1][i + w2 * j] = L1[0][2 * (i + 2 * w2 * j)];
	      }
	  }
  
     	fftwf_free(fim);	
		fftwf_free(conv);
		fftwf_free(ftmp);
		fftwf_free(finput);	
	
		//free(fLP0);	free(fHP0);	free(fLPtmp);	free(fHPtmp);	free(fB);		

	for (k = 0; k < K; k ++) delete [] fB[k];
	delete [] fB;
	delete [] fLP0;
	delete [] fHP0;
	delete [] fLPtmp;
	delete [] fHPtmp;

	const int ar_scale = 1;

	//iterative part of the decomposition
	//===================================
	
	for (scale = 1; scale < maxscale; ++scale)
	  {     	    
	    // w1 = (int)( w / pow(2, scale));
	     //h1 = (int)(h / pow(2, scale));
	     
	     w1 =  w / my_pow(2, scale);
	    h1 = h / my_pow(2, scale);
		

	     //printf("Scale = %d\n", scale);
	     
	     a = w1 / 2;
	     
	     fim    = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (w1  * h1)); 
	     conv   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (w1  * h1)); 
	     ftmp   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (w1  * h1)); 
	     finput = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (w1  * h1)); 
	     
	     for (j = 0; j < h1; j++) 
	       {
	         for (i = 0; i < w1; i++) 
	           {
	              int index = i + w1 * j;
		      finput[index][0] = L1[scale][index];
		      finput[index][1] = 0.0;
		   }
	       }
	
	     p = fftwf_plan_dft_2d(h1, w1, finput, fim, -1, FFTW_ESTIMATE); 
		 fftwf_execute(p);
	     
		 /*
	     fLPtmp = (float*)malloc(sizeof(float) * w1 * h1);
	     fHPtmp = (float*)malloc(sizeof(float) * w1 * h1);
	     
	     fB = (float**)malloc(sizeof(float *) * K);
	     for (k = 0; k < K; k ++)
	       {
	         fB[k] = (float*)malloc(sizeof(float) * w1 * h1);
	       }
*/

try {


	     fLPtmp = new float[w1 * h1];
	     fHPtmp = new float[w1 * h1];
	     
	     fB = new float*[ K ];;
	     for (k = 0; k < K; k ++) fB[k] = new float[w1 * h1];

} 
catch (...) {

			
     	     fftwf_free(fim);	fftwf_free(conv);	fftwf_free(ftmp);	fftwf_free(finput);	
	 	     for (k = 0; k < K; k ++) delete [] fB[k];
			 delete [] fB;
			 delete [] fLPtmp;
			 delete [] fHPtmp;
			return false;
}


//	     kfact  = exp(my_lgamma(K));
//	     kfact2 = exp(my_lgamma(2.0 * K));
	     kfact  = my_lgamma(K - 1);
	     kfact2 = my_lgamma(2 * K - 1);
	     normfactor = normC * kfact / sqrt(K * (kfact2));
	     
	     genHPfilter(fHPtmp, w1, h1, a/4, a/2); 	    
	     genLPfilter(fLPtmp, w1, h1, a/4, a/2);
	
	     for (j = 0; j < h1; j++) 
	       {
	         for (i = 0; i < w1; i++) 
	           {
	              int index = i + w1 * j;
		      int k, l, t;
		      float theta;
		      if (i <  w1/2) {k =     i;}
		      if (i >= w1/2) {k = i - w1;}
		      if (j <  h1/2) {l =     j;}
		      if (j >= h1/2) {l = j - h1;}
		      
		      for (t = 0; t < K; t ++)
		        {
		          theta = (float)(atan2((float)l, (float)k) - t * MM_PI / K);		
		          if ((fabs(theta) < MM_PI / 2.0) || (fabs(theta) > 3.0 * MM_PI / 2.0)) 
		            {
		              fB[t][index] = (float)(normfactor * pow((double)(2.0 * cos(theta)), (double)(K-1)));
		            } 
		          else 
		            {
		              fB[t][index] = 0.0;
		            }
		        }
                  }
               }		    
	     for (t = 0; t < K; t ++)
	       {
		 fourier2spatialband2(w1, h1, fHPtmp, fB[t],  Br[t][scale], Bc[t][scale], conv, fim, ftmp);
	       }
	     fourier2spatialband1(w1, h1, fLPtmp, L1[scale], conv, fim, ftmp);
	     
	     //subsampling for the next scale
	     //==============================
	     
	     //w2 = (int)(w / pow(2, scale+1));
	     //h2 = (int)(h / pow(2, scale+1));
	     w2 = w / my_pow(2, scale+1);
	     h2 = h / my_pow(2, scale+1);
	     
	     for (j = 0; j < h2; j ++)
	       {
	         for (i = 0; i < w2; i ++)
		   {
		      L1[scale + 1][i + w2 * j] = L1[scale][2 * (i + 2 * w2 * j)];
		   }
	       }
			
     	     fftwf_free(fim);	fftwf_free(conv);	fftwf_free(ftmp);	fftwf_free(finput);	
	     
			 //free(fLPtmp);	free(fHPtmp);		free(fB);		

	 	     for (k = 0; k < K; k ++) delete [] fB[k];
			 delete [] fB;
			 delete [] fLPtmp;
			 delete [] fHPtmp;

          }
	return true;
 }
