OpenCP
======
library for computational photography

Todo
----

** port various filters from my library.**  
* bilateral filter approxmations   
  * ~~separable~~
  + bilateral grid
  + constant time O(1) bilateral
  + real-time O(1) bilateral
* add dxt thresholding for denoising
* add super resolution demo for iterative back projection.   
* add recursive birateral filter for gray, more efficient implimentaion, line by line chatching?    
* add domain transform filter  
  * ~~add test for domain transform filter.6/11~~
  * ~~add  various simd implementation for RF filter.6/11~~  
  * add SIMD V-direction ver.  
  * add other implimentation of domain transform filter  
* write detail enhancement document
* add detail enhancement class.  

* add color joint nearest filter 
* add test for stencil and parallel filtering.  
* need stereo matching/cost filter for joint filtering.    
* need sse implimmentation cvtColorBGRA2BGR,cvtColorBGR2BGRA  
* need a floating birateral filter without LUT or quantization for back projection.  
* ~~remove FFTW
  * ~~C:\Users\fukushima\Documents\GitHub\OpenCP\OpenCP\libGaussian\gaussian_conv.h(2):#include "fftw/fftw3.h"
    * can be removed by replacing OpenCV based GF
  * ~~C:\Users\fukushima\Documents\GitHub\OpenCP\OpenCP\libimq\fourpyrtransf3.cpp(2):#include "fftw/fftw3.h"
    * can be removed by replacing other our class
  * ~~C:\Users\fukushima\Documents\GitHub\OpenCP\OpenCP\libimq\imq.h(5):#include "fftw/fftw3.h"
    * can be removed by replacing other our class
~~add joint nearest filter 6/18~~  
~~add view synthesis class 6/18~~  
~~add domain transform filter of RF implementation. 6/8~~  
~~add guided filter 6/5~~  
~~add denoising demo5/31~~  
~~add weighted birateral filter and joint birateral filter, but some implimentation is not same as non weighted version.5/31~~
~~add recursive birateral filter 5/28~~  
~~add detail ehnancement 5/28~~  
~~add birateral iterative back projection for debluring5/27~~  
~~add iterative back projection for debluring5/26~~  
~~add joint birateral filter to rect kernel implimentation. 5/26~~    
~~add weighted binary range filter5/25~~    
~~add joint birateral filter 5/24~~    
~~add massively parallel implimentaion of birateral filter 5/21~~    
~~add slowest birateral filter5/20~~  
~~update type of the destination of the SLIC. mean image? mesh? 5/19~~


Filter
------
###implemented and parallelized
######filter   
  Gaussian IIR filter  

######edge preserving filter  
  *bilateral filter and its fast implimentations or variants*  
      *sepalable filter  
      *bilateral grid  
      *realtime O(1) birateral filter  
      *joint bilateral filter  
      *trilateral filter  
      *dual bilateral filter  
      *weighted (joint) bilateral filter  
    
  *cost volume filters*
   *3D birateral filter  
   *3D guided filter    
  
  trilateral filter  
  non-local means filter  
  shiftable DXT thresholding filter  
  guided filter  
  domain transform filter  
  weighted mode filter  
  constant time median filter  
  joint nearest filter
######segmentation  
  SLIC  (forked from VLFeat(http://www.vlfeat.org/). The code, which is optimized by SSE and Intel TBB, is more efficient than the VLFeat.)
######upsample
  joint bilateral upsample  
  guided upsample  
  hqx  
  
**implimented but not optimized**  

*filter*  
  recursive birateral filter
  constant time O(1) birateral filter  
  L0 Smoothing  
  Weighted least squre (WLS) smoothing  
  Gaussian KD-Tree  
  Permutohedral Lattice  
  adaptive maniforld    

**Example of Applications**
-----------

  + denoise   
  + deblur  
  + up-sample/single image super resolution  
  + flash/non flash photograpy  
  + HDR  
  + colorization  
  + detail enhancement  
  + stylization, abstruction  
  + pencil sketche  
  + up-sample for pixel art, depth map  
  + removing coding noise  
  + blur regeneration  
  + Haze remove  
  + depth map estimation/refinement  
  + optical flow estimation/refinement      
  + alpha matting  

References
----------
