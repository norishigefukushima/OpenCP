OpenCP
======
library for computational photography

Todo
----
** port various filters from my library.**  
* bilateral filter approxmations   
* add denoising demo  
* add dxt thresholding demo  
* add demo for iterative back projection for super resolution  
* add detail enhancement document
* 
* stereo matching depth for joint binary weighted range filter.    
* need floating birateral filter without LUT for back projection.  
~~add detail ehnancement5/28~~  
~~add birateral iterative back projection for debluring5/27~~  
~~add iterative back projection for debluring5/26~~  
~~add joint birateral filter to rect kernel implimentation. 5/26~~    
~~add weighted binary range filter5/25~~    
~~add joint birateral filter 5/24~~    
~~add massively parallel implimentaion of birateral filter 5/21~~    
~~add slowest birateral filter5/20~~  
~~* update type of the destination of the SLIC. mean image? mesh? 5/19~~


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

**Application**
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
