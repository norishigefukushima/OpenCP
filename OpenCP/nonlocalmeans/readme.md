Documents: non-local means filter
=================================


**void nonLocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma=-1.0, int method=FILTER_DEFAULT)**
* Mat& src: input image.  
* Nat& dst: filtered image.  
* int templeteWindowSize: templete window size.  
* int searchWindowSize: serch window size.  
* double h: dominator of weight.  
* double sigma=-1.0, offset of weight. if sigma<0 then sigma = h.  
* int method: switch for various implimentations (default is FILTER_DEFAULT).   

The code is 10x faster than the opencv implimentation of non-local means filter (	fastNlMeansDenoising).
In addition, the code has higher denoising performance than the opencv.  

Reference
---------
1. A. Buades, B. Coll, J.M. Morel “A non local algorithm for image denoising” IEEE Computer Vision and Pattern Recognition 2005, Vol 2, pp: 60-65, 2005.  
2. J. Wang, Y. Guo, Y. Ying, Y. Liu, Q. Peng, “Fast Non-Local Algorithm for Image Denoising,” in Proc. IEEE International Conference on Image Processing 2006 (ICIP), pp. 1429 – 1432, Oct. 2006.  
