Documents: guided filter
===========================

**void guidedFilter(const Mat& src,  Mat& dest, const int radius,const float eps)**  
**void guidedFilter(const Mat& src, const Mat& guidance, Mat& dest, const int radius,const float eps)**  
* Mat& src: input image.  
* Mat& guidance: guidance image. If you use no guidance version, the guidance image is the same as src.   
* Mat& dest: filtered image.  
* int r kernel radius. The actual diameter of guided filter is 4*r+1.  
* float eps: smoothing factor.  

**void guidedFilterMultiCore(const Mat& src, Mat& dest, int r,float eps, int numcore=0)**
**void guidedFilterMultiCore(const Mat& src, const Mat& guide, Mat& dest, int r,float eps,int numcore=0)**
Parallel implementaions of the upper's guided filters. If numcore is set to 0, then the functions use maximum cores in your system.



Example of guided filter: computational speed
------------------------------------------------
**Tested on Dual CPU of Intel Xeon X5690 3.47Ghz (12 core * HT), 64bit OS, Visual Studio 2012's compiler**  

![birateral](birateral_time.png "birateraltime")



Reference
---------
1. K. He, S. Jian, and T. Xiaoou, "Guided image filtering," Proc. European Conference on Computer Vision–ECCV, 2010.  
2. K. He, S. Jian, and T. Xiaoou, "Guided image filtering," IEEE Trans. Pattern Analysis and Machine Intelligence, vol 35, issue 6, pp. 1397-1409, 2013.  


