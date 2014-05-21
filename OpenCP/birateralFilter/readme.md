Documents: Birateral Filter
===========================

**void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method, int borderType**
* Mat& src: input image.  
* Nat& dst: filtered image.  
* Size kernelSize: kernel size.  
* double sigma_color: Gaussian sigma for color weight.  
* double sigma_space: Gaussian sigma for color weight.  
* int method: switch for various implimentations (default is BILATERAL_DEFAULT).   
* int borderType: boundary copying function (default is cv::BORDER_REPLICATE).  


The *method* has following options:　　

    enum{
    BILATERAL_DEFAULT = 0, //default is circle type.
    BILATERAL_CIRCLE = 0, //kernel shape is circlar from. If a support pixel (q) in a kernel has a large distance |p-q|^2 > r^2, coefficient is forcibly 0.(p is a center pixe.)
    BILATERAL_RECT , //kernel shape is squre or reqtangler type. 
    BILATERAL_SEPARABLE,// sepalable implimentation of ref. 2.
    BILATERAL_ORDER2,//exponential function is approxmated by Taylor expansion(underconstruction)
    BILATERAL_ORDER2_SEPARABLE,// spalable implimentation of  ORDER2(underconstruction)
    BILATERAL_SLOWEST// non-parallel and un-effective implimentation for just comparison.    
    };
    
Example
-------
Computational time for a 1M pixel (1024 * 1024) and color image with following methods:  
* OpenCV implimentation  
* BILATERAL_DEFAULT  
* SEPARABLE  
* SLOWEST  

each median value in 10 times trials is plotted.  
**Tested on Dual CPU of Intel Xeon X5690 3.47Ghz (12 core * HT), 64bit OS, Visual Studio 2012's compiler*  

![birateral](birateral_time.png "birateraltime")



Reference
---------
1. Tomasi, Carlo, and Roberto Manduchi. "Bilateral filtering for gray and color images," Proc. IEEE International Conference on Computer Vision (ICCV), 1998.  
2. Pham, Tuan Q., and Lucas J. Van Vliet. "Separable bilateral filtering for fast video preprocessing," IEEE International Conference on Multimedia and Expo (ICME) 2005.  
