Documents: detail enhancement
=================================

**void detailEnhancementBilateral(Mat& src, Mat& dest, int d, float sigma_color, float sigma_space, float boost, int color)**
* Mat& src: input image.  
* Mat& dst: filtered image.  
* const Size ksize: size of filtering kernel.  
* const float sigma_color: sigma of color in bilateral filtering.    
* const float sigma_space: sigma of space in bilateral filtering.    
* const float boost: boosting factor.   
* const int color: color channel for using boosting.

        enum
        {
        	PROCESS_LAB=0,
        	PROCESS_BGR
        };

Example of detail enhancement
-----------------------------
![debf](Detail_Enhancement.png "debf")  
detail enhancement bilateral filter.

Reference
---------

R. Fattal, M. Agrawala, and S. Rusinkiewicz, "Multiscale shape and detail enhancement from multi-light image collections," ACM Transactions on Graphics, 26(3), Aug. 2007
