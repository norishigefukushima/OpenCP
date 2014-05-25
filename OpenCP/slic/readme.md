Documents: SLIC (Simple Linear Iterative Clustering) 
=========

SLIC is a fast segmentation function. The function of SLIC is called, and then drawSLIC could be called for rendering.   
**void SLIC(const Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration)**  
* Mat& src: input image.  
* Nat& segment: segmentimage image(CV_32S). Each pixel has integer value for labeling.
* int regionSize: nominal size of the regions( parameter S in the refernece paper).   
* float regularzation: a trade-off between appearance and spatial terms (parameter m in the refernece paper).
* int minRegionRatio: ratio of minimum size of a segment for regionSize*regionSize for threshoding minimum regions.   
* int max_iteration: number of max interations.  

**void drawSLIC(const Mat& src, Mat& segment, Mat& dest, bool isLine, Scalar line_color)**
* Mat& src: input image.  
* Nat& dest: segmenttation result from the SLIC function.
* bool isLine: draw line or not between segments.
* Scalar line_color: line color.

*Reference*
* Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, num. 11, p. 2274 - 2282, May 2012.
* Author's Webpage: http://ivrg.epfl.ch/research/superpixels

Example
-------
![SLIC](SLIC_screenshot.png "screenshot")
![SLIC2](SLIC_screenshot2.png "screenshot2")

Documents: Blending
-------------------
This is a alpha blending functions.  
**void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest)**    
**void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)**  
* Mat& src1: input image1.  
* Mat& src2: input image2.  
* Mat or double alpha: alpha channel map, and values are set from 0 to 1. if the value is scalar, src mat are blended a constant value.  If the type of alpha map is CV_8U or uchar, values are set from 0 to 255.
* Mat& dest: destination image.  

various test for interactive alpha blending.  
**void guiAlphaBlend(const Mat& src1, const Mat& src2)**  
*keyboard short cut*  
* q: quit  
* f: flip alpha value  

Example  
-------

![alphablend](alpha_blend.png "ab")
![guialphablend](gui_alpha_blend.png "gab")


merging or blending two image into an image like triangular matrix. The function is used for SLIC demo.  
**void patchBlendImage(Mat& src1, Mat& src2, Mat& dest, Scalar linecolor=CV_RGB(0,0,0), int linewidth = 2, int direction = 0)**  

Example  
-------


showing information of class Mat for debugging.  
**void showMatInfo(InputArray src_, string name)**  
The function prints out size, num. of channles, and depth of Mat infomation. Also, the mean, min, max value of matrix in each channel.  

Documents: Color   
----------------
**void cvtColorBGR2PLANE(const Mat& src, Mat& dest)**  

**void splitBGRLineInterleave( const Mat& src, Mat& dest)**  
src.cols%16==0   
