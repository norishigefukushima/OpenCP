Documents: SLIC (Simple Linear Iterative Clustering) 
=========

**void SLIC(Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration)**  
* Mat& src: input image.  
* Nat& segment: segmentimage image(CV_32S). Each pixel has integer value for labeling.
* int regionSize: nominal size of the regions( parameter S in the refernece paper).   
* float regularzation: a trade-off between appearance and spatial terms (parameter m in the refernece paper).
* int minRegionRatio: ratio of minimum size of a segment for regionSize*regionSize for threshoding minimum regions.   
* int max_iteration: number of max interations.  

**void drawSLIC(Mat& src, Mat& segment, Mat& dest, bool isLine, Scalar line_color)**
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
