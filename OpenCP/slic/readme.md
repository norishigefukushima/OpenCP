Documents
=========

**void SLIC(Mat& src, Mat& segment, int regionSize, float regularization, float minRegionRatio, int max_iteration)**  
* Mat& src: input image.  
* Nat& dest: destination image.  
* int regionSize: nominal size of the regions.   
* float regularzation: a trade-off between appearance and spatial terms, and input image is normized (0 to 1).
* int minRegionRatio: ratio of minimum size of a segment for regionSize*regionSize.   
* int max_iteration: number of max interations.  

*Reference*
* Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, num. 11, p. 2274 - 2282, May 2012.
* Author's Webpage: http://ivrg.epfl.ch/research/superpixels

![SLIC](SLIC_screenshot.png "screenshot")