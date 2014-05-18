Documents
=========
**void SLIC(Mat& src, Mat& dest, unsigned int regionSize, float regularization, int minRegionSize, int max_iteration)**  
* Mat& src: input image.  
* Nat& dest: destination image.  
* int regionSize: nominal size of the regions.   
* float regularzation: a trade-off between appearance and spatial terms.  
* int minRegionSize: minimum size of a segment.   
* int max_iteration: number of max interations.  

*Reference*
* Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, num. 11, p. 2274 - 2282, May 2012.
* Author's Webpage: http://ivrg.epfl.ch/research/superpixels
