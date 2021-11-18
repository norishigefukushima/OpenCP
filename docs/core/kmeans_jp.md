kmeans.hpp
==========
Kmeansクラスタリングをする関数群です．
testKmeansにテスト関数が実装されています．

# class KMeans
```cpp
	class CP_EXPORT KMeans
	{
	public:

		enum class MeanFunction
		{
			Mean,//box
			Gauss,
			GaussInv,
			Harmonic
		};
		void setSigma(const float sigma) { this->sigma = sigma; }//for Gauss means

		enum class Schedule
		{
			Auto,
			AoS_NKD,
			SoA_KND,
			AoS_KND,
			SoA_NKD,
			SoAoS_NKD,
			SoAoS_KND,

			SIZE
		};
		double clustering(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers, MeanFunction function = MeanFunction::Mean, Schedule schedule = Schedule::Auto);
	private:
		float sigma = 0.f;
		cv::Mat labels_internal;
		cv::AutoBuffer<float> _distance;//for kmeans++

		//initialize
		void generateKmeansRandomInitialCentroidSoA(const cv::Mat& data_points, cv::Mat& dest_centroids, const int K, cv::RNG& rng);
		void generateKmeansPPInitialCentroidSoA(const cv::Mat& data, cv::Mat& _out_centers, int K, cv::RNG& rng, int trials);

		void generateKmeansRandomInitialCentroidAoS(const cv::Mat& data_points, cv::Mat& dest_centroids, const int K, cv::RNG& rng);
		void generateKmeansPPInitialCentroidAoS(const cv::Mat& data, cv::Mat& _out_centers, int K, cv::RNG& rng, int trials);

		//get most outer samples for centroids (not used)
		void getOuterSample(cv::Mat& src_centroids, cv::Mat& dest_centroids, const cv::Mat& data_points, const cv::Mat& labels);

		//computing centroid
		void boxMeanCentroidSoA(cv::Mat& data_points, const int* labels, cv::Mat& dest_centroid, int* counters);//simple average
		void weightedMeanCentroid(cv::Mat& data_points, const int* labels, const cv::Mat& src_centroid, float* Table, cv::Mat& dest_centroid, float* centroid_weight, int* counters);
		void harmonicMeanCentroid(cv::Mat& data_points, const int* labels, const cv::Mat& src_centroid, cv::Mat& dest_centroid, float* centroid_weight, int* counters);

		void boxMeanCentroidAoS(cv::Mat& data_points, const int* labels, cv::Mat& dest_centroid, int* counters);//N*d simple average

		double clusteringSoA(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers, MeanFunction function = MeanFunction::Mean, int loop = 0);
		double clusteringAoS(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers, MeanFunction function = MeanFunction::Mean, int loop = 1);
		double clusteringSoAoS(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers, MeanFunction function = MeanFunction::Mean, int loop = 0);
	};
```
## Usage
K-meansクラスタリングをします．
基本OpenCVのK-meansと同じですが，データ構造をAoSとSoAで変更できます．
デフォルトは`Schedule schedule = Schedule::Auto`で自動で最良のものを選択しますが，`enum class Schedule`で指定したスケジュールで計算可能です．
また，MeanFunction function = MeanFunction::Meanで平均の方法を変えられます．
重み関数は，setSigmaでパラメータを変えられます．

# kmeans
```cpp
CP_EXPORT double kmeans(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels, cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers);
```
classのラッパー関数です．
OpenCVの呼び出しと全く同じです．
スケジュールは，Autoが自動選択されます．

# 参考文献
実装の詳細

* T. Otsuka, N. Fukushima, "Vectorized Implementation of K-means," in Proc. International Workshop on Advanced Image Technology (IWAIT), Jan. 2021.

K-means

* Forgy, E., “Cluster Analysis of Multivariate Data: Efficiency vs. Interpretability of Classification,” Biometrics, 21, 768 (1965).
* Lloyd, S., “Least Squares Quantization in PCM,” IEEE Transactions on Information Theory, 28, 2, 129–136 (1982).

K-means++

* Arthur, D., and Vassilvitskii, S., "k-means++: The Advantages of Careful Seeding.," Technical Report, Stanford (2006).




