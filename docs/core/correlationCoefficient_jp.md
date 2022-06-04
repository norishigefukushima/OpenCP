correlationCoefficient.hpp
===================
 相関係数や順位相関係数（SROCC or KROCC）を計算します．
 SRCC，KRCCと省略することもあります．

# class PearsonCorrelationCoefficient
Pearsonの相関係数を計算します。

```cpp
	class CP_EXPORT PearsonCorrelationCoefficient
	{
		template<typename T>
		double mean(std::vector<T>& s);
		template<typename T>
		double covariance(std::vector<T>& s1, std::vector<T>& s2);
		template<typename T>
		double stddev(std::vector<T>& v1);
	public:
		double compute(std::vector<int>& v1, std::vector<int>& v2);
		double compute(std::vector<float>& v1, std::vector<float>& v2);
		double compute(std::vector<double>& v1, std::vector<double>& v2);
	};
```

# class SpearmanRankOrderCorrelationCoefficient
引数のvectorに入った2つの指標のSpearmanの順位相関係数を計算します．

```cpp
	class CP_EXPORT SpearmanRankOrderCorrelationCoefficient
	{
		template<typename T>
		double rankTransformUsingAverageTieScore(std::vector<T>& src, std::vector<float>& dst);
		template<typename T>
		void rankTransformIgnoreTie(std::vector<T>& src, std::vector<float>& dst);
		template<typename T>
		void rankTransformBruteForce(std::vector<T>& src, std::vector<float>& dst);//not fast, and not used

		template<typename T>
		double computeRankDifference(std::vector<T>& Rsrc, std::vector<T>& Rref);

		template<typename T>
		double spearman_(std::vector<T>& v1, std::vector<T>& v2, const bool ignoreTie, const bool isPlot);

		template<typename T>
		void setPlotData(std::vector<T>& v1, std::vector<T>& v2, std::vector<cv::Point2d>& data);


		cp::Plot pt;
		std::vector<cv::Point2d> plotsRAW;
		std::vector<cv::Point2d> plotsRANK;
		double Tref = 0.0;
		std::vector<float> refRank;
		std::vector<float> srcRank;

		template<typename T>
		struct SpearmanOrder
		{
			T data;
			int order;
		};
		std::vector<SpearmanOrder<int>> sporder32i;
		std::vector<SpearmanOrder<float>> sporder32f;
		std::vector<SpearmanOrder<double>> sporder64f;
	public:
		void setReference(std::vector<int>& ref, const bool ignoreTie = false);
		void setReference(std::vector<float>& ref, const bool ignoreTie = false);
		void setReference(std::vector<double>& ref, const bool ignoreTie = false);
		double computeUsingReference(std::vector<int>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<float>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double computeUsingReference(std::vector<double>& v1, const bool ignoreTie = false);//compute SROCC (vector<float> NOT thread safe). 
		double compute(std::vector<int>& v1, std::vector<int>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<int> thread safe). 
		double compute(std::vector<float>& v1, std::vector<float>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC (vector<float> thread safe). 
		double compute(std::vector<double>& v1, std::vector<double>& v2, const bool ignoreTie = false, const bool isPlot = false);//compute SROCC ((vector<double>) thread safe). 
		void plot(const bool isWait = true, const double rawMin = 0.0, const double rawMax = 0.0);
	};
```

## Usage

public関数は，compute，setReferece/computeUsingReference，とplotです。


computeメソッドはint/float/doubleのvector配列を引数に取り計算します．
setReference/computeUsingReferenceメソッドは，片側の参照データを事前に計算することで高速化します．
なお，内部でソートをする関係上，intの計算が最も高速です．
また引数のignoreTieはタイスコアの計算を無視して若干高速化します．


plotメソッドは，順位相関として計算した場合に散布図がどのようになっているかを可視化します．
isWaitがtrueである場合グラフをプロットして関数を停止します．
isWaitがfalseなら，グラフを表示してすぐに抜けます．

使い方は以下の通りです．

```cpp
vector<double> psnr;
vector<double> mos;
... //compute or input PSNR and MOS

SpearmanRankOrderCorrelationCoefficient　srocc;
cout<<srocc.spearman(ppsnr,mos)<<endl;//相関係数の出力
srocc.plot();//順位相関係数の計算プロットを出力

```

# class KendallRankOrderCorrelationCoefficient
Kendallの順位相関係数を計算します．
Pearsonよりも低速です．

```cpp
	class CP_EXPORT KendallRankOrderCorrelationCoefficient
	{
		//not optimized but can be vectorized, not parallelization is off
		template<typename T>
		double body(std::vector<T>& x, std::vector<T>& y);
	public:

		double compute(std::vector<int>& x, std::vector<int>& y);
		double compute(std::vector<float>& x, std::vector<float >& y);
		double compute(std::vector<double>& x, std::vector<double>& y);
	};
```