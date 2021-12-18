 RankOrderCorrelationCoefficient.hpp
===================
 順位相関係数（SROCC or KROCC）を計算します．
 SRCC，KRCCと省略することもあります．

# class SpearmanRankOrderCorrelationCoefficient
引数のvectorに入った2つの指標のSpearmanの順位相関係数を計算します．

```cpp
class CP_EXPORT SpearmanRankOrderCorrelationCoefficient
{
		template<typename T>
		double mean(std::vector<T>& s);
		template<typename T>
		double covariance(std::vector<T>& s1, std::vector<T>& s2);
		template<typename T>
		double std_dev(std::vector<T>& v1);
		template<typename T>
		double pearson(std::vector<T>& v1, std::vector<T>& v2);
		template<typename T>
		void searchList(std::vector<T>& theArray, int sizeOfTheArray, double findFor, std::vector<int>& index);
		template<typename T>
		void Rank(std::vector<T>& vec, std::vector<T>& orig_vect, std::vector<T>& dest);
		template<typename T>
		double spearman_(std::vector<T>& v1, std::vector<T>& v2);

		void setPlotData(std::vector<float>& v1, std::vector<float>& v2, std::vector<cv::Point2d>& data);
		void setPlotData(std::vector<double>& v1, std::vector<double>& v2, std::vector<cv::Point2d>& data);

		cp::Plot pt;
		std::vector<cv::Point2d> plotsRAW;
		std::vector<cv::Point2d> plotsRANK;
		
	public:

		double spearman(std::vector<float> v1, std::vector<float> v2);//compute SROCC (vector<float>). 
		double spearman(std::vector<double> v1, std::vector<double> v2);//compute SROCC ((vector<double>)). 
		void plot(const bool isWait = true);
};

```

## Usage

public関数は以下の3つです．

```cpp
void plot(const bool isWait = true);
double spearman(std::vector<float> v1, std::vector<float> v2);//copy v1 and v2
double spearman(std::vector<double> v1, std::vector<double> v2);//copy v1 and v2
```

spearmanメソッドはfloatもしくはdoubleのvector配列を引数に取り計算します．
あまり重要な情報ではありませんが，内部でvectorのコピーが必要となるため参照渡しをせず，コピとなるように値を渡しています．

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
