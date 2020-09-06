guidedFilter.hpp
================

Documentation for implementation of [Guided Image Filtering](http://kaiminghe.com/eccv10/).

# guidedImageFilter
```cpp
void guidedImageFilter(cv::InputArray src, cv::InputArray guide, cv::OutputArray dest, 
	const int r, const float eps, 
	const GuidedTypes guidedType = GuidedTypes::GUIDED_SEP_VHI_SHARE, const BoxTypes boxType = BoxTypes::BOX_OPENCV, const ParallelTypes parallelType = ParallelTypes::OMP);
```
## Usage
`src`：入力，`guide`：ガイド画像．srcと同一でも可，`dest`：出力画像  
`r`：カーネル半径，`eps`：エッジキープ用のパラメータ

`guidedType`：ガイデットフィルタの計算スケジューリング方法．どれが速いかはCPUによるが，ｒが小さいなら`GUIDED_SEP_VHI_SHARE`が速い．
ある程度大きいなら`GUIDED_MERGE_SHARE_AVX`．

```cpp
enum GuidedTypes
{
	GUIDED_XIMGPROC,
	//--- Conventional Algorithm---
	GUIDED_NAIVE,
	GUIDED_NAIVE_SHARE,
	GUIDED_NAIVE_ONEPASS,
	GUIDED_SEP_VHI,
	GUIDED_SEP_VHI_SHARE,
	//--- Merge Algorithm --- 
   // SSAT	
   GUIDED_MERGE_AVX,
   GUIDED_MERGE_TRANSPOSE_AVX,
   GUIDED_MERGE_TRANSPOSE_INVERSE_AVX,
   // SSAT	(cov reuse)	
   GUIDED_MERGE_SHARE_AVX,
   GUIDED_MERGE_SHARE_EX_AVX,
   GUIDED_MERGE_SHARE_TRANSPOSE_AVX,
   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_AVX,

   GUIDED_MERGE,
   GUIDED_MERGE_SSE,
   GUIDED_MERGE_TRANSPOSE,
   GUIDED_MERGE_TRANSPOSE_SSE,
   GUIDED_MERGE_TRANSPOSE_INVERSE,
   GUIDED_MERGE_TRANSPOSE_INVERSE_SSE,
   GUIDED_MERGE_SHARE,
   GUIDED_MERGE_SHARE_SSE,
   GUIDED_MERGE_SHARE_EX,
   GUIDED_MERGE_SHARE_EX_SSE,
   GUIDED_MERGE_SHARE_TRANSPOSE,
   GUIDED_MERGE_SHARE_TRANSPOSE_SSE,
   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE,
   GUIDED_MERGE_SHARE_TRANSPOSE_INVERSE_SSE,
   // SSAT (BGR non-split)
   GUIDED_NONSPLIT,
   GUIDED_NONSPLIT_SSE,
   GUIDED_NONSPLIT_AVX,
   // OP-SAT
   GUIDED_MERGE_ONEPASS,
   GUIDED_MERGE_ONEPASS_2div,
   GUIDED_MERGE_ONEPASS_SIMD,

   // --- Fast Guided Filter --- 
   GUIDED_MERGE_ONEPASS_FAST,

   NumGuidedTypes	// num of guidedTypes. must be last element
};
```

`boxType`の計算方法：`GUIDED_NAIVE`と`GUIDED_NAIVE_SHARE`以外は何を指定しても意味がない．
[boxfilter.md](boxfilter.md)を参照のこと．

`parallelType`：並列化方法．
```cpp
enum ParallelTypes
{
	NAIVE,//並列化無し
	OMP,//OpenMPによる並列化
	PARALLEL_FOR_,//OpenCVのparallel_for_による並列化

	NumParallelTypes // num of parallelTypes. must be last element
};
```

## Optimization
* SSE/AVX/single
* Any depth/color

# class GuidedImageFilter
```cpp
class CP_EXPORT GuidedImageFilter
	{
		cv::Mat srcImage;
		cv::Mat guideImage;
		cv::Mat destImage;

		cv::Mat guidelowImage;

		int downsample_method = cv::INTER_LINEAR;
		int upsample_method = cv::INTER_CUBIC;
		int parallel_type = 0;
		int box_type = 0;

		cv::Size size = cv::Size(1, 1);

		std::vector<cv::Mat> vsrc;
		std::vector<cv::Mat> vdest;
		std::vector<cv::Ptr<GuidedFilterBase>> gf;
		cv::Ptr<GuidedFilterBase> getGuidedFilter(cv::Mat& src, cv::Mat& guide, cv::Mat& dest, const int r, const float eps, const int guided_type);
		bool initialize(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest);
	public:
		GuidedImageFilter()
		{
			size = cv::Size(1, 1);
			gf.resize(3);
		}

		void setDownsampleMethod(const int method);
		void setUpsampleMethod(const int method);
		void setBoxType(const int type);
		void filterColorParallel(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filter(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filterFast(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int ratio, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void upsample(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void upsample(cv::Mat& src, cv::Mat& guide_low, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);

//for tiling implementation
		void filter(cv::Mat& src, std::vector<cv::Mat>& guide, cv::Mat& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filter(std::vector<cv::Mat>& src, cv::Mat& guide, std::vector<cv::Mat>& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
		void filter(std::vector<cv::Mat>& src, std::vector<cv::Mat>& guide, std::vector<cv::Mat>& dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);
	};
```

guided image filterのclass実装．関数の実装はこのクラスをラップしているだけ．
重要なメソッドは下記5つ

* ```void filter(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);```
	* 通常のガイデットフィルタ
* ```void filterColorParallel(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);```
	* 並列化を色空間で並列化する実装
* ```void filterFast(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int ratio, const int * guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);```
	* 高速ガイデットフィルタ [fast guided filter](https://arxiv.org/abs/1505.00996)
	* 画像をダウンサンプルする．
* ```void upsample(cv::Mat& src, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);```
	* ガイデットアップサンプル．高速ガイデットフィルタをアップサンプルに利用．ガイド画像が高解像度画像．低解像度用のガイド画像は内部で計算．
* ```void upsample(cv::Mat& src, cv::Mat& guide_low, cv::Mat& guide, cv::OutputArray dest, const int r, const float eps, const int guided_type = GuidedTypes::GUIDED_SEP_VHI_SHARE, const int parallel_type = ParallelTypes::OMP);```
	* ガイデットアップサンプル．高速ガイデットフィルタをアップサンプルに利用．低解像度画像を事前に用意するバージョン．

また，関数実装のように，ボックスフィルタのタイプをしてするには，下記メソッドを呼び出出す．
```cpp
void setBoxType(const int type)
```

高速ガイデットフィルタやアップサンプルに使う場合のインタポレーションカーネルの指定は以下の関数．
引数の`method`は，OpenCVのresizeの指定方法と同じ．
```cpp
void setDownsampleMethod(const int method);
void setUpsampleMethod(const int method);
```

# class GuidedImageFilterTiling
```cpp
	class CP_EXPORT GuidedImageFilterTiling
	{
	protected:
		cv::Mat src;
		cv::Mat guide;
		cv::Mat dest;
		int r;
		float eps;
		int parallelType;

		cv::Size div = cv::Size(1, 1);

		std::vector<cv::Mat> vSrc;
		std::vector<cv::Mat> vGuide;
		std::vector<cv::Mat> vDest;

		std::vector<cv::Mat> src_sub_vec;
		std::vector<cv::Mat> guide_sub_vec;

		std::vector<cv::Mat> src_sub_b;
		std::vector<cv::Mat> src_sub_g;
		std::vector<cv::Mat> src_sub_r;

		std::vector<cv::Mat> dest_sub_b;
		std::vector<cv::Mat> dest_sub_g;
		std::vector<cv::Mat> dest_sub_r;

		std::vector<cv::Mat> src_sub_temp;
		std::vector<cv::Mat> guide_sub_temp;
		std::vector<cv::Mat> dest_sub_temp;

		std::vector<GuidedImageFilter> gf;
		std::vector<std::vector<cv::Mat>> sub_src;
		std::vector<std::vector<cv::Mat>> sub_guide;
		std::vector<cv::Mat> sub_guideColor;
		std::vector<std::vector<cv::Mat>> buffer;

	public:
		GuidedImageFilterTiling();
		GuidedImageFilterTiling(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, int _r, float _eps, cv::Size _div);

		void filter_SSAT();
		void filter_OPSAT();
		void filter_SSAT_AVX();
		void filter_func(int guidedType);
		void filter(int guidedType);
		void filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, const int _r, const float _eps, const cv::Size _div, const int guidedType);
	};
```

## Usage
タイリングで並列化する実装．下記のどちらかを呼び出せばいい．
引数のSize div = (x,y)で，x×yのブロックに分割してブロックごとに並列化する．
定数時間ではなくてO(r)の実装であるVHIを使う場合は，キャッシュ効率を考えるとブロック分割しないほうが速い．定数時間アルゴリズムの場合は，タイリングのほうが速い場合がある．

* `filter(int guidedType);`
	* コンストラクタで初期化した場合
* `void filter(cv::Mat& _src, cv::Mat& _guide, cv::Mat& _dest, const int _r, const float _eps, const cv::Size _div, const int guidedType);`
	* していない場合．

