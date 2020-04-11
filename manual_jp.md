Reference Manual
================

マニュアルは，みんなで更新していってください．  
福嶋が更新するのを期待して待っても，ほぼ更新されません．
（バグを発見したときについでに更新するくらい）

# Metrics.hpp
画質評価関数群
 * [getPSNR](metrics_jp.md "#getPSNR")
 * [getPSNR_PRECISION](metrics_jp.md "getPSNR_PRECISION")
 * [getPSNR_CHANNEL](metrics_jp.md "#getPSNR_CHANNEL")
 * [class PSNRMetrics](metrics_jp.md "class PSNRMetrics")
 * [localPSNRMap](metrics_jp.md "localPSNRMap")
 * [guiLocalPSNRMap](metrics_jp.md "guiLocalPSNRMap")
 * [getMSE](metrics_jp.md "#getMSE")
 * [getInacceptableRatio](metrics_jp.md "getInacceptableRatio")
 * [getEntropy](metrics_jp.md "#getEntropy")
 * [getTotalVariation](metrics_jp.md "#getTotalVariation")
 * SSIM：関数修正中
 
# crop.hpp
画像の切り抜き（クロップ）用関数群
 * [cropZoom](crop_jp.md "#cropZoom")
 * [cropZoomWithBoundingBox](crop_jp.md "#cropZoomWithBoundingBox")
 * [cropZoomWithSrcMarkAndBoundingBox](crop_jp.md "#cropZoomWithSrcMarkAndBoundingBox")
 * [guiCropZoom](crop_jp.md "#guiCropZoom")

# count.hpp
画像の画素の属性をカウントする関数群
 * [countNaN](count_jp.md "#countNaN")
 * [countInf](count_jp.md "#countInf")
 * [countDenormalizedNumber](count_jp.md "#countDenormalizedNumber")
 * [countDenormalizedNumberRatio](count_jp.md "#countDenormalizedNumberRatio")

# histogram.hpp
ヒストグラムの描画関数群
 * [drawHistogramImage](histogram_jp.md "#drawHistogramImage")
 * [drawHistogramImageGray](histogram_jp.md "#drawHistogramImageGray")
 * [drawAccumulateHistogramImage](histogram_jp.md "#drawAccumulateHistogramImage")
 * [drawAccumulateHistogramImageGray](histogram_jp.md "#drawAccumulateHistogramImageGray")

# memo
Optimization
* Naive：ほとんど最適化されていない素のC++で書かれたコード．
* OpenCV：OpenCVの最適化された処理を可能な限り使った
* SSE：SSEを使って最適化（昔作った関数で更新がされていない．）
* AVX：AVX/AVX2を使って最適化
* AVX512:現在はコメントアウト
* full optimized:最高レベルの最適化
* single:シングルスレッドの挙動のみ
* parallel：OpenCVのparallel_forを使ったマルチコア実装．
* openmp：OpenMPのparallel_for．コンパイル時コード生成になるため，最大パフォーマンスのためにはライブラリをコンパイルしなおす必要あり．
* コメントなし：NaiveかOpenCVのどちらかと思われる．
# Todo
優先すべきドキュメントは下記のデバッグと開発に頻繁に使用する関数群

* Timer
* ConsoleImage
* Stat
* UpdateCheck
* matInfo
* alphaBlend
* diff
* Plot
* noise
* inlineSIMDFunctions.hpp
* inlineMathFunctions.hpp
* draw