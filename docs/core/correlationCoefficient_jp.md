# 相関係数

OpenCP は Pearson の積率相関係数、Spearman の順位相関係数、Kendall の $\tau_b$、Chatterjee の $\xi$ を提供します。

## 参照値をキャッシュして4指標をまとめて計算する

同じ主観評価値に対して多数の候補スコアを評価する最適化処理では、`CorrelationCoefficientSet` を使用します。`setReference()` は参照側の中心化値、順位、tie統計、Kendall用のordinal rank、Chatterjee用の分母を一度だけ計算します。`compute()` は候補をIntel x86-simd-sortで1回だけargsortし、その結果をSROCC、KROCC、CROCCで共有します。

```cpp
std::vector<double> mos = /* 固定された主観評価値 */;
std::vector<double> score = /* 候補スコア */;

cp::CorrelationCoefficientSet correlation;
correlation.setReference(mos);

// 順位差の集計をdoubleで行う正式評価経路。
correlation.setAccumulation(
    cp::CorrelationCoefficientSet::Accumulation::Float64);
const auto result = correlation.compute(score);

std::cout << result.pearson << '\n';
std::cout << result.spearman << '\n';
std::cout << result.kendall << '\n';
std::cout << result.chatterjee << '\n';
```

5パラメータfit後のPLCCだけを計算する場合は、ソートを行わない `computePearson(fittedScore)` を使えます。

`CorrelationCoefficientSet` は計算用バッファを再利用するため、同じインスタンスの `compute()` を複数スレッドから同時に呼び出してはいけません。並列最適化ではworkerごとにインスタンスを1個用意してください。参照値と候補値は有限値で、要素数が一致し、2要素以上である必要があります。定数列など分母が0になる場合はNaNを返します。

## 集計精度

`Accumulation::Float32` はfloat順位をAVX2/FMAで集計し、256要素ごとにdoubleへ縮約します。速度を優先する探索向けです。`Accumulation::Float64` はfloat順位をレジスタ上でdoubleへ変換し、AVX2/FMAで集計します。順位配列をdoubleで複製せず、正式評価時の精度を高めます。

## 個別クラス

- `PearsonCorrelationCoefficient`: PLCCを計算します。
- `SpearmanRankOrderCorrelationCoefficient`: SROCCを計算します。`setReference()` と `computeUsingReference()` による参照順位のキャッシュ、および `Float32` / `Float64` 集計を選択できます。
- `KendallRankOrderCorrelationCoefficient`: tie補正を含むKendall $\tau_b$ を $O(N\log N)$ で計算します。
- `ChatterjeeCorrelationCoefficient`: responseのtie補正を含むChatterjee $\xi$ を計算します。

個別クラスは単独指標だけが必要な用途向けです。4指標を繰り返し計算する用途では、候補のソートを共有できる `CorrelationCoefficientSet` を優先してください。
