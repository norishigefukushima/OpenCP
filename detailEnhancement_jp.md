detailEnhancement.hpp
================

内部実装は下記のようになっており，基本的に平滑化フィルタの差分をブーストするもの．
`src+boost(src-g*src)`で変換，．

```cpp
void detailEnhancementBilateral(InputArray src, OutputArray dest, const int r, const float sigma_color, const float sigma_space, const float boost)
	{
		const int d = 2 * r + 1;
		Mat smooth;
		bilateralFilter(src, smooth, Size(d, d), sigma_color, sigma_space);
		addWeighted(src, 1.0 + boost, smooth, -boost, 0, dest);
	}
```
# detailEnhancementBox
```cpp
void detailEnhancementBox(InputArray src, OutputArray dest, const int r, const float boost)
```
ボックスフィルタで詳細強調

# detailEnhancementGauss
```cpp
void detailEnhancementGauss(InputArray src, OutputArray dest, const int r, const float sigma_space, const float boost)
```
ガウシアンフィルタで詳細強調

# detailEnhancementBilateral
```cpp
void detailEnhancementBilateral(InputArray src, OutputArray dest, const int r, const float sigma_color, const float sigma_space, const float boost)
```
バイラテラルフィルタで詳細強調

# detailEnhancementGuided
```cpp
void detailEnhancementGuided(InputArray src_, OutputArray dest, const int r, const float eps, const float boost)
```
ガイデットフィルタで詳細強調