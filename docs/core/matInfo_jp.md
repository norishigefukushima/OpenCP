matInfo.hpp
===================
cv::InputArrayの情報の型情報や中の統計情報を取得する関数．

# showMatInfo
```cpp
	CP_EXPORT void showMatInfo(cv::InputArray src, std::string name = "Mat", const bool isStatInfo = true);
```

## Usage
* srcの型を表示し，サイズ，デプス，チャネル情報を表示する．  
* また，行列が連続しているか，ROI（submatrix）を持っているかを表示する．  
* もし空の場合はemptyであることを表示し，0初期化された状態であるならその状態であることを表示する．  
* nameに適当に名前を付けると，printfデバッグ時に，どのshowMatInfoが呼ばれたか分かる．
	* ただし，引数の変数名を値に取れる下のマクロを使ったほうが便利．
* `isStatInfo`が`true`の場合は，行列の平均値，最小値，最大値を表示する．
表示される型は以下で，主の機能実装は`Mat`と`vector<Mat>`のみ

```cpp
Mat
vector<Mat>
GpuMat //typeの表示のみ
vector<GpuMat> //typeの表示のみ
UMat //typeの表示のみ
vector<UMat> //typeの表示のみ
```

# print_matinfo(a)
```cpp
#define print_matinfo(a) cp::showMatInfo(a, #a, false)
```
主に行列要素のprintfデバッグ用途の関数．
コンパイル時のマクロ展開で，第二引数に変数名を入れて関数を呼び出す．
ただし，詳細情報は省略．

# print_matinfo_detail(a)
```cpp
#define print_matinfo_detail(a) cp::showMatInfo(a, #a)
```

## Sample

コード
```cpp
Mat a = Mat::ones(3,3,CV_32F);
print_matinfo(a);
```

出力
```console
type    : Mat
name    : a
size    : [3 x 3]
channel : 1
depth   : 32F
continue: true
ROI     : false
```
