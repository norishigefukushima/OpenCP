stream.hpp
===================
キャッシュを経由せずメモリ書き込みを行うことでキャッシュを汚さないcopyやconvert命令です．
なお，通常はキャッシュを経由させたほうが良く，時間計測などでキャッシュをできるだけ汚したくないときに使用します．


# streamCopy
```cpp
	void streamCopy(cv::InputArray src, cv::OutputArray dst);
```
## Usage
入力をdstにキャッシュを経由せずにコピーします．  
具体的には，`_mm256_stream_ps`や`_mm256_stream_load_ps`などのキャッシュを経由しないロードストア命令を使って実現します．  
なお，8で割り切れない場所はスカラ命令になっているため，割り切れない場合はわずかに汚濁します．  

# streamConvertTo8U
```cpp
	void streamConvertTo8U(cv::InputArray src, cv::OutputArray dst);
```

入力をdstにキャッシュを経由せずに８U型にキャストします．  
Copyと同様に，`_mm256_stream_ps`や`_mm256_stream_load_ps`などのキャッシュを経由しないロードストア命令を使って実現します．  
なお，8で割り切れない場所はスカラ命令になっているため，割り切れない場合はわずかに汚濁します．  

また，8Sから8Uへの変換は，ビット深度が不足するため負の数を0にクリップする処理になります．
