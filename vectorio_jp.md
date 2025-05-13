vectorio.hpp
===================
キャッシュを経由せずメモリ書き込みを行うことでキャッシュを汚さないcopyやconvert命令です．
なお，通常はキャッシュを経由させたほうが良く，時間計測などでキャッシュをできるだけ汚したくないときに使用します．

# writevector
CP_EXPORT 


# writevector
```cpp
	void writevector(std::vector<char>& buf, const std::string name);
	void writevector(std::vector<uchar>& buf, const std::string name);
	...
	void writevector(std::vector<std::vector<char>>& buf, const std::string name);
	void writevector(std::vector<std::vector<uchar>>& buf, const std::string name);
```

## Usage
vector配列を生のバイナリとしてダンプします．
`char`,`uchar`,`short`,`ushort`, `int`, `uint`, `float`, `double`に対応しています．
また，2次元配列のvectorまで対応しています．

# readvector
```cpp
	void readvector(std::vector<char>& buf, const std::string name);
	void readvector(std::vector<uchar>& buf, const std::string name);
	...
	void readvector(std::vector<std::vector<char>>& buf, const std::string name);
	void readvector(std::vector<std::vector<uchar>>& buf, const std::string name);
```

## Usage
writevectorされた配列を読み込みます．
2次元配列のvectorまで対応しています．