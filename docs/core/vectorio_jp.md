vectorio.hpp
===================
vector配列をそのままダンプする関数群です．

# writevector
```cpp
	void writevector(std::vector<char>& buf, const std::string name);
	void writevector(std::vector<uchar>& buf, const std::string name);
	void writevector(std::vector<short>& buf, const std::string name);
	void writevector(std::vector<ushort>& buf, const std::string name);
	void writevector(std::vector<int>& buf, const std::string name);
	void writevector(std::vector<uint>& buf, const std::string name);
	void writevector(std::vector<float>& buf, const std::string name);
	void writevector(std::vector<double>& buf, const std::string name);

	void writevector(std::vector<std::vector<char>>& buf, const std::string name);
	void writevector(std::vector<std::vector<uchar>>& buf, const std::string name);
	void writevector(std::vector<std::vector<short>>& buf, const std::string name);
	void writevector(std::vector<std::vector<ushort>>& buf, const std::string name);
	void writevector(std::vector<std::vector<int>>& buf, const std::string name);
	void writevector(std::vector<std::vector<uint>>& buf, const std::string name);
	void writevector(std::vector<std::vector<float>>& buf, const std::string name);
	void writevector(std::vector<std::vector<double>>& buf, const std::string name);
```

## Usage
vector配列を生のバイナリとしてダンプします．
`char`,`uchar`,`short`,`ushort`, `int`, `uint`, `float`, `double`に対応しています．
また，2次元配列のvectorまで対応しています．

# readvector
```cpp
	void readvector(std::vector<char>& buf, const std::string name);
	void readvector(std::vector<uchar>& buf, const std::string name);
	void readvector(std::vector<short>& buf, const std::string name);
	void readvector(std::vector<ushort>& buf, const std::string name);
	void readvector(std::vector<int>& buf, const std::string name);
	void readvector(std::vector<uint>& buf, const std::string name);
	void readvector(std::vector<float>& buf, const std::string name);
	void readvector(std::vector<double>& buf, const std::string name);
	
	void readvector(std::vector<std::vector<char>>& buf, const std::string name);
	void readvector(std::vector<std::vector<uchar>>& buf, const std::string name);
	void readvector(std::vector<std::vector<short>>& buf, const std::string name);
	void readvector(std::vector<std::vector<ushort>>& buf, const std::string name);
	void readvector(std::vector<std::vector<int>>& buf, const std::string name);
	void readvector(std::vector<std::vector<uint>>& buf, const std::string name);
	void readvector(std::vector<std::vector<float>>& buf, const std::string name);
	void readvector(std::vector<std::vector<double>>& buf, const std::string name);
```

## Usage
writevectorされた配列を読み込みます．
2次元配列のvectorまで対応しています．