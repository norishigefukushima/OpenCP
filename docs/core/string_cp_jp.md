vectorio.hpp
===================
std::stringを拡張する関数群

# string_split
```cpp
	std::vector<std::string> string_split(const std::string& str, const char splitchar);
```

## Usage
stringをセパレータ記号splitcharに応じてvectorに分解します。
例えば下記にように使えばカンマ区切りの値をstringのベクター配列に分解できます
```cpp
string_split(strings, `,`) //comma sep
```

# string_remove
```cpp
	std::string string_remove(const std::string& src, const std::string& toRemove);
```

## Usage
入力stringから指定のstring文字列を削除します．
例えば下記にように使えば，.pngの文字列を消した文字列が取得できます．
```cpp
string removed = string_remove(string, ".png") ;
```

# string_remove
```cpp
	std::string string_format_with_commas(int value);
```

## Usage
整数を，3桁のカンマ区切りの文字列に変換します．入力stringから指定のstring文字列を削除します．

例えば，1000は1,000に変換されます．



