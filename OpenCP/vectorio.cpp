#include "vectorio.hpp"
#include <fstream>
#include <filesystem>

using namespace std;
namespace cp
{
#pragma region write
	template<typename T>
	void writevector_(vector<T>& buf, const string name)
	{
		std::ofstream ofs(name, std::ios::binary);
		if (!ofs)
		{
			std::cerr << "file open error: " + name << std::endl;
			return;
		}
		ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(T));
		ofs.close();
	}
	void writevector(vector<char>& buf, const string name)
	{
		writevector_<char>(buf, name);
	}
	void writevector(vector<uchar>& buf, const string name)
	{
		writevector_<uchar>(buf, name);
	}
	void writevector(vector<short>& buf, const string name)
	{
		writevector_<short>(buf, name);
	}
	void writevector(vector<ushort>& buf, const string name)
	{
		writevector_<ushort>(buf, name);
	}
	void writevector(vector<int>& buf, const string name)
	{
		writevector_<int>(buf, name);
	}
	void writevector(vector<uint>& buf, const string name)
	{
		writevector_<uint>(buf, name);
	}
	void writevector(vector<float>& buf, const string name)
	{
		writevector_<float>(buf, name);
	}
	void writevector(vector<double>& buf, const string name)
	{
		writevector_<double>(buf, name);
	}

	template<typename T>
	void writevector_(vector<vector<T>>& buf, const string name)
	{
		std::ofstream ofs(name, std::ios::binary);
		if (!ofs)
		{
			std::cerr << "file open error: " + name << std::endl;
			return;
		}
		for (int i = 0; i < buf.size(); i++)
		{
			ofs.write(reinterpret_cast<const char*>(buf[i].data()), buf[i].size() * sizeof(T));
		}
		ofs.close();
	}
	void writevector(vector<vector<char>>& buf, const string name)
	{
		writevector_<char>(buf, name);
	}
	void writevector(vector<vector<uchar>>& buf, const string name)
	{
		writevector_<uchar>(buf, name);
	}
	void writevector(vector<vector<short>>& buf, const string name)
	{
		writevector_<short>(buf, name);
	}
	void writevector(vector<vector<ushort>>& buf, const string name)
	{
		writevector_<ushort>(buf, name);
	}
	void writevector(vector<vector<int>>& buf, const string name)
	{
		writevector_<int>(buf, name);
	}
	void writevector(vector<vector<uint>>& buf, const string name)
	{
		writevector_<uint>(buf, name);
	}
	void writevector(vector <vector<float>>& buf, const string name)
	{
		writevector_<float>(buf, name);
	}
	void writevector(vector<vector<double>>& buf, const string name)
	{
		writevector_<double>(buf, name);
	}
#pragma endregion

#pragma region read
	template<typename T>
	void readvector_(vector<T>& buf, const string name)
	{
		const int size = (int)std::filesystem::file_size(name);
		buf.resize(size / sizeof(T));
		std::ifstream ifs(name, std::ios::binary);
		if (!ifs)
		{
			std::cerr << "file open error: " + name << std::endl;
			return;
		}
		ifs.read(reinterpret_cast<char*>(buf.data()), size);
		ifs.close();
	}
	void readvector(vector<char>& buf, const string name)
	{
		readvector_<char>(buf, name);
	}
	void readvector(vector<uchar>& buf, const string name)
	{
		readvector_<uchar>(buf, name);
	}
	void readvector(vector<short>& buf, const string name)
	{
		readvector_<short>(buf, name);
	}
	void readvector(vector<ushort>& buf, const string name)
	{
		readvector_<ushort>(buf, name);
	}
	void readvector(vector<int>& buf, const string name)
	{
		readvector_<int>(buf, name);
	}
	void readvector(vector<uint>& buf, const string name)
	{
		readvector_<uint>(buf, name);
	}
	void readvector(vector<float>& buf, const string name)
	{
		readvector_<float>(buf, name);
	}
	void readvector(vector<double>& buf, const string name)
	{
		readvector_<double>(buf, name);
	}

	template<typename T>
	void readvector_(vector<vector<T>>& buf, const string name)
	{
		const int size = (int)std::filesystem::file_size(name);
		vector<T> tmp(size / sizeof(T));
		std::ifstream ifs(name, std::ios::binary);
		if (!ifs)
		{
			std::cerr << "file open error: " + name << std::endl;
			return;
		}
		ifs.read(reinterpret_cast<char*>(tmp.data()), size);
		int idx = 0;
		for (int j = 0; j < buf.size(); j++)
		{
			for (int i = 0; i < buf[j].size(); i++)
			{
				buf[j][i] = tmp[idx++];
			}
		}

		ifs.close();
	}
	void readvector(vector<vector<char>>& buf, const string name)
	{
		readvector_<char>(buf, name);
	}
	void readvector(vector<vector<uchar>>& buf, const string name)
	{
		readvector_<uchar>(buf, name);
	}
	void readvector(vector<vector<short>>& buf, const string name)
	{
		readvector_<short>(buf, name);
	}
	void readvector(vector<vector<ushort>>& buf, const string name)
	{
		readvector_<ushort>(buf, name);
	}
	void readvector(vector<vector<int>>& buf, const string name)
	{
		readvector_<int>(buf, name);
	}
	void readvector(vector<vector<uint>>& buf, const string name)
	{
		readvector_<uint>(buf, name);
	}
	void readvector(vector<vector<float>>& buf, const string name)
	{
		readvector_<float>(buf, name);
	}
	void readvector(vector<vector<double>>& buf, const string name)
	{
		readvector_<double>(buf, name);
	}
#pragma endregion
}