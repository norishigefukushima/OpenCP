#pragma once

#include "common.hpp"

namespace cp
{
	CP_EXPORT void writevector(std::vector<char>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<uchar>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<short>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<ushort>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<int>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<uint>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<float>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<double>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<char>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<uchar>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<short>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<ushort>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<int>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<uint>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<float>>& buf, const std::string name);
	CP_EXPORT void writevector(std::vector<std::vector<double>>& buf, const std::string name);

	CP_EXPORT void readvector(std::vector<char>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<uchar>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<short>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<ushort>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<int>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<uint>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<float>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<double>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<char>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<uchar>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<short>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<ushort>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<int>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<uint>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<float>>& buf, const std::string name);
	CP_EXPORT void readvector(std::vector<std::vector<double>>& buf, const std::string name);
}