#include "string_cp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	std::vector<std::string> string_split(const std::string& str, const char del)
	{
		int first = 0;
		int last = (int)str.find_first_of(del);


		std::vector<std::string> result;

		while (first < str.size()) {
			std::string subStr(str, first, last - first);

			result.push_back(subStr);

			first = last + 1;
			last = (int)str.find_first_of(del, first);

			if (last == std::string::npos) {
				last = (int)str.size();
			}
		}

		return result;
	}

	std::string string_remove(const std::string& src, const std::string& toRemove)
	{
		size_t pos;
		std::string dst = src;
		while ((pos = dst.find(toRemove)) != std::string::npos) 
		{
			dst.erase(pos, toRemove.length());
		}
		return dst;
	}

	std::string string_format_with_commas(int value)
	{
		std::string buffer = std::to_string(value);
		std::string result;

		int len = (int)buffer.length();
		int count = 0;

		for (int i = len - 1; i >= 0; --i) {
			result.insert(0, 1, buffer[i]);
			count++;
			if (count == 3 && i != 0 && buffer[i - 1] != '-') {
				result.insert(0, 1, ',');
				count = 0;
			}
		}

		return result;
	}
}