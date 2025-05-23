#pragma once

#include "common.hpp"

namespace cp
{
	//ex: string_split(string, `,`) //comma sep
	CP_EXPORT std::vector<std::string> string_split(const std::string& str, const char splitchar);

	//ex: string_remove(string, ".png") 
	CP_EXPORT std::string string_remove(const std::string& src, const std::string& toRemove);

	//ex 1000->1,000
	CP_EXPORT std::string string_format_with_commas(int value);
}
