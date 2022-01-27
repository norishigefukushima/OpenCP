#pragma once

#include "common.hpp"

namespace cp
{
	enum class SEPARABLE_METHOD
	{
		SWITCH_VH, //Switching dual kernel (SDK) vertical-filtering then horizontal-filtering
		SWITCH_HV, //Switching dual kernel (SDK) horizontal-filtering then vertical-filtering
		DIRECT_VH, //Usual separable vertical-filtering then horizontal-filtering
		DIRECT_HV  //Usual separable horizontal-filtering then vertical-filtering
	};
	std::string getSeparableMethodName(SEPARABLE_METHOD method);
	std::string getSeparableMethodName(int method);
}