#include "separableFilterCore.hpp"
using namespace std;

namespace cp
{
	string getSeparableMethodName(SEPARABLE_METHOD method)
	{
		string ret = "";
		switch (method)
		{
		case cp::SEPARABLE_METHOD::SWITCH_VH: ret = "SWITCH_VH"; break;
		case cp::SEPARABLE_METHOD::SWITCH_HV: ret = "SWITCH_HV"; break;
		case cp::SEPARABLE_METHOD::DIRECT_VH: ret = "DIRECT_VH"; break;
		case cp::SEPARABLE_METHOD::DIRECT_HV: ret = "DIRECT_HV"; break;
		default: break;
		}
		return ret;
	}

	std::string getSeparableMethodName(int method)
	{
		return getSeparableMethodName(SEPARABLE_METHOD(method));
	}
}