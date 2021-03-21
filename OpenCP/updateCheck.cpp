#include "updateCheck.hpp"

using namespace std;
using namespace cv;

namespace cp
{
	bool UpdateCheck::firstTimeCheck(const bool flag)
	{
		if (isFourceRetTrueFirstTime)
		{
			if (isSkip)
			{
				isSkip = false;
				return true;
			}
			else
			{
				return flag;
			}
		}
		else
		{
			return flag;
		}
	}

	void UpdateCheck::setIsFourceRetTrueFirstTime(const bool flag)
	{
		isFourceRetTrueFirstTime = flag;
	}

	UpdateCheck::UpdateCheck(double p0)
	{
		parameters.push_back(p0);
	}

	UpdateCheck::UpdateCheck(double p0, double p1)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
		parameters.push_back(p7);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
		parameters.push_back(p7);
		parameters.push_back(p8);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
		parameters.push_back(p7);
		parameters.push_back(p8);
		parameters.push_back(p9);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10)
	{
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
		parameters.push_back(p7);
		parameters.push_back(p8);
		parameters.push_back(p9);
		parameters.push_back(p10);
	}

	
	bool UpdateCheck::isUpdate(double p0)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		if (parameters[6] != p6)
		{
			parameters[6] = p6;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		if (parameters[6] != p6)
		{
			parameters[6] = p6;
			ret = true;
		}
		if (parameters[7] != p7)
		{
			parameters[7] = p7;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		if (parameters[6] != p6)
		{
			parameters[6] = p6;
			ret = true;
		}
		if (parameters[7] != p7)
		{
			parameters[7] = p7;
			ret = true;
		}
		if (parameters[8] != p8)
		{
			parameters[8] = p8;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		if (parameters[6] != p6)
		{
			parameters[6] = p6;
			ret = true;
		}
		if (parameters[7] != p7)
		{
			parameters[7] = p7;
			ret = true;
		}
		if (parameters[8] != p8)
		{
			parameters[8] = p8;
			ret = true;
		}
		if (parameters[9] != p9)
		{
			parameters[9] = p9;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10)
	{
		bool ret = false;
		if (parameters[0] != p0)
		{
			parameters[0] = p0;
			ret = true;
		}
		if (parameters[1] != p1)
		{
			parameters[1] = p1;
			ret = true;
		}
		if (parameters[2] != p2)
		{
			parameters[2] = p2;
			ret = true;
		}
		if (parameters[3] != p3)
		{
			parameters[3] = p3;
			ret = true;
		}
		if (parameters[4] != p4)
		{
			parameters[4] = p4;
			ret = true;
		}
		if (parameters[5] != p5)
		{
			parameters[5] = p5;
			ret = true;
		}
		if (parameters[6] != p6)
		{
			parameters[6] = p6;
			ret = true;
		}
		if (parameters[7] != p7)
		{
			parameters[7] = p7;
			ret = true;
		}
		if (parameters[8] != p8)
		{
			parameters[8] = p8;
			ret = true;
		}
		if (parameters[9] != p9)
		{
			parameters[9] = p9;
			ret = true;
		}
		if (parameters[10] != p10)
		{
			parameters[10] = p10;
			ret = true;
		}
		return firstTimeCheck(ret);
	}
}