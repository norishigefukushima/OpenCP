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

#pragma region init
	void UpdateCheck::init(double p0)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
	}

	void UpdateCheck::init(double p0, double p1)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
	}

	void UpdateCheck::init(double p0, double p1, double p2)
	{
		parameters.clear();
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7)
	{
		parameters.clear();
		parameters.resize(0);
		parameters.push_back(p0);
		parameters.push_back(p1);
		parameters.push_back(p2);
		parameters.push_back(p3);
		parameters.push_back(p4);
		parameters.push_back(p5);
		parameters.push_back(p6);
		parameters.push_back(p7);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8)
	{
		parameters.clear();
		parameters.resize(0);
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

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9)
	{
		parameters.clear();
		parameters.resize(0);
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

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10)
	{
		parameters.clear();
		parameters.resize(0);
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

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
		parameters.push_back(p12);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
		parameters.push_back(p12);
		parameters.push_back(p13);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
		parameters.push_back(p12);
		parameters.push_back(p13);
		parameters.push_back(p14);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
		parameters.push_back(p12);
		parameters.push_back(p13);
		parameters.push_back(p14);
		parameters.push_back(p15);
	}

	void UpdateCheck::init(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16)
	{
		parameters.clear();
		parameters.resize(0);
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
		parameters.push_back(p11);
		parameters.push_back(p12);
		parameters.push_back(p13);
		parameters.push_back(p14);
		parameters.push_back(p15);
		parameters.push_back(p16);
	}
#pragma endregion

#pragma region constructor
	UpdateCheck::UpdateCheck()
	{
		;
	}

	UpdateCheck::UpdateCheck(double p0)
	{
		init(p0);
	}

	UpdateCheck::UpdateCheck(double p0, double p1)
	{
		init(p0, p1);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2)
	{
		init(p0, p1, p2);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3)
	{
		init(p0, p1, p2, p3);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4)
	{
		init(p0, p1, p2, p3, p4);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5)
	{
		init(p0, p1, p2, p3, p4, p5);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6)
	{
		init(p0, p1, p2, p3, p4, p5, p6);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
	}

	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
	}
	UpdateCheck::UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16)
	{
		init(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
	}
#pragma endregion

#pragma region isUpdate
	bool UpdateCheck::isUpdate(double p0)
	{
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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
		push();
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

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		if (parameters[12] != p12)
		{
			parameters[12] = p12;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		if (parameters[12] != p12)
		{
			parameters[12] = p12;
			ret = true;
		}
		if (parameters[13] != p13)
		{
			parameters[13] = p13;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		if (parameters[12] != p12)
		{
			parameters[12] = p12;
			ret = true;
		}
		if (parameters[13] != p13)
		{
			parameters[13] = p13;
			ret = true;
		}
		if (parameters[14] != p14)
		{
			parameters[14] = p14;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		if (parameters[12] != p12)
		{
			parameters[12] = p12;
			ret = true;
		}
		if (parameters[13] != p13)
		{
			parameters[13] = p13;
			ret = true;
		}
		if (parameters[14] != p14)
		{
			parameters[14] = p14;
			ret = true;
		}
		if (parameters[15] != p15)
		{
			parameters[15] = p15;
			ret = true;
		}
		return firstTimeCheck(ret);
	}

	bool UpdateCheck::isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16)
	{
		push();
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
		if (parameters[11] != p11)
		{
			parameters[11] = p11;
			ret = true;
		}
		if (parameters[12] != p12)
		{
			parameters[12] = p12;
			ret = true;
		}
		if (parameters[13] != p13)
		{
			parameters[13] = p13;
			ret = true;
		}
		if (parameters[14] != p14)
		{
			parameters[14] = p14;
			ret = true;
		}
		if (parameters[15] != p15)
		{
			parameters[15] = p15;
			ret = true;
		}
		if (parameters[16] != p16)
		{
			parameters[16] = p16;
			ret = true;
		}
		return firstTimeCheck(ret);
	}
#pragma endregion

#pragma region get64
	void UpdateCheck::get(double& p0)
	{
		p0 = parameters[0];
	}
	void UpdateCheck::get(double& p0, double& p1)
	{
		p0 = parameters[0];
		p1 = parameters[1];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
		p12 = parameters[12];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
		p12 = parameters[12];
		p13 = parameters[13];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
		p12 = parameters[12];
		p13 = parameters[13];
		p14 = parameters[14];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14, double& p15)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
		p12 = parameters[12];
		p13 = parameters[13];
		p14 = parameters[14];
		p15 = parameters[15];
	}
	void UpdateCheck::get(double& p0, double& p1, double& p2, double& p3, double& p4, double& p5, double& p6, double& p7, double& p8, double& p9, double& p10, double& p11, double& p12, double& p13, double& p14, double& p15, double& p16)
	{
		p0 = parameters[0];
		p1 = parameters[1];
		p2 = parameters[2];
		p3 = parameters[3];
		p4 = parameters[4];
		p5 = parameters[5];
		p6 = parameters[6];
		p7 = parameters[7];
		p8 = parameters[8];
		p9 = parameters[9];
		p10 = parameters[10];
		p11 = parameters[11];
		p12 = parameters[12];
		p13 = parameters[13];
		p14 = parameters[14];
		p15 = parameters[15];
		p16 = parameters[16];
	}
#pragma endregion
#pragma region get_int32
	void UpdateCheck::get(int& p0)
	{
		p0 = (int)parameters[0];
	}
	void UpdateCheck::get(int& p0, int& p1)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
		p12 = (int)parameters[12];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
		p12 = (int)parameters[12];
		p13 = (int)parameters[13];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
		p12 = (int)parameters[12];
		p13 = (int)parameters[13];
		p14 = (int)parameters[14];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14, int& p15)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
		p12 = (int)parameters[12];
		p13 = (int)parameters[13];
		p14 = (int)parameters[14];
		p15 = (int)parameters[15];
	}
	void UpdateCheck::get(int& p0, int& p1, int& p2, int& p3, int& p4, int& p5, int& p6, int& p7, int& p8, int& p9, int& p10, int& p11, int& p12, int& p13, int& p14, int& p15, int& p16)
	{
		p0 = (int)parameters[0];
		p1 = (int)parameters[1];
		p2 = (int)parameters[2];
		p3 = (int)parameters[3];
		p4 = (int)parameters[4];
		p5 = (int)parameters[5];
		p6 = (int)parameters[6];
		p7 = (int)parameters[7];
		p8 = (int)parameters[8];
		p9 = (int)parameters[9];
		p10 = (int)parameters[10];
		p11 = (int)parameters[11];
		p12 = (int)parameters[12];
		p13 = (int)parameters[13];
		p14 = (int)parameters[14];
		p15 = (int)parameters[15];
		p16 = (int)parameters[16];
	}
#pragma endregion
	void UpdateCheck::push()
	{
		if (prevparameters.size() != parameters.size()) prevparameters.resize(parameters.size());
		for (int i = 0; i < parameters.size(); i++)
		{
			prevparameters[i] = parameters[i];
		}
	}
	void UpdateCheck::pop()
	{
		for (int i = 0; i < parameters.size(); i++)
		{
			parameters[i] = prevparameters[i];
		}
	}

	void UpdateCheck::print(int num_args)
	{
		for (int i = 0; i < num_args; i++)
		{
			cout << i << ": " << parameters[i] << endl;
		}
	}
}
