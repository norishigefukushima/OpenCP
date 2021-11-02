#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testStreamConvert8U()
{
	Mat a64(Size(512, 512), CV_64F);
	Mat a32(Size(512, 512), CV_32F);
	Mat a32s(Size(512, 512), CV_32S);
	Mat a16s(Size(512, 512), CV_16S);
	Mat a16u(Size(512, 512), CV_16U);
	Mat a8s(Size(512, 512), CV_8S);
	Mat a8u(Size(512, 512), CV_8U);

	randu(a64, -DBL_MAX, DBL_MAX);
	randu(a32, -FLT_MAX, FLT_MAX);
	randu(a32s, INT_MIN, INT_MAX);
	randu(a16s, SHRT_MIN, SHRT_MAX);
	randu(a16u, 0, USHRT_MAX);
	randu(a8s, CHAR_MIN, CHAR_MAX);
	randu(a8u, 0, UCHAR_MAX);

	Mat ans, tst;

	a64.convertTo(ans, CV_8U);
	streamConvertTo8U(a64, tst);
	cout << "64F "; cp::isSame(ans, tst);

	a32.convertTo(ans, CV_8U);
	streamConvertTo8U(a32, tst);
	cout << "32F "; cp::isSame(ans, tst);

	a32s.convertTo(ans, CV_8U);
	streamConvertTo8U(a32s, tst);
	cout << "32S "; cp::isSame(ans, tst);

	a16s.convertTo(ans, CV_8U);
	streamConvertTo8U(a16s, tst);
	cout << "16S "; cp::isSame(ans, tst);

	a16u.convertTo(ans, CV_8U);
	streamConvertTo8U(a16u, tst);
	cout << "16U "; cp::isSame(ans, tst);

	a8s.convertTo(ans, CV_8U);
	streamConvertTo8U(a8s, tst);
	cout << "8S "; cp::isSame(ans, tst);

	a8u.convertTo(ans, CV_8U);
	streamConvertTo8U(a8u, tst);
	cout << "8U "; cp::isSame(ans, tst);
}
