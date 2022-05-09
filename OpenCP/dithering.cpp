#include "dithering.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

//#define FloydSteinbergTrackBar

namespace cp
{
	std::string getDitheringMethodName(const int method)
	{
		string ret = "";
		switch (method)
		{
		case OSTROMOUKHOW:		ret = "OSTROMOUKHOW"; break;
		case FLOYD_STEINBERG:	ret = "FLOYD_STEINBERG"; break;
		case SIERRA2:			ret = "SIERRA2"; break;
		case SIERRA3:			ret = "SIERRA3"; break;
		case JARVIS:			ret = "JARVIS"; break;
		case STUCKI:			ret = "STUCKI"; break;
		case BURKES:			ret = "BURKES"; break;
		case STEAVENSON:		ret = "STEAVENSON"; break;

		default:
			break;
		}

		return ret;
	}

	std::string getDitheringOrderName(const int method)
	{
		string ret;
		switch (method)
		{
		case FORWARD:				ret = "FORWARD"; break;
		case MEANDERING:			ret = "MEANDERING"; break;
		case IN2OUT:				ret = "IN2OUT"; break;
		case OUT2IN:				ret = "OUT2IN"; break;
		case FOURDIRECTION:			ret = "FOURDIRECTION"; break;
		case FOURDIRECTIONIN2OUT:	ret = "FOURDIRECTIONIN2OUT"; break;
		default: ret = "no support"; break;
		}
		return ret;
	}

	typedef struct t_three_coefs {
		int i_r;
		int i_dl;
		int i_d;
		int i_sum;

	}t_three_coefs;

	t_three_coefs var_coefs_tab[256] = {
		13,     0,     5,    18,     /*    0 */
		13,     0,     5,    18,     /*    1 */
		21,     0,    10,    31,     /*    2 */
		7,     0,     4,    11,     /*    3 */
		8,     0,     5,    13,     /*    4 */
		47,     3,    28,    78,     /*    5 */
		23,     3,    13,    39,     /*    6 */
		15,     3,     8,    26,     /*    7 */
		22,     6,    11,    39,     /*    8 */
		43,    15,    20,    78,     /*    9 */
		7,     3,     3,    13,     /*   10 */
		501,   224,   211,   936,     /*   11 */
		249,   116,   103,   468,     /*   12 */
		165,    80,    67,   312,     /*   13 */
		123,    62,    49,   234,     /*   14 */
		489,   256,   191,   936,     /*   15 */
		81,    44,    31,   156,     /*   16 */
		483,   272,   181,   936,     /*   17 */
		60,    35,    22,   117,     /*   18 */
		53,    32,    19,   104,     /*   19 */
		237,   148,    83,   468,     /*   20 */
		471,   304,   161,   936,     /*   21 */
		3,     2,     1,     6,     /*   22 */
		459,   304,   161,   924,     /*   23 */
		38,    25,    14,    77,     /*   24 */
		453,   296,   175,   924,     /*   25 */
		225,   146,    91,   462,     /*   26 */
		149,    96,    63,   308,     /*   27 */
		111,    71,    49,   231,     /*   28 */
		63,    40,    29,   132,     /*   29 */
		73,    46,    35,   154,     /*   30 */
		435,   272,   217,   924,     /*   31 */
		108,    67,    56,   231,     /*   32 */
		13,     8,     7,    28,     /*   33 */
		213,   130,   119,   462,     /*   34 */
		423,   256,   245,   924,     /*   35 */
		5,     3,     3,    11,     /*   36 */
		281,   173,   162,   616,     /*   37 */
		141,    89,    78,   308,     /*   38 */
		283,   183,   150,   616,     /*   39 */
		71,    47,    36,   154,     /*   40 */
		285,   193,   138,   616,     /*   41 */
		13,     9,     6,    28,     /*   42 */
		41,    29,    18,    88,     /*   43 */
		36,    26,    15,    77,     /*   44 */
		289,   213,   114,   616,     /*   45 */
		145,   109,    54,   308,     /*   46 */
		291,   223,   102,   616,     /*   47 */
		73,    57,    24,   154,     /*   48 */
		293,   233,    90,   616,     /*   49 */
		21,    17,     6,    44,     /*   50 */
		295,   243,    78,   616,     /*   51 */
		37,    31,     9,    77,     /*   52 */
		27,    23,     6,    56,     /*   53 */
		149,   129,    30,   308,     /*   54 */
		299,   263,    54,   616,     /*   55 */
		75,    67,    12,   154,     /*   56 */
		43,    39,     6,    88,     /*   57 */
		151,   139,    18,   308,     /*   58 */
		303,   283,    30,   616,     /*   59 */
		38,    36,     3,    77,     /*   60 */
		305,   293,    18,   616,     /*   61 */
		153,   149,     6,   308,     /*   62 */
		307,   303,     6,   616,     /*   63 */
		1,     1,     0,     2,     /*   64 */
		101,   105,     2,   208,     /*   65 */
		49,    53,     2,   104,     /*   66 */
		95,   107,     6,   208,     /*   67 */
		23,    27,     2,    52,     /*   68 */
		89,   109,    10,   208,     /*   69 */
		43,    55,     6,   104,     /*   70 */
		83,   111,    14,   208,     /*   71 */
		5,     7,     1,    13,     /*   72 */
		172,   181,    37,   390,     /*   73 */
		97,    76,    22,   195,     /*   74 */
		72,    41,    17,   130,     /*   75 */
		119,    47,    29,   195,     /*   76 */
		4,     1,     1,     6,     /*   77 */
		4,     1,     1,     6,     /*   78 */
		4,     1,     1,     6,     /*   79 */
		4,     1,     1,     6,     /*   80 */
		4,     1,     1,     6,     /*   81 */
		4,     1,     1,     6,     /*   82 */
		4,     1,     1,     6,     /*   83 */
		4,     1,     1,     6,     /*   84 */
		4,     1,     1,     6,     /*   85 */
		65,    18,    17,   100,     /*   86 */
		95,    29,    26,   150,     /*   87 */
		185,    62,    53,   300,     /*   88 */
		30,    11,     9,    50,     /*   89 */
		35,    14,    11,    60,     /*   90 */
		85,    37,    28,   150,     /*   91 */
		55,    26,    19,   100,     /*   92 */
		80,    41,    29,   150,     /*   93 */
		155,    86,    59,   300,     /*   94 */
		5,     3,     2,    10,     /*   95 */
		5,     3,     2,    10,     /*   96 */
		5,     3,     2,    10,     /*   97 */
		5,     3,     2,    10,     /*   98 */
		5,     3,     2,    10,     /*   99 */
		5,     3,     2,    10,     /*  100 */
		5,     3,     2,    10,     /*  101 */
		5,     3,     2,    10,     /*  102 */
		5,     3,     2,    10,     /*  103 */
		5,     3,     2,    10,     /*  104 */
		5,     3,     2,    10,     /*  105 */
		5,     3,     2,    10,     /*  106 */
		5,     3,     2,    10,     /*  107 */
		305,   176,   119,   600,     /*  108 */
		155,    86,    59,   300,     /*  109 */
		105,    56,    39,   200,     /*  110 */
		80,    41,    29,   150,     /*  111 */
		65,    32,    23,   120,     /*  112 */
		55,    26,    19,   100,     /*  113 */
		335,   152,   113,   600,     /*  114 */
		85,    37,    28,   150,     /*  115 */
		115,    48,    37,   200,     /*  116 */
		35,    14,    11,    60,     /*  117 */
		355,   136,   109,   600,     /*  118 */
		30,    11,     9,    50,     /*  119 */
		365,   128,   107,   600,     /*  120 */
		185,    62,    53,   300,     /*  121 */
		25,     8,     7,    40,     /*  122 */
		95,    29,    26,   150,     /*  123 */
		385,   112,   103,   600,     /*  124 */
		65,    18,    17,   100,     /*  125 */
		395,   104,   101,   600,     /*  126 */
		4,     1,     1,     6,     /*  127 */
		4,     1,     1,     6,     /*  128 */
		395,   104,   101,   600,     /*  129 */
		65,    18,    17,   100,     /*  130 */
		385,   112,   103,   600,     /*  131 */
		95,    29,    26,   150,     /*  132 */
		25,     8,     7,    40,     /*  133 */
		185,    62,    53,   300,     /*  134 */
		365,   128,   107,   600,     /*  135 */
		30,    11,     9,    50,     /*  136 */
		355,   136,   109,   600,     /*  137 */
		35,    14,    11,    60,     /*  138 */
		115,    48,    37,   200,     /*  139 */
		85,    37,    28,   150,     /*  140 */
		335,   152,   113,   600,     /*  141 */
		55,    26,    19,   100,     /*  142 */
		65,    32,    23,   120,     /*  143 */
		80,    41,    29,   150,     /*  144 */
		105,    56,    39,   200,     /*  145 */
		155,    86,    59,   300,     /*  146 */
		305,   176,   119,   600,     /*  147 */
		5,     3,     2,    10,     /*  148 */
		5,     3,     2,    10,     /*  149 */
		5,     3,     2,    10,     /*  150 */
		5,     3,     2,    10,     /*  151 */
		5,     3,     2,    10,     /*  152 */
		5,     3,     2,    10,     /*  153 */
		5,     3,     2,    10,     /*  154 */
		5,     3,     2,    10,     /*  155 */
		5,     3,     2,    10,     /*  156 */
		5,     3,     2,    10,     /*  157 */
		5,     3,     2,    10,     /*  158 */
		5,     3,     2,    10,     /*  159 */
		5,     3,     2,    10,     /*  160 */
		155,    86,    59,   300,     /*  161 */
		80,    41,    29,   150,     /*  162 */
		55,    26,    19,   100,     /*  163 */
		85,    37,    28,   150,     /*  164 */
		35,    14,    11,    60,     /*  165 */
		30,    11,     9,    50,     /*  166 */
		185,    62,    53,   300,     /*  167 */
		95,    29,    26,   150,     /*  168 */
		65,    18,    17,   100,     /*  169 */
		4,     1,     1,     6,     /*  170 */
		4,     1,     1,     6,     /*  171 */
		4,     1,     1,     6,     /*  172 */
		4,     1,     1,     6,     /*  173 */
		4,     1,     1,     6,     /*  174 */
		4,     1,     1,     6,     /*  175 */
		4,     1,     1,     6,     /*  176 */
		4,     1,     1,     6,     /*  177 */
		4,     1,     1,     6,     /*  178 */
		119,    47,    29,   195,     /*  179 */
		72,    41,    17,   130,     /*  180 */
		97,    76,    22,   195,     /*  181 */
		172,   181,    37,   390,     /*  182 */
		5,     7,     1,    13,     /*  183 */
		83,   111,    14,   208,     /*  184 */
		43,    55,     6,   104,     /*  185 */
		89,   109,    10,   208,     /*  186 */
		23,    27,     2,    52,     /*  187 */
		95,   107,     6,   208,     /*  188 */
		49,    53,     2,   104,     /*  189 */
		101,   105,     2,   208,     /*  190 */
		1,     1,     0,     2,     /*  191 */
		307,   303,     6,   616,     /*  192 */
		153,   149,     6,   308,     /*  193 */
		305,   293,    18,   616,     /*  194 */
		38,    36,     3,    77,     /*  195 */
		303,   283,    30,   616,     /*  196 */
		151,   139,    18,   308,     /*  197 */
		43,    39,     6,    88,     /*  198 */
		75,    67,    12,   154,     /*  199 */
		299,   263,    54,   616,     /*  200 */
		149,   129,    30,   308,     /*  201 */
		27,    23,     6,    56,     /*  202 */
		37,    31,     9,    77,     /*  203 */
		295,   243,    78,   616,     /*  204 */
		21,    17,     6,    44,     /*  205 */
		293,   233,    90,   616,     /*  206 */
		73,    57,    24,   154,     /*  207 */
		291,   223,   102,   616,     /*  208 */
		145,   109,    54,   308,     /*  209 */
		289,   213,   114,   616,     /*  210 */
		36,    26,    15,    77,     /*  211 */
		41,    29,    18,    88,     /*  212 */
		13,     9,     6,    28,     /*  213 */
		285,   193,   138,   616,     /*  214 */
		71,    47,    36,   154,     /*  215 */
		283,   183,   150,   616,     /*  216 */
		141,    89,    78,   308,     /*  217 */
		281,   173,   162,   616,     /*  218 */
		5,     3,     3,    11,     /*  219 */
		423,   256,   245,   924,     /*  220 */
		213,   130,   119,   462,     /*  221 */
		13,     8,     7,    28,     /*  222 */
		108,    67,    56,   231,     /*  223 */
		435,   272,   217,   924,     /*  224 */
		73,    46,    35,   154,     /*  225 */
		63,    40,    29,   132,     /*  226 */
		111,    71,    49,   231,     /*  227 */
		149,    96,    63,   308,     /*  228 */
		225,   146,    91,   462,     /*  229 */
		453,   296,   175,   924,     /*  230 */
		38,    25,    14,    77,     /*  231 */
		459,   304,   161,   924,     /*  232 */
		3,     2,     1,     6,     /*  233 */
		471,   304,   161,   936,     /*  234 */
		237,   148,    83,   468,     /*  235 */
		53,    32,    19,   104,     /*  236 */
		60,    35,    22,   117,     /*  237 */
		483,   272,   181,   936,     /*  238 */
		81,    44,    31,   156,     /*  239 */
		489,   256,   191,   936,     /*  240 */
		123,    62,    49,   234,     /*  241 */
		165,    80,    67,   312,     /*  242 */
		249,   116,   103,   468,     /*  243 */
		501,   224,   211,   936,     /*  244 */
		7,     3,     3,    13,     /*  245 */
		43,    15,    20,    78,     /*  246 */
		22,     6,    11,    39,     /*  247 */
		15,     3,     8,    26,     /*  248 */
		23,     3,    13,    39,     /*  249 */
		47,     3,    28,    78,     /*  250 */
		8,     0,     5,    13,     /*  251 */
		7,     0,     4,    11,     /*  252 */
		21,     0,    10,    31,     /*  253 */
		13,     0,     5,    18, /*  254 */
		13, 0, 5, 18 };/*  255 */

	int ditheringOstromoukhov(Mat& remap, Mat& dest, const bool isMeandering)
	{
		int sample_num = 0;

		const float threshold = 255.f * 0.5f;
		if (remap.depth() == CV_8U)
		{
			if (isMeandering)
			{
				cout << "do not support for 8U" << endl;
			}
			for (int y = 0; y < remap.rows - 1; y++)
			{
				uchar* s = remap.ptr<uchar>(y);
				uchar* s_next = remap.ptr<uchar>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e = 0.f;
				int x = 0;
				t_three_coefs coefs = var_coefs_tab[s[x]];
				{
					if (s[x] > threshold)
					{
						e = (float)(s[x] - 255);
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += (uchar)(coefs.i_r * e / coefs.i_sum);
					s_next[x] += (uchar)(coefs.i_d * e / coefs.i_sum);
				}

				for (int x = 1; x < remap.cols - 1; x++)
				{
					t_three_coefs coefs = var_coefs_tab[s[x]];
					if (s[x] > threshold)
					{
						e = (float)(s[x] - 255);
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					//e = s[x] - thread;
					s[x + 1] += (uchar)(coefs.i_r * e / coefs.i_sum);
					s_next[x - 1] += (uchar)(coefs.i_dl * e / coefs.i_sum);
					s_next[x] += (uchar)(coefs.i_d * e / coefs.i_sum);
				}

				x = remap.cols - 1;
				{
					t_three_coefs coefs = var_coefs_tab[s[x]];
					if (s[x] > threshold)
					{
						e = (float)(s[x] - 255);
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					//e = s[x] - thread;
					s_next[x - 1] += (uchar)(coefs.i_dl * e / coefs.i_sum);
					s_next[x] += (uchar)(coefs.i_d * e / coefs.i_sum);
				}
			}

			{
				uchar* s = remap.ptr<uchar>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;//error
				for (int x = 0; x < remap.cols - 1; x++)
				{
					t_three_coefs coefs = var_coefs_tab[s[x]];

					if (s[x] > threshold)
					{
						e = (float)(s[x] - 255);
						//e = 1.f - s[x];
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += (uchar)(coefs.i_r * e / coefs.i_sum);
					//	s[x + 1] = s[x + 1] + e * coeff5_16;
				}
				if (s[remap.cols - 1] > threshold)
				{
					d[remap.cols - 1] = 255;
					sample_num++;
				}
				else
				{
					d[remap.cols - 1] = 0;
				}
			}
		}
		else
		{
			remap *= 255.f;

			if (isMeandering)
			{
				for (int y = 0; y < remap.rows - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s_next = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e = 0.f;
					t_three_coefs coefs;
					if (y % 2 == 0)
					{
						int x = 0;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
						for (x = 1; x < remap.cols - 1; x++)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r * e / coefs.i_sum);
							s_next[x - 1] += (coefs.i_dl * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
						x = remap.cols - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s_next[x - 1] += (coefs.i_dl * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
					}
					else
					{
						int x = remap.cols - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
						for (x = remap.cols - 2; x > 0; x--)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r * e / coefs.i_sum);
							s_next[x + 1] += (coefs.i_dl * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
						x = 0;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s_next[x + 1] += (coefs.i_dl * e / coefs.i_sum);
							s_next[x] += (coefs.i_d * e / coefs.i_sum);
						}
					}
				}
				{
					float* s = remap.ptr<float>(remap.rows - 1);
					uchar* d = dest.ptr<uchar>(remap.rows - 1);
					float e;//error
					for (int x = 0; x < remap.cols - 1; x++)
					{
						t_three_coefs coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
						if (s[x] > threshold)
						{
							e = s[x] - 255.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += (coefs.i_r * e / coefs.i_sum);
					}

					if (s[remap.cols - 1] > threshold)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
			else if (FOURDIRECTION)
			{
				int center_width = remap.cols / 2;
				int center_height = remap.rows / 2;

				{
					for (int y = 0; y < center_height - 1; y++)
					{
						float* s = remap.ptr<float>(y);
						float* s1 = remap.ptr<float>(y + 1);
						uchar* d = dest.ptr<uchar>(y);
						float e;
						t_three_coefs coefs;

						int x = 0;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
						}
						for (x = 1; x < center_width - 1; x++)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x - 1] += (coefs.i_dl / coefs.i_sum);
						}
						x = center_width - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x - 1] += (coefs.i_dl / coefs.i_sum);
						}
					}
					{
						float* s = remap.ptr<float>(center_height - 1);
						uchar* d = dest.ptr<uchar>(center_height - 1);
						float e;
						t_three_coefs coefs;
						for (int x = 0; x < center_width - 1; x++)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
						}
						if (s[center_width - 1] > threshold)
						{
							d[center_width - 1] = 255;
							sample_num++;
						}
						else
						{
							d[center_width - 1] = 0;
						}
					}
				}
				{
					for (int y = 0; y < center_height - 1; y++)
					{
						float* s = remap.ptr<float>(y);
						float* s1 = remap.ptr<float>(y + 1);
						uchar* d = dest.ptr<uchar>(y);
						float e;
						t_three_coefs coefs;

						int x = remap.cols - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
						}
						for (x = remap.cols - 2; x > center_width; x--)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x + 1] += (coefs.i_dl / coefs.i_sum);
						}
						x = center_width;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x + 1] += (coefs.i_dl / coefs.i_sum);
						}
					}
					{
						float* s = remap.ptr<float>(center_height - 1);
						uchar* d = dest.ptr<uchar>(center_height - 1);
						float e;
						t_three_coefs coefs;
						for (int x = remap.cols - 1; x > center_width; x--)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
						}
						{
							if (s[center_width] > threshold)
							{
								d[center_width] = 255;
								sample_num++;
							}
							else
							{
								d[center_width] = 0;
							}
						}
					}
				}
				{
					for (int y = remap.rows - 1; y > center_height; y--)
					{
						float* s = remap.ptr<float>(y);
						float* s1 = remap.ptr<float>(y - 1);
						uchar* d = dest.ptr<uchar>(y);
						float e;
						t_three_coefs coefs;
						int x = 0;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
							s[x] += (coefs.i_d / coefs.i_sum);
						}
						for (x = 1; x < center_width - 1; x++)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x - 1] += (coefs.i_dl / coefs.i_sum);
						}
						x = center_width - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x - 1] += (coefs.i_dl / coefs.i_sum);
						}
					}
					{
						float* s = remap.ptr<float>(center_height);
						uchar* d = dest.ptr<uchar>(center_height);
						float e;
						t_three_coefs coefs;
						for (int x = 0; x < center_width - 1; x++)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += (coefs.i_r / coefs.i_sum);
						}
						{
							if (s[center_width - 1] > threshold)
							{
								d[center_width - 1] = 255;
								sample_num++;
							}
							else
							{
								d[center_width - 1] = 0;
							}
						}
					}
				}
				{
					for (int y = remap.rows - 1; y > center_height; y--)
					{
						float* s = remap.ptr<float>(y);
						float* s1 = remap.ptr<float>(y - 1);
						uchar* d = dest.ptr<uchar>(y);
						float e;
						t_three_coefs coefs;

						int x = remap.cols - 1;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
						}
						for (x = remap.cols - 2; x > center_width; x--)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x + 1] += (coefs.i_dl / coefs.i_sum);
						}
						x = center_width;
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x] += (coefs.i_d / coefs.i_sum);
							s1[x + 1] += (coefs.i_dl / coefs.i_sum);
						}
					}
					{
						float* s = remap.ptr<float>(center_height);
						uchar* d = dest.ptr<uchar>(center_height);
						float e;
						t_three_coefs coefs;

						for (int x = remap.cols - 1; x > center_width; x--)
						{
							coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
							if (s[x] > threshold)
							{
								e = s[x] - 255.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += (coefs.i_r / coefs.i_sum);
						}

						{
							if (s[center_width] > threshold)
							{
								d[center_width] = 255;
								sample_num++;
							}
							else
							{
								d[center_width] = 0;
							}
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < remap.rows - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s_next = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e = 0.f;
					t_three_coefs coefs;

					int x = 0;
					{
						coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
						if (s[x] > threshold)
						{
							e = s[x] - 255.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += (coefs.i_r * e / coefs.i_sum);
						s_next[x] += (coefs.i_d * e / coefs.i_sum);
					}
					for (x = 1; x < remap.cols - 1; x++)
					{
						coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
						if (s[x] > threshold)
						{
							e = s[x] - 255.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += (coefs.i_r * e / coefs.i_sum);
						s_next[x - 1] += (coefs.i_dl * e / coefs.i_sum);
						s_next[x] += (coefs.i_d * e / coefs.i_sum);
					}
					x = remap.cols - 1;
					{
						coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
						if (s[x] > threshold)
						{
							e = s[x] - 255.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s_next[x - 1] += (coefs.i_dl * e / coefs.i_sum);
						s_next[x] += (coefs.i_d * e / coefs.i_sum);
					}
				}
				{
					float* s = remap.ptr<float>(remap.rows - 1);
					uchar* d = dest.ptr<uchar>(remap.rows - 1);
					float e;//error
					for (int x = 0; x < remap.cols - 1; x++)
					{
						t_three_coefs coefs = var_coefs_tab[saturate_cast<uchar>(s[x])];
						if (s[x] > threshold)
						{
							e = s[x] - 255.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += (coefs.i_r * e / coefs.i_sum);
					}

					if (s[remap.cols - 1] > threshold)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}
		return sample_num;
	}

	int ditheringFloydSteinberg(Mat& remap, Mat& dest, int process_order)
	{
		//  x 7
		//3 5 1
		CV_Assert(remap.depth() == CV_32F);

		int sample_num = 0;

#ifdef FloydSteinbergTrackBar
		static int coeff1 = 1;
		static int coeff3 = 3;
		static int coeff5 = 5;
		static int coeff7 = 7;

		createTrackbar("1", "", &coeff1, 15);
		createTrackbar("3", "", &coeff3, 15);
		createTrackbar("5", "", &coeff5, 15);
		createTrackbar("7", "", &coeff7, 15);
#else 
		int coeff1 = 1;
		int coeff3 = 3;
		int coeff5 = 5;
		int coeff7 = 7;
#endif

		float total = 1.f / (coeff1 + coeff3 + coeff5 + coeff7);
		const float coeff7_16 = coeff7 * total;
		const float coeff5_16 = coeff5 * total;
		const float coeff3_16 = coeff3 * total;
		const float coeff1_16 = coeff1 * total;

		if (process_order == MEANDERING)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s_next = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;//error

				int x = 0;
				if (y % 2 == 1) //odd
				{
					x = 0;
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x + 1] = s[x + 1] + e * coeff7_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
					}
					for (x = 1; x < remap.cols - 1; x++)
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x + 1] = s[x + 1] + e * coeff7_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
					}
					//end of x
					x = remap.cols - 1;
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						//s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
						//s_next[x] = s_next[x] + e * coeff7_16;
						s_next[x - 1] = s_next[x - 1] + e * 0.5f;
						s_next[x] = s_next[x] + e * 0.5f;
					}
				}
				else //even
				{
					x = remap.cols - 1;
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x - 1] = s[x - 1] + e * coeff7_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff1_16;
					}
					for (x = remap.cols - 2; x > 0; x--)
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x - 1] = s[x - 1] + e * coeff7_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff1_16;
					}

					//start of x
					x = 0;
					{
						if (s[x] >= 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						///s_next[x + 1] = s_next[x + 1] + e * coeff3_16;
						//s_next[x] = s_next[x] + e * coeff7_16;
						s_next[x + 1] = s_next[x + 1] + e * 0.5f;
						s_next[x] = s_next[x] + e * 0.5f;
					}
				}
			}
			// bottom y loop
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;//error
				for (int x = 0; x < remap.cols - 1; x++)
				{
					if (s[x] >= 0.5f)
					{
						e = s[x] - 1.f;
						//e = 1.f - s[x];
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + e * coeff7_16;
				}

				//x=remap.cols - 1
				if (s[remap.cols - 1] >= 0.5f)
				{
					d[remap.cols - 1] = 255;
					sample_num++;
				}
				else
				{
					d[remap.cols - 1] = 0;
				}
			}
		}
		else if (process_order == FORWARD)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s_next = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;//error

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					s[x + 1] = s[x + 1] + e * coeff7_16;
					s_next[x] = s_next[x] + e * coeff5_16;
					s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
				}

				for (x = 1; x < remap.cols - 1; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					s[x + 1] = s[x + 1] + e * coeff7_16;
					s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
					s_next[x] = s_next[x] + e * coeff5_16;
					s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
				}

				x = remap.cols - 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
					s_next[x] = s_next[x] + e * coeff5_16;
				}
			}
			// bottom y loop
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;//error
				for (int x = 0; x < remap.cols - 1; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						//e = 1.f - s[x];
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + e * coeff7_16;
				}

				//x=remap.cols - 1
				if (s[remap.cols - 1] > 0.5f)
				{
					d[remap.cols - 1] = 255;
					sample_num++;
				}
				else
				{
					d[remap.cols - 1] = 0;
				}
			}
		}
		else if (process_order == IN2OUT)
		{
			int r = remap.cols / 2;
			int y = 0;
			{
				int x = 0;
				float e;
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					e = remap.at<float>(y + r, x + r) - 1.f;
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y + r, x + r);
					dest.at<uchar>(y + r, x + r) = 0;
				}
				remap.at<float>(y + r, x + r - 1) += coeff7_16 * e;
				remap.at<float>(y + r + 1, x + r - 1) += coeff1_16 * e;
				remap.at<float>(y + r + 1, x + r) += coeff5_16 * e;
				remap.at<float>(y + r + 1, x + r + 1) += coeff3_16 * e;
			}

			for (int lr = 1; lr <= r - 1; lr++)
			{
				int x = -lr;
				y = 0;
				for (; y >= -lr; y--)
				{
					float* s = remap.ptr<float>(y + r);
					float* s0 = remap.ptr<float>(y + r - 1);
					float* s1 = remap.ptr<float>(y + r + 1);
					uchar* d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == -lr)
					{
						s[x + r + 1] += coeff7_16 * e;
						s0[x + r + 1] += coeff1_16 * e;
						s0[x + r] += coeff5_16 * e;
						s0[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
				}
				for (x++, y++; x <= lr; x++)
				{
					float* s = remap.ptr<float>(y + r);
					float* s0 = remap.ptr<float>(y + r - 1);
					uchar* d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (x == lr)
					{
						float* s1 = remap.ptr<float>(y + r + 1);
						s1[x + r] += coeff7_16 * e;
						s0[x + r + 1] += coeff3_16 * e;
						s[x + r + 1] += coeff5_16 * e;
						s1[x + r + 1] += coeff1_16 * e;
					}
					else
					{
						s[x + r + 1] += coeff7_16 * e;
						s0[x + r - 1] += coeff3_16 * e;
						s0[x + r] += coeff5_16 * e;
						s0[x + r + 1] += coeff1_16 * e;
					}
				}
				for (x--, y++; y <= lr; y++)
				{
					float* s = remap.ptr<float>(y + r);
					float* s0 = remap.ptr<float>(y + r - 1);
					float* s1 = remap.ptr<float>(y + r + 1);
					uchar* d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == lr)
					{
						s[x + r - 1] += coeff7_16 * e;
						s1[x + r - 1] += coeff1_16 * e;
						s1[x + r] += coeff5_16 * e;
						s1[x + r + 1] += coeff3_16 * e;
					}
					else
					{
						s1[x + r] += coeff7_16 * e;
						s0[x + r + 1] += coeff3_16 * e;
						s[x + r + 1] += coeff5_16 * e;
						s1[x + r + 1] += coeff1_16 * e;
					}
				}
				for (x--, y--; x >= -lr; x--)
				{
					float* s = remap.ptr<float>(y + r);
					float* s1 = remap.ptr<float>(y + r + 1);
					uchar* d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (x == -lr)
					{
						float* s0 = remap.ptr<float>(y + r - 1);
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s[x + r - 1] += coeff7_16 * e;
						s1[x + r - 1] += coeff1_16 * e;
						s1[x + r] += coeff5_16 * e;
						s1[x + r + 1] += coeff3_16 * e;
					}
				}
				for (x++, y--; y > 0; y--)
				{
					float* s = remap.ptr<float>(y + r);
					float* s0 = remap.ptr<float>(y + r - 1);
					float* s1 = remap.ptr<float>(y + r + 1);
					uchar* d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == 1)
					{
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
				}

			}
			int lr = r;
			int x = -lr;
			for (; y > -lr; y--)
			{
				float* s = remap.ptr<float>(y + r);
				float* s0 = remap.ptr<float>(y + r - 1);
				float* s1 = remap.ptr<float>(y + r + 1);
				uchar* d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s0[x + r] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (x++; x < lr; x++)
			{
				float* s = remap.ptr<float>(y + r);
				float* s0 = remap.ptr<float>(y + r - 1);
				uchar* d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s[x + r + 1] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (y++; y < lr; y++)
			{
				float* s = remap.ptr<float>(y + r);
				float* s0 = remap.ptr<float>(y + r - 1);
				float* s1 = remap.ptr<float>(y + r + 1);
				uchar* d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s1[x + r] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (x--; x > -lr; x--)
			{
				float* s = remap.ptr<float>(y + r);
				float* s1 = remap.ptr<float>(y + r + 1);
				uchar* d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s[x + r - 1] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (y--; y > 0; y--)
			{
				float* s = remap.ptr<float>(y + r);
				float* s0 = remap.ptr<float>(y + r - 1);
				float* s1 = remap.ptr<float>(y + r + 1);
				uchar* d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				if (y != 1)
					s0[x + r] += coeff7_16 * e;
			}

		}
		else if (process_order == OUT2IN)
		{
			int r = remap.cols / 2;
			int y = 0;
			int x = 0;
			int xend = 0;
			int yend = 0;
			int index = 0;

			for (int lr = r; lr >= 2; lr--)
			{
				x = index;
				y = index;
				xend = remap.cols - index;
				yend = remap.rows - index;

				for (; x < xend; x++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					if (x == xend - 1)
					{
						s1[x] += coeff7_16 * e;
						s1[x - 1] += coeff1_16 * e;
					}
					else if (x == index)
					{
						s[x + 1] += coeff7_16 * e;
						s1[x] += coeff5_16 * e;
						s1[x + 1] += coeff1_16 * e;
					}
					else
					{
						s[x + 1] += coeff7_16 * e;
						s1[x - 1] += coeff3_16 * e;
						s1[x] += coeff5_16 * e;
						s1[x + 1] += coeff1_16 * e;
					}
				}
				for (x--, y++; y < yend; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s0 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					if (y == yend - 1)
					{
						s[x - 1] += coeff7_16 * e;
						s0[x - 1] += coeff1_16 * e;
					}
					else
					{
						float* s1 = remap.ptr<float>(y + 1);
						s1[x] += coeff7_16 * e;
						s1[x - 1] += coeff1_16 * e;
						s[x - 1] += coeff5_16 * e;
						s0[x - 1] += coeff3_16 * e;
					}
				}

				for (x--, y--; x >= index; x--)
				{
					float* s = remap.ptr<float>(y);
					float* s0 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					if (x == index)
					{
						s0[x] += coeff7_16 * e;
						s0[x + 1] += coeff1_16 * e;
					}
					else
					{
						s[x - 1] += coeff7_16 * e;
						s0[x - 1] += coeff1_16 * e;
						s0[x] += coeff5_16 * e;
						s0[x + 1] += coeff3_16 * e;
					}
				}
				for (x++, y--; y > index; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					if (y == index + 1)
					{
						s[x + 1] += coeff7_16 * e;
						s1[x + 1] += coeff1_16 * e;
					}
					else
					{
						float* s0 = remap.ptr<float>(y - 1);
						s0[x] += coeff7_16 * e;
						s0[x + 1] += coeff1_16 * e;
						s[x + 1] += coeff5_16 * e;
						s1[x + 1] += coeff3_16 * e;
					}
				}
				index++;
			}
			{
				x = r - 1;
				y = r - 1;
				float e;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff7_16 * e;
				remap.at<float>(y + 1, x) += coeff5_16 * e;
				remap.at<float>(y + 1, x + 1) += coeff1_16 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff7_16 * e;
				remap.at<float>(y + 1, x) += coeff5_16 * e;
				remap.at<float>(y + 1, x + 1) += coeff1_16 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff7_16 * e;
				remap.at<float>(y + 1, x - 1) += coeff1_16 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff7_16 * e;
				remap.at<float>(y + 1, x - 1) += coeff1_16 * e;
				remap.at<float>(y, x - 1) += coeff5_16 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x - 1) += coeff7_16 * e;
				remap.at<float>(y - 1, x - 1) += coeff1_16 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x - 1) += coeff7_16 * e;
				remap.at<float>(y - 1, x - 1) += coeff1_16 * e;
				remap.at<float>(y - 1, x) += coeff5_16 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 1, x) += coeff7_16 * e;
				remap.at<float>(y - 1, x + 1) += coeff1_16 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_16 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y, x) = 0;
				}
			}
		}
		else if (process_order == FOURDIRECTION)
		{
#if 0
			int center_width = remap.cols / 2;
			int center_height = remap.rows / 2;

			//lt
			{
				for (int y = 0; y < center_height - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
					}
					for (x = 1; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x - 1] += e * coeff3_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
					}
					x = center_width - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 1] += e * coeff3_16;
						s1[x] += e * coeff5_16;
					}
				}

				int y = center_height - 1;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					for (int x = 0; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}

					int x = center_width - 1;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
			}

			//rt
			{
				for (int y = 0; y < center_height - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = remap.cols - 2; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
					x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
				}

				int y = center_height - 1;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					for (int x = remap.cols - 1; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}

					int x = center_width;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
			}

			//lb
			{
				for (int y = remap.rows - 1; y > center_height; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = 1; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
					x = center_width - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
				}

				int y = center_height;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = remap.ptr<uchar>(y);
					float e;
					for (int x = 0; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}

					int x = center_width - 1;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
			}

			//rb
			{
				for (int y = remap.rows - 1; y > center_height; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = remap.cols - 2; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
					x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
				}

				int y = center_height;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					for (int x = remap.cols - 1; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}

					int x = center_width;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						s[x] = 0;
					}
				}
			}
#else
			int center_width = remap.cols / 2;
			int center_height = remap.rows / 2;

			//lt
			{
				for (int y = 0; y < center_height; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
					}
					for (x = 1; x < center_width; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x - 1] += e * coeff3_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
					}
					/*
					x = center_width - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 1] += e * coeff3_16;
						s1[x] += e * coeff5_16;
					}
					*/
				}

				/*
				int y = center_height - 1;
				{
					float *s = remap.ptr<float>(y);
					uchar *d = dest.ptr<uchar>(y);
					float e;
					for (int x = 0; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}

					int x = center_width - 1;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
				*/
			}

			//rt
			{
				for (int y = 0; y < center_height; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = remap.cols - 2; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
					x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
				}

				/*
				int y = center_height - 1;
				{
					float *s = remap.ptr<float>(y);
					uchar *d = dest.ptr<uchar>(y);
					float e;
					for (int x = remap.cols - 1; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}

					int x = center_width;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
				*/
			}

			//lb
			{
				for (int y = remap.rows - 1; y > center_height; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = 1; x < center_width; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
					/*
					x = center_width - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
					*/
				}

				int y = center_height;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = remap.ptr<uchar>(y);
					float e;
					for (int x = 0; x < center_width - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}

					int x = center_width - 1;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
			}

			//rb
			{
				for (int y = remap.rows - 1; y > center_height; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = remap.cols - 2; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
					x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}
				}

				int y = center_height;
				{
					float* s = remap.ptr<float>(y);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					for (int x = remap.cols - 1; x > center_width; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}

					int x = center_width;
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						s[x] = 0;
					}
				}
			}
#endif
		}

		else if (process_order == FOURDIRECTIONIN2OUT)
		{
			int center_width = remap.cols / 2;
			int center_height = remap.rows / 2;

			{
				for (int y = center_height; y > 0; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff1_16;
					}
					for (x = center_width - 1; x > 0; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff1_16;
						s1[x + 1] += e * coeff3_16;
					}
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff3_16;
					}

				}
				{
					float* s = remap.ptr<float>(0);
					uchar* d = dest.ptr<uchar>(0);
					float e;
					for (int x = center_width; x > 0; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}
					if (s[0] > 0.5f)
					{
						d[0] = 255;
						sample_num++;
					}
					else
					{
						d[0] = 0;
					}
				}
			}
			{
				for (int y = center_height; y > 0; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y - 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = center_width + 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = center_width + 2; x < remap.cols - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
				}
				{
					float* s = remap.ptr<float>(0);
					uchar* d = dest.ptr<uchar>(0);
					float e;
					for (int x = center_width + 1; x < remap.cols - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
			{
				for (int y = center_height + 1; y < remap.rows - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = center_width;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x - 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
					for (x = center_width - 1; x > 0; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
						s1[x + 1] += e * coeff3_16;
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff1_16;
					}
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 1] += e * coeff1_16;
						s1[x] += e * coeff5_16;
					}
				}
				{
					float* s = remap.ptr<float>(remap.rows - 1);
					uchar* d = dest.ptr<uchar>(remap.rows - 1);
					float e;
					for (int x = center_width; x > 0; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += e * coeff7_16;
					}
					if (s[0] > 0.5f)
					{
						d[0] = 255;
						sample_num++;
					}
					else
					{
						d[0] = 0;
					}
				}
			}
			{
				for (int y = center_height + 1; y < remap.rows - 1; y++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					int x = center_width + 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
					}
					for (x = center_width + 2; x < remap.cols - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
						s1[x] += e * coeff5_16;
						s1[x + 1] += e * coeff1_16;
						s1[x - 1] += e * coeff3_16;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] += e * coeff5_16;
						s1[x - 1] += e * coeff3_16;
					}
				}
				{
					float* s = remap.ptr<float>(remap.rows - 1);
					uchar* d = dest.ptr<uchar>(remap.rows - 1);
					float e;
					for (int x = center_width + 1; x < remap.cols - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += e * coeff7_16;
					}
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}

				}
			}
		}
		return sample_num;
	}

	int ditheringSteavenson(Mat& remap, Mat& dest, const bool isMeandering)
	{
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff32_200 = 32.f / 200;
		const float coeff12_200 = 12.f / 200;
		const float coeff26_200 = 26.f / 200;
		const float coeff30_200 = 30.f / 200;
		const float coeff16_200 = 16.f / 200;
		const float coeff5_200 = 5.f / 200;

		if (isMeandering)
		{
			for (int y = 0; y < remap.rows - 3; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				float* s3 = remap.ptr<float>(y + 3);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				if (y % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
					}
					for (x = 3; x < remap.cols - 3; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
					}
					x = remap.cols - 3;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
						s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff16_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					}
					for (x = remap.cols - 4; x > 2; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
						s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					}
					x = 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
						s3[x - 1] = s2[x - 1] + coeff12_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s3[x + 3] = s3[x + 3] + coeff5_200 * e;
						s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 3);
				float* s1 = remap.ptr<float>(remap.rows - 2);
				float* s2 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 3);
				float e;
				if ((remap.rows - 3) % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					}
					for (x = 3; x < remap.cols - 3; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					}
					x = remap.cols - 3;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					}
					for (x = remap.cols - 4; x > 2; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					}
					x = 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
						s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s2[x + 2] = s2[x + 2] + coeff12_200 * e;
						s2[x] = s2[x] + coeff26_200 * e;
					}
				}
			}

			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				if ((remap.rows - 2) % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					}
					for (x = 3; x < remap.cols - 3; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
						s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					}
					x = remap.cols - 3;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
						s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 3] = s1[x - 3] + coeff12_200 * e;
						s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
					}
					for (x = remap.cols - 4; x > 2; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
						s1[x - 3] = s1[x - 3] + coeff16_200 * e;
					}
					x = 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
						s1[x - 1] = s1[x - 1] + coeff30_200 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 3] = s1[x + 3] + coeff12_200 * e;
						s1[x + 1] = s1[x + 1] + coeff26_200 * e;
					}
				}
			}

			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				if ((remap.rows - 1) % 2 == 1)
				{
					int x = 0;
					for (; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 2] = s[x + 2] + coeff32_200 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							d[x] = 255;
							sample_num++;
						}
						else
						{
							d[x] = 0;
						}
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							d[x] = 255;
							sample_num++;
						}
						else
						{
							d[x] = 0;
						}
					}
				}
				else
				{
					int x = remap.cols - 1;
					for (; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 2] = s[x - 2] + coeff32_200 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							d[x] = 255;
							sample_num++;
						}
						else
						{
							d[x] = 0;
						}
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							d[x] = 255;
							sample_num++;
						}
						else
						{
							d[x] = 0;
						}
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				float* s3 = remap.ptr<float>(y + 3);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					s3[x + 3] = s3[x + 3] + coeff5_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					s3[x + 3] = s3[x + 3] + coeff5_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					s3[x + 3] = s3[x + 3] + coeff5_200 * e;
				}
				for (x = 3; x < remap.cols - 3; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
					s3[x + 3] = s3[x + 3] + coeff5_200 * e;
				}
				x = remap.cols - 3;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
					s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
					s3[x + 1] = s3[x + 1] + coeff12_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s3[x - 3] = s3[x - 3] + coeff5_200 * e;
					s3[x - 1] = s3[x - 1] + coeff12_200 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 3);
				float* s1 = remap.ptr<float>(remap.rows - 2);
				float* s2 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 3);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
				}
				for (x = 3; x < remap.cols - 3; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
				}
				x = remap.cols - 3;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
					s2[x + 2] = s2[x + 2] + coeff12_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s2[x - 2] = s2[x - 2] + coeff12_200 * e;
					s2[x] = s2[x] + coeff26_200 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
				}
				for (x = 3; x < remap.cols - 3; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
					s1[x + 3] = s1[x + 3] + coeff16_200 * e;
				}
				x = remap.cols - 3;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
					s1[x + 1] = s1[x + 1] + coeff30_200 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 3] = s1[x - 3] + coeff12_200 * e;
					s1[x - 1] = s1[x - 1] + coeff26_200 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = 0;
				for (; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 2] = s[x + 2] + coeff32_200 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						d[x] = 255;
						sample_num++;
					}
					else
					{
						d[x] = 0;
					}
				}
			}
		}
		return sample_num;
	}

	int ditheringBurkes(Mat& remap, Mat& dest, const bool isMeandering)
	{
		//    x 8
		//2 4 8 4
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff8_32 = 8.f / 32;
		const float coeff4_32 = 4.f / 32;
		const float coeff2_32 = 2.f / 32;

		if (isMeandering)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;
				if (y % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff8_32 * e;
						s[x + 2] += coeff4_32 * e;
						s1[x] += coeff8_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_32 * e;
						s[x + 2] = s[x + 2] + coeff4_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_32 * e;
						s[x + 2] = s[x + 2] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
					}

				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_32 * e;
						s[x - 2] = s[x - 2] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_32 * e;
						s[x - 2] = s[x - 2] + coeff4_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_32 * e;
						s[x - 2] = s[x - 2] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff8_32 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = 0;
				for (; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s[x + 2] = s[x + 2] + coeff4_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
				}
				{
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s[x + 2] = s[x + 2] + coeff4_32 * e;
					s1[x] = s1[x] + coeff8_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s[x + 2] = s[x + 2] + coeff4_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff8_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s[x + 2] = s[x + 2] + coeff4_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff8_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff8_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff8_32 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = 0;
				for (; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
					s[x + 2] = s[x + 2] + coeff4_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_32 * e;
				}
				{
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}
		return sample_num;
	}

	int ditheringStucki(Mat& remap, Mat& dest, const bool isMeandering)
	{
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff8_42 = 8.f / 42;
		const float coeff4_42 = 4.f / 42;
		const float coeff2_42 = 2.f / 42;
		const float coeff1_42 = 1.f / 42;

		if (isMeandering)
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				if (y % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff8_42 * e;
						s[x + 2] += coeff4_42 * e;
						s1[x] += coeff8_42 * e;
						s1[x + 1] += coeff4_42 * e;
						s1[x + 2] += coeff2_42 * e;
						s2[x] += coeff4_42 * e;
						s2[x + 1] += coeff2_42 * e;
						s2[x + 2] += coeff1_42 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff8_42 * e;
						s[x + 2] += coeff4_42 * e;
						s1[x - 1] += coeff4_42 * e;
						s1[x] += coeff8_42 * e;
						s1[x + 1] += coeff4_42 * e;
						s1[x + 2] += coeff2_42 * e;
						s2[x - 1] += coeff2_42 * e;
						s2[x] += coeff4_42 * e;
						s2[x + 1] += coeff2_42 * e;
						s2[x + 2] += coeff1_42 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff8_42 * e;
						s[x + 2] += coeff4_42 * e;
						s1[x - 2] += coeff2_42 * e;
						s1[x - 1] += coeff4_42 * e;
						s1[x] += coeff8_42 * e;
						s1[x + 1] += coeff4_42 * e;
						s1[x + 2] += coeff2_42 * e;
						s2[x - 2] += coeff1_42 * e;
						s2[x - 1] += coeff2_42 * e;
						s2[x] += coeff4_42 * e;
						s2[x + 1] += coeff2_42 * e;
						s2[x + 2] += coeff1_42 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s2[x - 2] = s2[x - 2] + coeff1_42 * e;
						s2[x - 1] = s2[x - 1] + coeff2_42 * e;
						s2[x] = s2[x] + coeff4_42 * e;
						s2[x + 1] = s2[x + 1] + coeff2_42 * e;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s2[x - 2] = s2[x - 2] + coeff1_42 * e;
						s2[x - 1] = s2[x - 1] + coeff2_42 * e;
						s2[x] = s2[x] + coeff4_42 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += coeff8_42 * e;
						s[x - 2] += coeff4_42 * e;
						s1[x] += coeff8_42 * e;
						s1[x - 1] += coeff4_42 * e;
						s1[x - 2] += coeff2_42 * e;
						s2[x] += coeff4_42 * e;
						s2[x - 1] += coeff2_42 * e;
						s2[x - 2] += coeff1_42 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s[x - 2] = s[x - 2] + coeff4_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s2[x + 1] = s2[x + 1] + coeff2_42 * e;
						s2[x] = s2[x] + coeff4_42 * e;
						s2[x - 1] = s2[x - 1] + coeff2_42 * e;
						s2[x - 2] = s2[x - 2] + coeff1_42 * e;
					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += coeff8_42 * e;
						s[x - 2] += coeff4_42 * e;
						s1[x + 2] += coeff2_42 * e;
						s1[x + 1] += coeff4_42 * e;
						s1[x] += coeff8_42 * e;
						s1[x - 1] += coeff4_42 * e;
						s1[x - 2] += coeff2_42 * e;
						s2[x + 2] += coeff1_42 * e;
						s2[x + 1] += coeff2_42 * e;
						s2[x] += coeff4_42 * e;
						s2[x - 1] += coeff2_42 * e;
						s2[x - 2] += coeff1_42 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s2[x + 2] = s2[x + 2] + coeff1_42 * e;
						s2[x + 1] = s2[x + 1] + coeff2_42 * e;
						s2[x] = s2[x] + coeff4_42 * e;
						s2[x - 1] = s2[x - 1] + coeff2_42 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s2[x + 2] = s2[x + 2] + coeff1_42 * e;
						s2[x + 1] = s2[x + 1] + coeff2_42 * e;
						s2[x] = s2[x] + coeff4_42 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				if ((remap.rows - 2) % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s[x + 2] = s[x + 2] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s[x + 2] = s[x + 2] + coeff4_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s[x + 2] = s[x + 2] + coeff4_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s[x - 2] = s[x - 2] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s[x - 2] = s[x - 2] + coeff4_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s[x - 2] = s[x - 2] + coeff4_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
						s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
						s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 2] = s1[x + 2] + coeff2_42 * e;
						s1[x + 1] = s1[x + 1] + coeff4_42 * e;
						s1[x] = s1[x] + coeff8_42 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.cols - 1);
				uchar* d = dest.ptr<uchar>(remap.cols - 1);
				float e;
				if ((remap.cols - 1) % 2 == 1)
				{
					int x = 0;
					for (; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
						s[x + 2] = s[x + 2] + coeff4_42 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff8_42 * e;
					}
					{
						if (s[remap.cols - 1])
						{
							d[remap.cols - 1] = 255;
							sample_num++;
						}
						else
						{
							d[remap.cols - 1] = 0;
						}
					}
				}
				else
				{
					int x = remap.cols - 1;
					for (; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
						s[x - 2] = s[x - 2] + coeff4_42 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff8_42 * e;
					}
					{
						if (s[remap.cols - 1] > 0.5f)
						{
							d[remap.cols - 1] = 255;
							sample_num++;
						}
						else
						{
							d[remap.cols - 1] = 0;
						}
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					s2[x] = s2[x] + coeff4_42 * e;
					s2[x + 1] = s2[x + 1] + coeff2_42 * e;
					s2[x + 2] = s2[x + 2] + coeff1_42 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					s2[x - 1] = s2[x - 1] + coeff2_42 * e;
					s2[x] = s2[x] + coeff4_42 * e;
					s2[x + 1] = s2[x + 1] + coeff2_42 * e;
					s2[x + 2] = s2[x + 2] + coeff1_42 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
					s2[x - 2] = s2[x - 2] + coeff1_42 * e;
					s2[x - 1] = s2[x - 1] + coeff2_42 * e;
					s2[x] = s2[x] + coeff4_42 * e;
					s2[x + 1] = s2[x + 1] + coeff2_42 * e;
					s2[x + 2] = s2[x + 2] + coeff1_42 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s2[x - 2] = s2[x - 2] + coeff1_42 * e;
					s2[x - 1] = s2[x - 1] + coeff2_42 * e;
					s2[x] = s2[x] + coeff4_42 * e;
					s2[x + 1] = s2[x + 1] + coeff2_42 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s2[x - 2] = s2[x - 2] + coeff1_42 * e;
					s2[x - 1] = s2[x - 1] + coeff2_42 * e;
					s2[x] = s2[x] + coeff4_42 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
					s1[x + 2] = s1[x + 2] + coeff2_42 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
					s1[x + 1] = s1[x + 1] + coeff4_42 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff2_42 * e;
					s1[x - 1] = s1[x - 1] + coeff4_42 * e;
					s1[x] = s1[x] + coeff8_42 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.cols - 1);
				uchar* d = dest.ptr<uchar>(remap.cols - 1);
				float e;
				int x = 0;
				for (; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
					s[x + 2] = s[x + 2] + coeff4_42 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff8_42 * e;
				}
				{
					if (s[remap.cols - 1])
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}
		return sample_num;
	}

	int ditheringJarvis(Mat& remap, Mat& dest, const bool isMeandering)
	{
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff7_48 = 7.f / 48;
		const float coeff5_48 = 5.f / 48;
		const float coeff3_48 = 3.f / 48;
		const float coeff4_48 = 4.f / 48;
		const float coeff1_48 = 1.f / 48;

		if (isMeandering)
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr <float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				if (y % 2 == 1)
				{
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff7_48 * e;
						s[x + 2] = s[x + 2] + coeff5_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff5_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x] = s2[x] + coeff5_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff7_48 * e;
						s[x + 2] = s[x + 2] + coeff5_48 * e;
						s1[x - 1] = s1[x - 1] + coeff4_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff5_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x] = s2[x] + coeff5_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff7_48 * e;
						s[x + 2] = s[x + 2] + coeff5_48 * e;
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x - 1] = s1[x - 1] + coeff4_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff5_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x] = s2[x] + coeff5_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff7_48 * e;
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x - 1] = s1[x - 1] + coeff4_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff5_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x] = s2[x] + coeff5_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x - 1] = s1[x - 1] + coeff4_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x] = s2[x] + coeff5_48 * e;
					}
				}
				else
				{
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff7_48 * e;
						s[x - 2] = s[x - 2] + coeff5_48 * e;
						s1[x - 1] = s1[x - 1] + coeff5_48 * e;
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x] = s2[x] + coeff3_48 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff7_48 * e;
						s[x - 2] = s[x - 2] + coeff5_48 * e;
						s1[x - 1] = s1[x - 1] + coeff5_48 * e;
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff4_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x] = s2[x] + coeff3_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;

					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff7_48 * e;
						s[x - 2] = s[x - 2] + coeff5_48 * e;
						s1[x - 1] = s1[x - 1] + coeff5_48 * e;
						s1[x - 2] = s1[x - 2] + coeff3_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff4_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x - 2] = s2[x - 2] + coeff1_48 * e;
						s2[x] = s2[x] + coeff3_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff7_48 * e;
						s1[x - 1] = s1[x - 1] + coeff5_48 * e;
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff4_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x - 1] = s2[x - 1] + coeff3_48 * e;
						s2[x] = s2[x] + coeff3_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;

					}
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x] = s1[x] + coeff7_48 * e;
						s1[x + 1] = s1[x + 1] + coeff4_48 * e;
						s1[x + 2] = s1[x + 2] + coeff3_48 * e;
						s2[x] = s2[x] + coeff3_48 * e;
						s2[x + 1] = s2[x + 1] + coeff3_48 * e;
						s2[x + 2] = s2[x + 2] + coeff1_48 * e;
					}
				}
			}
			if ((remap.rows - 2) % 2 == 1)
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += coeff7_48 * e;
					s[x + 2] += coeff5_48 * e;
					s1[x] += coeff7_48 * e;
					s1[x + 1] += coeff5_48 * e;
					s1[x + 2] += coeff3_48 * e;
				}
				x = 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += coeff7_48 * e;
					s[x + 2] += coeff5_48 * e;
					s1[x - 1] += coeff4_48 * e;
					s1[x] += coeff7_48 * e;
					s1[x + 1] += coeff5_48 * e;
					s1[x + 2] += coeff3_48 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += coeff7_48 * e;
					s1[x - 2] += coeff3_48 * e;
					s1[x - 1] += coeff4_48 * e;
					s1[x] += coeff7_48 * e;
					s1[x + 1] += coeff5_48 * e;
				}
				x = remap.cols - 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] += coeff3_48 * e;
					s1[x - 1] += coeff4_48 * e;
					s1[x] += coeff7_48 * e;
				}
			}
			else
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				int x = remap.cols - 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] = s[x - 1] + coeff7_48 * e;
					s[x - 2] = s[x - 2] + coeff5_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x - 1] = s1[x - 1] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
				}
				x--;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] = s[x - 1] + coeff7_48 * e;
					s[x - 2] = s[x - 2] + coeff5_48 * e;
					s1[x + 1] = s1[x + 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x - 1] = s1[x - 1] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
				}
				for (x = remap.cols - 3; x > 1; x--)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] = s[x - 1] + coeff7_48 * e;
					s[x - 2] = s[x - 2] + coeff5_48 * e;
					s1[x + 1] = s1[x + 1] + coeff4_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x - 1] = s1[x - 1] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
				}
				x = 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] = s[x - 1] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff4_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x - 1] = s1[x - 1] + coeff5_48 * e;
				}
				x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x + 1] = s1[x + 1] + coeff4_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
				}
			}

			if ((remap.rows - 1) % 2 == 1)
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = 0;
				for (x = 0; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += coeff7_48 * e;
					s[x + 2] += coeff5_48 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] += coeff7_48 * e;
				}
				{
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
			else
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = remap.cols - 1;
				for (x = remap.cols - 1; x > 1; x--)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] += coeff7_48 * e;
					s[x - 2] += coeff5_48 * e;
				}
				x = 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x - 1] += coeff7_48 * e;
				}
				{
					if (s[0] > 0.5f)
					{
						d[0] = 255;
						sample_num++;
					}
					else
					{
						d[0] = 0;
					}
				}
			}
		}
		else
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s2[x] = s2[x] + coeff5_48 * e;
					s2[x + 1] = s2[x + 1] + coeff3_48 * e;
					s2[x + 2] = s2[x + 2] + coeff1_48 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s2[x - 1] = s2[x - 1] + coeff3_48 * e;
					s2[x] = s2[x] + coeff5_48 * e;
					s2[x + 1] = s2[x + 1] + coeff3_48 * e;
					s2[x + 2] = s2[x + 2] + coeff1_48 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
					s2[x - 2] = s2[x - 2] + coeff1_48 * e;
					s2[x - 1] = s2[x - 1] + coeff3_48 * e;
					s2[x] = s2[x] + coeff5_48 * e;
					s2[x + 1] = s2[x + 1] + coeff3_48 * e;
					s2[x + 2] = s2[x + 2] + coeff1_48 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s2[x - 2] = s2[x - 2] + coeff1_48 * e;
					s2[x - 1] = s2[x - 1] + coeff3_48 * e;
					s2[x] = s2[x] + coeff5_48 * e;
					s2[x + 1] = s2[x + 1] + coeff3_48 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s2[x - 2] = s2[x - 2] + coeff1_48 * e;
					s2[x - 1] = s2[x - 1] + coeff3_48 * e;
					s2[x] = s2[x] + coeff5_48 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
					s1[x + 2] = s1[x + 2] + coeff3_48 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
					s1[x + 1] = s1[x + 1] + coeff5_48 * e;
				}
				x = remap.cols - 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff3_48 * e;
					s1[x - 1] = s1[x - 1] + coeff4_48 * e;
					s1[x] = s1[x] + coeff7_48 * e;
				}

			}
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				int x = 0;
				for (x = 0; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
					s[x + 2] = s[x + 2] + coeff5_48 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff7_48 * e;
				}
				x = remap.cols - 1;
				{
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}

		return sample_num;
	}

	int ditheringSierra3line(Mat& remap, Mat& dest, int process_order)
	{
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff5_32 = 5.f / 32;
		const float coeff3_32 = 3.f / 32;
		const float coeff2_32 = 2.f / 32;
		const float coeff4_32 = 4.f / 32;

		if (process_order == MEANDERING)
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				if (y % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] += coeff5_32 * e;
						s1[x - 2] += coeff2_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] += coeff2_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] += coeff5_32 * e;
						s[x - 2] += coeff3_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x - 2] += coeff2_32 * e;
						s2[x + 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x - 1] += coeff2_32 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					}
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;
				if ((remap.rows - 2) % 2 == 1)
				{
					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
					}
				}
				else
				{
					int x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					for (x = remap.cols - 3; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					}
					x--;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
					}
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 1);
				float e;
				if ((remap.rows - 1) % 2 == 1)
				{
					int x = 0;
					for (; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
					}
					{
						if (s[remap.cols - 1] > 0.5f)
						{
							d[remap.cols - 1] = 255;
							sample_num++;
						}
						else
						{
							d[remap.cols - 1] = 0;
						}
					}
				}
				else
				{
					int x = remap.cols - 1;
					for (; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
						s[x - 2] = s[x - 2] + coeff3_32 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff5_32 * e;
					}
					{
						if (s[remap.cols - 1] > 0.5f)
						{
							d[remap.cols - 1] = 255;
							sample_num++;
						}
						else
						{
							d[remap.cols - 1] = 0;
						}
					}
				}
			}
		}
		else if (FORWARD)
		{
			for (int y = 0; y < remap.rows - 2; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s1 = remap.ptr<float>(y + 1);
				float* s2 = remap.ptr<float>(y + 2);
				uchar* d = dest.ptr<uchar>(y);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					s2[x] = s2[x] + coeff3_32 * e;
					s2[x + 1] = s2[x + 1] + coeff2_32 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					s2[x] = s2[x] + coeff3_32 * e;
					s2[x + 1] = s2[x + 1] + coeff2_32 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					s2[x] = s2[x] + coeff3_32 * e;
					s2[x + 1] = s2[x + 1] + coeff2_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					s2[x] = s2[x] + coeff3_32 * e;
					s2[x + 1] = s2[x + 1] + coeff2_32 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					s2[x] = s2[x] + coeff3_32 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.rows - 2);
				float* s1 = remap.ptr<float>(remap.rows - 1);
				uchar* d = dest.ptr<uchar>(remap.rows - 2);
				float e;

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;

				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					s1[x + 2] = s1[x + 2] + coeff2_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
					s1[x + 1] = s1[x + 1] + coeff4_32 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s1[x - 2] = s1[x - 2] + coeff2_32 * e;
					s1[x - 1] = s1[x - 1] + coeff4_32 * e;
					s1[x] = s1[x] + coeff5_32 * e;
				}
			}
			{
				float* s = remap.ptr<float>(remap.cols - 1);
				uchar* d = dest.ptr<uchar>(remap.cols - 1);
				float e;

				int x = 0;
				for (x = 0; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
					s[x + 2] = s[x + 2] + coeff3_32 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff5_32 * e;
				}
				{
					if (s[remap.cols - 1] > 0.5f)
					{
						d[remap.cols - 1] = 255;
						sample_num++;
					}
					else
					{
						d[remap.cols - 1] = 0;
					}
				}
			}
		}
		else if (OUT2IN)
		{
			int r = remap.cols / 2;
			int y = 0;
			int x = 0;
			int xend = 0;
			int yend = 0;
			int index = 0;

			for (int lr = r; lr >= 3; lr--)
			{
				x = index;
				y = index;
				xend = remap.cols - index;
				yend = remap.rows - index;

				for (; x < xend; x++)
				{
					float* s = remap.ptr<float>(y);
					float* s1 = remap.ptr<float>(y + 1);
					float* s2 = remap.ptr<float>(y + 2);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					if (x == index)
					{
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					else if (x == index + 1)
					{
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					else if (x == xend - 2)
					{
						s1[x - 2] += coeff2_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s[x + 1] += coeff5_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
					else if (x == xend - 1)
					{
						s1[x] += coeff5_32 * e;
						s2[x] += coeff3_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s1[x - 2] += coeff2_32 * e;
					}
					else
					{
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s1[x] += coeff5_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s2[x] += coeff3_32 * e;
						s2[x + 1] += coeff2_32 * e;
					}
				}
				for (x--, y++; y < yend; y++)
				{
					float* s = remap.ptr<float>(y);
					uchar* d = dest.ptr<uchar>(y);
					float e;

					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					if (y == index + 1)
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s2 = remap.ptr<float>(y + 2);
						s1[x] += coeff5_32 * e;
						s2[x] += coeff3_32 * e;
						s[x - 1] += coeff5_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s[x - 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;
					}
					else if (y == index + 2)
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s2 = remap.ptr<float>(y + 2);
						float* s0 = remap.ptr<float>(y - 1);
						s1[x] += coeff5_32 * e;
						s2[x] += coeff3_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s[x - 1] += coeff5_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s[x - 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;
					}
					else if (y == yend - 2)
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s0 = remap.ptr<float>(y - 1);
						float* s00 = remap.ptr<float>(y - 2);
						s1[x] += coeff5_32 * e;
						s00[x - 1] += coeff2_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s[x - 1] += coeff5_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s[x - 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;

					}
					else if (y == yend - 1)
					{
						float* s0 = remap.ptr<float>(y - 1);
						float* s00 = remap.ptr<float>(y - 2);
						s[x - 1] += coeff5_32 * e;
						s[x - 2] += coeff3_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s00[x - 1] += coeff2_32 * e;
					}
					else
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s2 = remap.ptr<float>(y + 2);
						float* s0 = remap.ptr<float>(y - 1);
						float* s00 = remap.ptr<float>(y - 2);
						s1[x] += coeff5_32 * e;
						s2[x] += coeff3_32 * e;
						s00[x - 1] += coeff2_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s[x - 1] += coeff5_32 * e;
						s1[x - 1] += coeff4_32 * e;
						s2[x - 1] += coeff2_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s[x - 2] += coeff3_32 * e;
						s1[x - 2] += coeff2_32 * e;
					}
				}
				for (x--, y--; x >= index; x--)
				{
					float* s = remap.ptr<float>(y);
					float* s0 = remap.ptr<float>(y - 1);
					float* s00 = remap.ptr<float>(y - 2);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					if (x == xend - 2)
					{
						s[x - 1] += coeff5_32 * e;
						s[x - 2] += coeff3_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x - 1] += coeff2_32 * e;
					}
					else if (x == xend - 3)
					{
						s[x - 1] += coeff5_32 * e;
						s[x - 2] += coeff3_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s00[x + 1] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x - 1] += coeff2_32 * e;
					}
					else if (x == index + 1)
					{
						s[x - 1] += coeff5_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s00[x + 1] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x - 1] += coeff2_32 * e;
					}
					else if (x == index)
					{
						s0[x] += coeff5_32 * e;
						s00[x] += coeff3_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s00[x + 1] += coeff2_32 * e;
					}
					else
					{
						s[x - 1] += coeff5_32 * e;
						s[x - 2] += coeff3_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x - 1] += coeff4_32 * e;
						s0[x - 2] += coeff2_32 * e;
						s00[x + 1] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x - 1] += coeff2_32 * e;
					}
				}
				for (x++, y--; y > index; y--)
				{
					float* s = remap.ptr<float>(y);
					float* s0 = remap.ptr<float>(y - 1);
					float* s00 = remap.ptr<float>(y - 2);
					uchar* d = dest.ptr<uchar>(y);
					float e;
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					if (y == yend - 2)
					{
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x + 1] += coeff2_32 * e;
					}
					else if (y == yend - 3)
					{
						float* s1 = remap.ptr<float>(y + 1);
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x + 1] += coeff2_32 * e;
					}
					else if (y == index + 2)
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s2 = remap.ptr<float>(y + 2);
						s2[x + 1] += coeff2_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x + 2] += coeff2_32 * e;
					}
					else if (y == index + 1)
					{
						float* s1 = remap.ptr<float>(y + 1);
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
					}
					else
					{
						float* s1 = remap.ptr<float>(y + 1);
						float* s2 = remap.ptr<float>(y + 2);
						s2[x + 1] += coeff2_32 * e;
						s1[x + 1] += coeff4_32 * e;
						s1[x + 2] += coeff2_32 * e;
						s[x + 1] += coeff5_32 * e;
						s[x + 2] += coeff3_32 * e;
						s0[x] += coeff5_32 * e;
						s0[x + 1] += coeff4_32 * e;
						s0[x + 2] += coeff2_32 * e;
						s00[x] += coeff3_32 * e;
						s00[x + 1] += coeff2_32 * e;
					}
				}
				index++;
			}
			{
				x = r - 2;
				y = r - 2;
				float e;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x + 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y - 2, x + 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x + 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y - 1, x + 2) += coeff2_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x + 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y - 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 2, x) += coeff3_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 2, x + 1) += coeff2_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y - 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y, x + 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x + 2) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x + 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 1, x + 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 2, x + 1) += coeff2_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y + 2, x) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y + 2, x - 1) += coeff2_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y + 1, x) += coeff5_32 * e;
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y + 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				remap.at<float>(y + 1, x - 2) += coeff2_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y, x - 2) += coeff3_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				remap.at<float>(y - 1, x - 2) += coeff2_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 1, x - 1) += coeff4_32 * e;
				x--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y - 1, x) += coeff5_32 * e;
				remap.at<float>(y - 1, x + 1) += coeff4_32 * e;
				y--;
				if (remap.at<float>(y, x) > 0.5f)
				{
					e = remap.at<float>(y, x) - 1.f;
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y, x);
					dest.at<uchar>(y, x) = 0;
				}
				remap.at<float>(y, x - 1) += coeff5_32 * e;
				x++;
				if (remap.at<float>(y, x) > 0.5f)
				{
					dest.at<uchar>(y, x) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y, x) = 0;
				}
			}
		}
		return sample_num;
	}

	int ditheringSierra2line(Mat& remap, Mat& dest, const bool isMeandering)
	{
		CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff4_16 = 4.f / 16;
		const float coeff3_16 = 3.f / 16;
		const float coeff1_16 = 1.f / 16;
		const float coeff2_16 = 2.f / 16;

		if (isMeandering)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s_next = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;//error

				int x = 0;
				if (y % 2 == 1)
				{
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff4_16 * e;
						s[x + 2] = s[x + 2] + coeff3_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff4_16 * e;
						s[x + 2] = s[x + 2] + coeff3_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff4_16 * e;
						s[x + 2] = s[x + 2] + coeff3_16 * e;
						s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff4_16 * e;
						s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
					}
				}
				else
				{
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff4_16 * e;
						s[x - 2] = s[x - 2] + coeff3_16 * e;
						s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
					}

					for (x = remap.cols - 2; x > 1; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff4_16 * e;
						s[x - 2] = s[x - 2] + coeff3_16 * e;
						s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
					x = 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x - 1] = s[x - 1] + coeff4_16 * e;
						s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s_next[x] = s_next[x] + coeff3_16 * e;
						s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
						s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					}
				}
			}

		}
		else
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float* s = remap.ptr<float>(y);
				float* s_next = remap.ptr<float>(y + 1);
				uchar* d = dest.ptr<uchar>(y);
				float e;
				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff4_16 * e;
					s[x + 2] = s[x + 2] + coeff3_16 * e;
					s_next[x] = s_next[x] + coeff3_16 * e;
					s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
					s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff4_16 * e;
					s[x + 2] = s[x + 2] + coeff3_16 * e;
					s_next[x] = s_next[x] + coeff3_16 * e;
					s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
					s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
					s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
					s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
				}
				for (x = 2; x < remap.cols - 2; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff4_16 * e;
					s[x + 2] = s[x + 2] + coeff3_16 * e;
					s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
					s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
					s_next[x] = s_next[x] + coeff3_16 * e;
					s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
					s_next[x + 2] = s_next[x + 2] + coeff1_16 * e;
				}
				x = remap.cols - 2;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + coeff4_16 * e;
					s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
					s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
					s_next[x] = s_next[x] + coeff3_16 * e;
					s_next[x + 1] = s_next[x + 1] + coeff2_16 * e;
				}
				x++;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s_next[x - 2] = s_next[x - 2] + coeff1_16 * e;
					s_next[x - 1] = s_next[x - 1] + coeff2_16 * e;
					s_next[x] = s_next[x] + coeff3_16 * e;
				}

			}
		}
		{
			float* s = remap.ptr<float>(remap.rows - 1);
			uchar* d = dest.ptr<uchar>(remap.rows - 1);
			float e;
			int x = 0;
			for (x = 0; x < remap.cols - 2; x++)
			{
				if (s[x] > 0.5f)
				{
					e = s[x] - 1.f;
					d[x] = 255;
					sample_num++;
				}
				else
				{
					e = s[x];
					d[x] = 0;
				}
				s[x + 1] = s[x + 1] + coeff4_16 * e;
				s[x + 2] = s[x + 2] + coeff3_16 * e;
			}
			x = remap.cols - 2;
			{
				if (s[x] > 0.5f)
				{
					e = s[x] - 1.f;
					d[x] = 255;
					sample_num++;
				}
				else
				{
					e = s[x];
					d[x] = 0;
				}
				s[x + 1] = s[x + 1] + coeff4_16 * e;
			}
			if (s[remap.cols - 1] > 0.5f)
			{
				d[remap.cols - 1] = 255;
				sample_num++;
			}
			else
			{
				d[remap.cols - 1] = 0;
			}
		}
		return sample_num;
	}

	int ditherDestruction(Mat& src, Mat& dest, const int dithering_method, int process_order)
	{
		int sample_num;
		if (dithering_method == FLOYD_STEINBERG)
			sample_num = ditheringFloydSteinberg(src, dest, process_order);
		else if (dithering_method == OSTROMOUKHOW)
			sample_num = ditheringOstromoukhov(src, dest, process_order);
		else if (dithering_method == JARVIS)
			sample_num = ditheringJarvis(src, dest, process_order);
		else if (dithering_method == SIERRA2)
			sample_num = ditheringSierra2line(src, dest, process_order);
		else if (dithering_method == SIERRA3)
			sample_num = ditheringSierra3line(src, dest, process_order);
		else if (dithering_method == STUCKI)
			sample_num = ditheringStucki(src, dest, process_order);
		else if (dithering_method == BURKES)
			sample_num = ditheringBurkes(src, dest, process_order);
		else if (dithering_method == STEAVENSON)
			sample_num = ditheringSteavenson(src, dest, process_order);

		return sample_num;
	}

	int dither(const Mat& src_, Mat& dest, const int dithering_method, int process_order)
	{
		Mat src = src_.clone();

		int sample_num;
		if (dithering_method == OSTROMOUKHOW)
			sample_num = ditheringOstromoukhov(src, dest, process_order);
		else if (dithering_method == FLOYD_STEINBERG)
			sample_num = ditheringFloydSteinberg(src, dest, process_order);
		else if (dithering_method == JARVIS)
			sample_num = ditheringJarvis(src, dest, process_order);
		else if (dithering_method == SIERRA2)
			sample_num = ditheringSierra2line(src, dest, process_order);
		else if (dithering_method == SIERRA3)
			sample_num = ditheringSierra3line(src, dest, process_order);
		else if (dithering_method == STUCKI)
			sample_num = ditheringStucki(src, dest, process_order);
		else if (dithering_method == BURKES)
			sample_num = ditheringBurkes(src, dest, process_order);
		else if (dithering_method == STEAVENSON)
			sample_num = ditheringSteavenson(src, dest, process_order);

		return sample_num;
	}

	void ditheringOrderViz(Mat& src, int process_order)
	{
		/*if (process_order == MEANDERING)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float *s = remap.ptr<float>(y);
				float *s_next = remap.ptr<float>(y + 1);
				uchar *d = dest.ptr<uchar>(y);
				float e;//error

				int x = 0;
				if (y % 2 == 1) //odd
				{
					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x + 1] = s[x + 1] + e * coeff7_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
					}
					for (x = 1; x < remap.cols - 1; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x + 1] = s[x + 1] + e * coeff7_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
					}
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
					}
				}
				else //even
				{
					x = remap.cols - 1;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x - 1] = s[x - 1] + e * coeff7_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff1_16;
					}
					for (x = remap.cols - 2; x > 0; x--)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}

						s[x - 1] = s[x - 1] + e * coeff7_16;
						s_next[x + 1] = s_next[x + 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
						s_next[x - 1] = s_next[x - 1] + e * coeff1_16;
					}

					x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s_next[x + 1] = s_next[x + 1] + e * coeff3_16;
						s_next[x] = s_next[x] + e * coeff5_16;
					}
				}
			}
			// bottom y loop
			{
				float *s = remap.ptr<float>(remap.rows - 1);
				uchar *d = dest.ptr<uchar>(remap.rows - 1);
				float e;//error
				for (int x = 0; x < remap.cols - 1; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						//e = 1.f - s[x];
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + e * coeff7_16;
				}

				//x=remap.cols - 1
				if (s[remap.cols - 1] > 0.5f)
				{
					d[remap.cols - 1] = 255;
					sample_num++;
				}
				else
				{
					d[remap.cols - 1] = 0;
				}
			}
		}
		else if (process_order == FORWARD)
		{
			for (int y = 0; y < remap.rows - 1; y++)
			{
				float *s = remap.ptr<float>(y);
				float *s_next = remap.ptr<float>(y + 1);
				uchar *d = dest.ptr<uchar>(y);
				float e;//error

				int x = 0;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					s[x + 1] = s[x + 1] + e * coeff7_16;
					s_next[x] = s_next[x] + e * coeff5_16;
					s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
				}

				for (x = 1; x < remap.cols - 1; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}

					s[x + 1] = s[x + 1] + e * coeff7_16;
					s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
					s_next[x] = s_next[x] + e * coeff5_16;
					s_next[x + 1] = s_next[x + 1] + e * coeff1_16;
				}

				x = remap.cols - 1;
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s_next[x - 1] = s_next[x - 1] + e * coeff3_16;
					s_next[x] = s_next[x] + e * coeff5_16;
				}
			}
			// bottom y loop
			{
				float *s = remap.ptr<float>(remap.rows - 1);
				uchar *d = dest.ptr<uchar>(remap.rows - 1);
				float e;//error
				for (int x = 0; x < remap.cols - 1; x++)
				{
					if (s[x] > 0.5f)
					{
						e = s[x] - 1.f;
						//e = 1.f - s[x];
						d[x] = 255;
						sample_num++;
					}
					else
					{
						e = s[x];
						d[x] = 0;
					}
					s[x + 1] = s[x + 1] + e * coeff7_16;
				}

				//x=remap.cols - 1
				if (s[remap.cols - 1] > 0.5f)
				{
					d[remap.cols - 1] = 255;
					sample_num++;
				}
				else
				{
					d[remap.cols - 1] = 0;
				}
			}
		}
		else if (process_order == IN2OUT)
		{
			int r = remap.cols / 2;
			int y = 0;
			{
				int x = 0;
				float e;
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					e = remap.at<float>(y + r, x + r) - 1.f;
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					e = remap.at<float>(y + r, x + r);
					dest.at<uchar>(y + r, x + r) = 0;
				}
				remap.at<float>(y + r, x + r - 1) += coeff7_16 * e;
				remap.at<float>(y + r + 1, x + r - 1) += coeff1_16 * e;
				remap.at<float>(y + r + 1, x + r) += coeff5_16 * e;
				remap.at<float>(y + r + 1, x + r + 1) += coeff3_16 * e;
			}

			for (int lr = 1; lr <= r - 1; lr++)
			{
				int x = -lr;
				y = 0;
				for (; y >= -lr; y--)
				{
					float *s = remap.ptr<float>(y + r);
					float *s0 = remap.ptr<float>(y + r - 1);
					float *s1 = remap.ptr<float>(y + r + 1);
					uchar *d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == -lr)
					{
						s[x + r + 1] += coeff7_16 * e;
						s0[x + r + 1] += coeff1_16 * e;
						s0[x + r] += coeff5_16 * e;
						s0[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
				}
				for (x++, y++; x <= lr; x++)
				{
					float *s = remap.ptr<float>(y + r);
					float *s0 = remap.ptr<float>(y + r - 1);
					uchar *d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (x == lr)
					{
						float *s1 = remap.ptr<float>(y + r + 1);
						s1[x + r] += coeff7_16 * e;
						s0[x + r + 1] += coeff3_16 * e;
						s[x + r + 1] += coeff5_16 * e;
						s1[x + r + 1] += coeff1_16 * e;
					}
					else
					{
						s[x + r + 1] += coeff7_16 * e;
						s0[x + r - 1] += coeff3_16 * e;
						s0[x + r] += coeff5_16 * e;
						s0[x + r + 1] += coeff1_16 * e;
					}
				}
				for (x--, y++; y <= lr; y++)
				{
					float *s = remap.ptr<float>(y + r);
					float *s0 = remap.ptr<float>(y + r - 1);
					float *s1 = remap.ptr<float>(y + r + 1);
					uchar *d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == lr)
					{
						s[x + r - 1] += coeff7_16 * e;
						s1[x + r - 1] += coeff1_16 * e;
						s1[x + r] += coeff5_16 * e;
						s1[x + r + 1] += coeff3_16 * e;
					}
					else
					{
						s1[x + r] += coeff7_16 * e;
						s0[x + r + 1] += coeff3_16 * e;
						s[x + r + 1] += coeff5_16 * e;
						s1[x + r + 1] += coeff1_16 * e;
					}
				}
				for (x--, y--; x >= -lr; x--)
				{
					float *s = remap.ptr<float>(y + r);
					float *s1 = remap.ptr<float>(y + r + 1);
					uchar *d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (x == -lr)
					{
						float *s0 = remap.ptr<float>(y + r - 1);
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s[x + r - 1] += coeff7_16 * e;
						s1[x + r - 1] += coeff1_16 * e;
						s1[x + r] += coeff5_16 * e;
						s1[x + r + 1] += coeff3_16 * e;
					}
				}
				for (x++, y--; y > 0; y--)
				{
					float *s = remap.ptr<float>(y + r);
					float *s0 = remap.ptr<float>(y + r - 1);
					float *s1 = remap.ptr<float>(y + r + 1);
					uchar *d = dest.ptr<uchar>(y + r);
					float e;
					if (s[x + r] > 0.5f)
					{
						e = s[x + r] - 1.f;
						d[x + r] = 255;
						sample_num++;
					}
					else
					{
						e = s[x + r];
						d[x + r] = 0;
					}
					if (y == 1)
					{
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
					else
					{
						s0[x + r] += coeff7_16 * e;
						s0[x + r - 1] += coeff1_16 * e;
						s[x + r - 1] += coeff5_16 * e;
						s1[x + r - 1] += coeff3_16 * e;
					}
				}

			}
			int lr = r;
			int x = -lr;
			for (; y > -lr; y--)
			{
				float *s = remap.ptr<float>(y + r);
				float *s0 = remap.ptr<float>(y + r - 1);
				float *s1 = remap.ptr<float>(y + r + 1);
				uchar *d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s0[x + r] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (x++; x < lr; x++)
			{
				float *s = remap.ptr<float>(y + r);
				float *s0 = remap.ptr<float>(y + r - 1);
				uchar *d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s[x + r + 1] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (y++; y < lr; y++)
			{
				float *s = remap.ptr<float>(y + r);
				float *s0 = remap.ptr<float>(y + r - 1);
				float *s1 = remap.ptr<float>(y + r + 1);
				uchar *d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s1[x + r] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (x--; x > -lr; x--)
			{
				float *s = remap.ptr<float>(y + r);
				float *s1 = remap.ptr<float>(y + r + 1);
				uchar *d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				s[x + r - 1] += coeff7_16 * e;
			}
			{
				if (remap.at<float>(y + r, x + r) > 0.5f)
				{
					dest.at<uchar>(y + r, x + r) = 255;
					sample_num++;
				}
				else
				{
					dest.at<uchar>(y + r, x + r) = 0;
				}
			}
			for (y--; y > 0; y--)
			{
				float *s = remap.ptr<float>(y + r);
				float *s0 = remap.ptr<float>(y + r - 1);
				float *s1 = remap.ptr<float>(y + r + 1);
				uchar *d = dest.ptr<uchar>(y + r);
				float e;
				if (s[x + r] > 0.5f)
				{
					e = s[x + r] - 1.f;
					d[x + r] = 255;
					sample_num++;
				}
				else
				{
					e = s[x + r];
					d[x + r] = 0;
				}
				if (y != 1)
					s0[x + r] += coeff7_16 * e;
			}

		}
		else if (process_order == OUT2IN)*/
		//{

		//	int r = dest.cols / 2;
		//	int y = 0;
		//	int x = 0;
		//	int xend = 0;
		//	int yend = 0;
		//	int index = 0;
		//	string wname = "sample";
		//	int t = 1;
		//	for (int lr = r; lr >= 2; lr--)
		//	{
		//		x = index;
		//		y = index;
		//		xend = dest.cols - index;
		//		yend = dest.rows - index;

		//		for (; x < xend; x++)
		//		{
		//			uchar *d = dest.ptr<uchar>(y);
		//			d[x] = 255;
		//			imshow(wname, dest); waitKey(t);

		//		}
		//		for (x--, y++; y < yend; y++)
		//		{
		//			uchar *d = dest.ptr<uchar>(y);
		//			d[x] = 255;
		//			imshow(wname, dest); waitKey(t);
		//		}

		//		for (x--, y--; x >= index; x--)
		//		{
		//			uchar *d = dest.ptr<uchar>(y);
		//			d[x] = 255;
		//			imshow(wname, dest); waitKey(t);
		//		}
		//		for (x++, y--; y > index; y--)
		//		{
		//			uchar *d = dest.ptr<uchar>(y);
		//			d[x] = 255;
		//			imshow(wname, dest); waitKey(t);
		//		}
		//		index++;
		//	}
		//	{
		//		x = r - 1;
		//		y = r - 1;
		//		uchar *d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		imshow(wname, dest); waitKey(t);

		//		x++;
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		imshow(wname, dest); waitKey(t);

		//		x++;
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		imshow(wname, dest); waitKey(t);
		//		y++;
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		imshow(wname, dest); waitKey(t);
		//		y++;
		//		d = dest.ptr<uchar>(y);
		//		imshow(wname, dest); waitKey(t);
		//		d[x] = 255;
		//		x--;
		//		imshow(wname, dest); waitKey(t);
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		x--;
		//		imshow(wname, dest); waitKey(t);
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		y--;
		//		imshow(wname, dest); waitKey(t);
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		x++;
		//		imshow(wname, dest); waitKey(t);
		//		d = dest.ptr<uchar>(y);
		//		d[x] = 255;
		//		imshow(wname, dest); waitKey(t);
		//	}
		//}

		//CV_Assert(remap.depth() == CV_32F);
		int sample_num = 0;
		const float coeff5_32 = 5.f / 32;
		const float coeff3_32 = 3.f / 32;
		const float coeff2_32 = 2.f / 32;
		const float coeff4_32 = 4.f / 32;
		string wname = "sample";

		if (process_order == MEANDERING)
		{
			/*	for (int y = 0; y < remap.rows - 2; y++)
				{
					float *s = remap.ptr<float>(y);
					float *s1 = remap.ptr<float>(y + 1);
					float *s2 = remap.ptr<float>(y + 2);
					uchar *d = dest.ptr<uchar>(y);
					float e;

					if (y % 2 == 1)
					{
						int x = 0;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += coeff5_32 * e;
							s[x + 2] += coeff3_32 * e;
							s1[x] += coeff5_32 * e;
							s1[x + 1] += coeff4_32 * e;
							s1[x + 2] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
							s2[x + 1] += coeff2_32 * e;
						}
						x = 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += coeff5_32 * e;
							s[x + 2] += coeff3_32 * e;
							s1[x - 1] += coeff4_32 * e;
							s1[x] += coeff5_32 * e;
							s1[x + 1] += coeff4_32 * e;
							s1[x + 2] += coeff2_32 * e;
							s2[x - 1] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
							s2[x + 1] += coeff2_32 * e;
						}
						for (x = 2; x < remap.cols - 2; x++)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += coeff5_32 * e;
							s[x + 2] += coeff3_32 * e;
							s1[x - 2] += coeff2_32 * e;
							s1[x - 1] += coeff4_32 * e;
							s1[x] += coeff5_32 * e;
							s1[x + 1] += coeff4_32 * e;
							s1[x + 2] += coeff2_32 * e;
							s2[x - 1] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
							s2[x + 1] += coeff2_32 * e;
						}
						x = remap.cols - 2;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] += coeff5_32 * e;
							s1[x - 2] += coeff2_32 * e;
							s1[x - 1] += coeff4_32 * e;
							s1[x] += coeff5_32 * e;
							s1[x + 1] += coeff4_32 * e;
							s2[x - 1] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
							s2[x + 1] += coeff2_32 * e;
							}
						x = remap.cols - 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x - 2] += coeff2_32 * e;
							s1[x - 1] += coeff4_32 * e;
							s1[x] += coeff5_32 * e;
							s2[x - 1] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
						}
						}
					else
					{
						int x = remap.cols - 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
							s2[x] = s2[x] + coeff3_32 * e;
							s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						}
						x = remap.cols - 2;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
							s2[x + 1] = s2[x + 1] + coeff2_32 * e;
							s2[x] = s2[x] + coeff3_32 * e;
							s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						}
						for (x = remap.cols - 3; x > 1; x--)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] += coeff5_32 * e;
							s[x - 2] += coeff3_32 * e;
							s1[x + 2] += coeff2_32 * e;
							s1[x + 1] += coeff4_32 * e;
							s1[x] += coeff5_32 * e;
							s1[x - 1] += coeff4_32 * e;
							s1[x - 2] += coeff2_32 * e;
							s2[x + 1] += coeff2_32 * e;
							s2[x] += coeff3_32 * e;
							s2[x - 1] += coeff2_32 * e;
						}
						x = 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s2[x + 1] = s2[x + 1] + coeff2_32 * e;
							s2[x] = s2[x] + coeff3_32 * e;
							s2[x - 1] = s2[x - 1] + coeff2_32 * e;
					}
						x = 0;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s2[x + 1] = s2[x + 1] + coeff2_32 * e;
							s2[x] = s2[x] + coeff3_32 * e;
						}
					}
				}
				{
					float *s = remap.ptr<float>(remap.rows - 2);
					float *s1 = remap.ptr<float>(remap.rows - 1);
					uchar *d = dest.ptr<uchar>(remap.rows - 2);
					float e;
					if ((remap.rows - 2) % 2 == 1)
					{
						int x = 0;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
							s[x + 2] = s[x + 2] + coeff3_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						}
						x++;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
							s[x + 2] = s[x + 2] + coeff3_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						}
						for (x = 2; x < remap.cols - 2; x++)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
							s[x + 2] = s[x + 2] + coeff3_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						}
						x = remap.cols - 2;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
						}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					}
						x++;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
						}
				}
					else
					{
						int x = remap.cols - 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						}
						x--;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						}
						for (x = remap.cols - 3; x > 1; x--)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
							s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						}
						x = 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
							s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						}
						x--;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s1[x + 2] = s1[x + 2] + coeff2_32 * e;
							s1[x + 1] = s1[x + 1] + coeff4_32 * e;
							s1[x] = s1[x] + coeff5_32 * e;
						}
					}
			}
				{
					float *s = remap.ptr<float>(remap.rows - 1);
					uchar *d = dest.ptr<uchar>(remap.rows - 1);
					float e;
					if ((remap.rows - 1) % 2 == 1)
					{
						int x = 0;
						for (; x < remap.cols - 2; x++)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
							s[x + 2] = s[x + 2] + coeff3_32 * e;
						}
						x = remap.cols - 2;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x + 1] = s[x + 1] + coeff5_32 * e;
						}
						{
							if (s[remap.cols - 1] > 0.5f)
							{
								d[remap.cols - 1] = 255;
								sample_num++;
							}
							else
							{
								d[remap.cols - 1] = 0;
							}
						}
							}
					else
					{
						int x = remap.cols - 1;
						for (; x > 1; x--)
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
							s[x - 2] = s[x - 2] + coeff3_32 * e;
						}
						x = 1;
						{
							if (s[x] > 0.5f)
							{
								e = s[x] - 1.f;
								d[x] = 255;
								sample_num++;
							}
							else
							{
								e = s[x];
								d[x] = 0;
							}
							s[x - 1] = s[x - 1] + coeff5_32 * e;
						}
						{
							if (s[remap.cols - 1] > 0.5f)
							{
								d[remap.cols - 1] = 255;
								sample_num++;
							}
							else
							{
								d[remap.cols - 1] = 0;
							}
						}
					}
						}
					}
			else if (FORWARD)
			{
				for (int y = 0; y < remap.rows - 2; y++)
				{
					float *s = remap.ptr<float>(y);
					float *s1 = remap.ptr<float>(y + 1);
					float *s2 = remap.ptr<float>(y + 2);
					uchar *d = dest.ptr<uchar>(y);
					float e;

					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
						s2[x + 1] = s2[x + 1] + coeff2_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s2[x - 1] = s2[x - 1] + coeff2_32 * e;
						s2[x] = s2[x] + coeff3_32 * e;
					}
				}
				{
					float *s = remap.ptr<float>(remap.rows - 2);
					float *s1 = remap.ptr<float>(remap.rows - 1);
					uchar *d = dest.ptr<uchar>(remap.rows - 2);
					float e;

					int x = 0;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;

					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					for (x = 2; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
						s1[x + 2] = s1[x + 2] + coeff2_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
						s1[x + 1] = s1[x + 1] + coeff4_32 * e;
					}
					x++;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s1[x - 2] = s1[x - 2] + coeff2_32 * e;
						s1[x - 1] = s1[x - 1] + coeff4_32 * e;
						s1[x] = s1[x] + coeff5_32 * e;
					}
				}
				{
					float *s = remap.ptr<float>(remap.cols - 1);
					uchar *d = dest.ptr<uchar>(remap.cols - 1);
					float e;

					int x = 0;
					for (x = 0; x < remap.cols - 2; x++)
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
						s[x + 2] = s[x + 2] + coeff3_32 * e;
					}
					x = remap.cols - 2;
					{
						if (s[x] > 0.5f)
						{
							e = s[x] - 1.f;
							d[x] = 255;
							sample_num++;
						}
						else
						{
							e = s[x];
							d[x] = 0;
						}
						s[x + 1] = s[x + 1] + coeff5_32 * e;
					}
					{
						if (s[remap.cols - 1] > 0.5f)
						{
							d[remap.cols - 1] = 255;
							sample_num++;
						}
						else
						{
							d[remap.cols - 1] = 0;
						}
					}
				}*/
		}
		else if (OUT2IN)
		{
			int r = src.cols / 2;
			int y = 0;
			int x = 0;
			int xend = 0;
			int yend = 0;
			int index = 0;
			int t = 1;

			for (int lr = r; lr >= 3; lr--)
			{
				x = index;
				y = index;
				xend = src.cols - index;
				yend = src.rows - index;

				for (; x < xend; x++)
				{
					float* s = src.ptr<float>(y);
					float* s1 = src.ptr<float>(y + 1);
					float* s2 = src.ptr<float>(y + 2);
					uchar* d = src.ptr<uchar>(y);

					if (x == index)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == index + 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == xend - 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == xend - 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
				}
				for (x--, y++; y < yend; y++)
				{
					float* s = src.ptr<float>(y);
					uchar* d = src.ptr<uchar>(y);

					if (y == index + 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (y == index + 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (y == yend - 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);

					}
					else if (y == yend - 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
				}
				for (x--, y--; x >= index; x--)
				{
					float* s = src.ptr<float>(y);
					float* s0 = src.ptr<float>(y - 1);
					float* s00 = src.ptr<float>(y - 2);
					uchar* d = src.ptr<uchar>(y);
					if (x == xend - 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == xend - 3)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == index + 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (x == index)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
				}
				for (x++, y--; y > index; y--)
				{
					float* s = src.ptr<float>(y);
					float* s0 = src.ptr<float>(y - 1);
					float* s00 = src.ptr<float>(y - 2);
					uchar* d = src.ptr<uchar>(y);
					if (y == yend - 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (y == yend - 3)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (y == index + 2)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else if (y == index + 1)
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
					else
					{
						d[x] = 255;
						imshow(wname, src); waitKey(t);
					}
				}
				index++;
			}
			{
				x = r - 2;
				y = r - 2;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				y--;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
				x++;

				src.at<uchar>(y, x) = 255;
				imshow(wname, src); waitKey(t);
			}
		}
		waitKey(0);
	}
}