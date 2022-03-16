#pragma once
#ifndef FPPLUS_DD_H
#define FPPLUS_DD_H

#include <fpplus/eft.h>

/**
 * @defgroup DD Double-double arithmetic
 */


 /**
  * @ingroup DD
  * @brief Double-double number.
  */
typedef struct {
	/**
	 * @brief The high (largest in magnitude) part of the number.
	 * @note The high part is the best double-precision approximation of the double-double number.
	 */
	double hi;
	/**
	 * @brief The low (smallest in magnitude) part of the number.
	 */
	double lo;
} doubledouble;

static void ignore_space(std::string& s)
{
	int p = 0;
	while (p < s.size() && isspace(s[p])) p++;
	s = s.substr(p);
}

static int get_sign(std::string& s)
{
	int r, p;

	p = 0;
	r = 1;
	if (p < s.size() && s[p] == '-') {
		r = -1;
		p++;
	}
	else if (p < s.size() && s[p] == '+') {
		r = 1;
		p++;
	}

	s = s.substr(p);
	return r;
}

static std::string get_number(std::string& s)
{
	std::string r;
	int p;

	p = 0;
	r = "";
	while (p < s.size() && isdigit(s[p]))
	{
		r += s[p];
		p++;
	}

	s = s.substr(p);
	return r;
}

// mode == -1 : down
// mode ==  0 : nearest
// mode ==  1 : up
static void string2DD(std::string s, double& x1, double& x2, const int mode = 0, const bool fast = false) {
	int i, tmp;
	bool flag;
	int sign, e10, esign;
	std::string num1_s, num2_s, nume_s;

	ignore_space(s);
	sign = get_sign(s);
	num1_s = get_number(s);

	if (0 < s.size() && s[0] == '.') {
		s = s.substr(1);
		num2_s = get_number(s);
	}

	if (0 < s.size() && (s[0] == 'e' || s[0] == 'E')) {
		s = s.substr(1);
		esign = get_sign(s);
		nume_s = get_number(s);
		e10 = esign * atoi(nume_s.c_str());
	}
	else {
		e10 = 0;
	}

	// delete 0s from the head of num1_s
	while (0 < num1_s.size() && num1_s[0] == '0') {
		num1_s = num1_s.substr(1);
	}

	// delete 0s from the tail of num2_s
	while (0 < num2_s.size() && num2_s[num2_s.size() - 1] == '0') {
		num2_s = num2_s.substr(0, num2_s.size() - 1);
	}

	// set table and offset
	// |x| = \sum_{table_min}^{table_max} table[offset + i] * 10^i
	int table_max, table_min, offset;
	std::vector<int> table;

	table_max = (int)num1_s.size() - 1 + e10;
	table_min = e10 - (int)num2_s.size();
	table.resize(table_max - table_min + 1);
	offset = -table_min;

	for (i = 0; i < num1_s.size(); i++) {
		table[offset + num1_s.size() - 1 - i + e10] = num1_s[i] - '0';
	}

	for (i = 0; i < num2_s.size(); i++) {
		table[offset - i - 1 + e10] = num2_s[i] - '0';
	}

	// extend table
	if (table_min > 0) {
		tmp = (int)table.size();
		table.resize(tmp + table_min);
		for (i = tmp - 1; i >= 0; i--) {
			table[i + table_min] = table[i];
		}
		for (i = 0; i < table_min; i++) {
			table[i] = 0;
		}
		offset += table_min;
		table_min = 0;
	}

	if (table_max < -1) {
		tmp = (int)table.size();
		table.resize(tmp + (-1 - table_max));
		for (i = 0; i < (-1 - table_max); i++) {
			table[tmp + i] = 0;
		}
		table_max = -1;
	}

#if 0
	for (i = table_max; i >= table_min; i--) {
		std::cout << i << ':' << table[offset + i] << "\n";
	}
#endif

	// convert decimal number to binary number
	// set result and offset2
	// |x| = \sum_{result_min}^{reuslt_max} result[offset2 + i] * 2^i

	int result_min, result_max, m, pm, carry, carry2;
	std::list<bool> result1, result2;

	// integer part

	result_max = -1;

	while (table_max >= 0) {
		if (table_max >= 5) m = 16;
		else if (table_max >= 4) m = 13;
		else if (table_max >= 3) m = 9;
		else if (table_max >= 2) m = 6;
		else if (table_max >= 1) m = 3;
		else m = 1;
		pm = 1 << m;

		carry = 0;
		for (i = table_max; i >= 0; i--) {
			tmp = carry * 10 + table[offset + i];
			table[offset + i] = tmp / pm;
			carry = tmp % pm;
		}
		for (i = 0; i < m; i++) {
			result_max++;
			result1.push_back(carry % 2);
			carry = carry / 2;
		}
		while (table_max >= 0 && table[offset + table_max] == 0) {
			table_max--;
		}
	}

	// fraction part

	//  flag means whether most significant bit already found or not
	if (result_max >= 0) flag = true;
	else flag = false;

	result_min = 0;

	while (table_min < 0) {
		if (fast) {
			tmp = 106 - (result_max - result_min);
		}
		else {
			tmp = result_min + 1075;
		}
		if (flag && tmp <= 0) break;
		if (!flag) {
			m = 16;
		}
		else {
			m = std::min(16, tmp);
		}
		pm = 1 << m;

		carry = 0;
		for (i = table_min; i <= -1; i++) {
			tmp = table[offset + i] * pm + carry;
			table[offset + i] = tmp % 10;
			carry = tmp / 10;
		}

		for (i = 0; i < m; i++) {
			result_min--;
			pm /= 2;
			carry2 = carry / pm;
			carry = carry % pm;

			if (flag) {
				result2.push_back(carry2);
			}
			else {
				if (carry2 != 0) {
					result2.push_back(carry2);
					result_max = result_min;
					flag = true;
				}
			}
		}

		while (table_min < 0 && table[offset + table_min] == 0) {
			table_min++;
		}
	}

	// append integer and fraction part

	std::vector<bool> result;
	int offset2;

	result.resize(result_max - result_min + 1);
	offset2 = -result_min;
	for (i = 0; i <= result_max; i++) {
		result[offset2 + i] = result1.front();
		result1.pop_front();
	}
	for (i = std::min(-1, result_max); i >= result_min; i--) {
		result[offset2 + i] = result2.front();
		result2.pop_front();
	}

#if 0
	for (i = result_max; i >= result_min; i--) {
		printf("%d %d\n", i, result[offset2 + i]);
	}
#endif

	// convert binary to double double number

	double dtmp;

	if (result_max > 1023) {
		if ((sign == 1 && mode == -1) || (sign == -1 && mode == 1)) {
			x1 = sign * (std::numeric_limits<double>::max)();
			x2 = std::ldexp(x1, -54);
			return;
		}
		dtmp = sign * std::numeric_limits<double>::infinity();
		x1 = dtmp;
		x2 = dtmp;
		return;
	}

	if (result_max < -1075) {
		if ((sign == 1 && mode == 1) || (sign == -1 && mode == -1)) {
			x1 = sign * std::numeric_limits<double>::denorm_min();
			x2 = sign * 0.;
			return;
		}
		dtmp = sign * 0.;
		x1 = dtmp;
		x2 = dtmp;
		return;
	}

	double r, r2;
	int result_max2;
	int msb;

	r = 0.;
	flag = false; // roundup first part or not
	result_max2 = result_min - 1;
	for (i = result_max; i >= result_min; i--) {
		if (result_max - i == 53 || i == -1075) {
			if (sign == 1) {
				if (result[offset2 + i] == 0) {
				}
				else {
					r += std::ldexp(1., i + 1);
					flag = true;
				}
			}
			else {
				if (result[offset2 + i] == 0) {
				}
				else {
					r += std::ldexp(1., i + 1);
					flag = true;
				}
			}
			result_max2 = i;
			break;
		}
		r += std::ldexp((double)result[offset2 + i], i);
	}

	if (flag) {
		r2 = -std::ldexp(1., result_max2 + 1);
	}
	else {
		r2 = 0.;
	}

	msb = result_min - 1; // outside result bits

	for (i = result_max2; i >= result_min; i--) {
		if (fast) {
			tmp = result_max2 - i;
		}
		else {
			tmp = msb - i;
		}
		if (tmp == 53 || i == -1075) {
			if (sign == 1) {
				if (mode == -1) {
				}
				else if (mode == 0) {
					if (result[offset2 + i] == 0) {
					}
					else {
						r2 += std::ldexp(1., i + 1);
					}
				}
				else {
					r2 += std::ldexp(1., i + 1);
				}
			}
			else {
				if (mode == -1) {
					r2 += std::ldexp(1., i + 1);
				}
				else if (mode == 0) {
					if (result[offset2 + i] == 0) {
					}
					else {
						r2 += std::ldexp(1., i + 1);
					}
				}
				else {
				}
			}
			break;
		}
		tmp = result[offset2 + i];
		r2 += std::ldexp((double)tmp, i);
		if (msb == result_min - 1 && ((flag && tmp == 0) || (!flag && tmp == 1))) {
			msb = i;
		}
	}

	x1 = sign * r;
	x2 = sign * r2;
	return;
}

static doubledouble string2DD(std::string s, const int mode = 0, const bool fast = false)
{
	doubledouble ret;
	string2DD(s, ret.hi, ret.lo, mode, fast);
	return ret;
}

//M_E
static const doubledouble M_E_DD = string2DD("2.7182818284590452353602874713526624977572470937000");
static const doubledouble M_PI_DD = string2DD("3.1415926535897932384626433832795028841971693993751");
static const doubledouble M_2PI_DD = string2DD("6.2831853071795864769252867665590057683943387987502");
static const doubledouble M_SQRT2PI_DD = string2DD("2.5066282746310005024157652848110452530069867406099");
static const doubledouble M_LN2_DD = string2DD("0.69314718055994530941723212145817656807550013436026");


/**
 * @ingroup DD
 * @brief Long addition of double-precision numbers.
 * @details Adds two double-precision numbers and produces double-double result.
 * The algorith is a version of error-free addition due to @cite Knuth1997.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th> <th>Count (default ISA)</th> <th>Count (with ADDRE)</th></tr>
 *         <tr><td>FP ADD   </td> <td>6                  </td> <td>1                 </td></tr>
 *         <tr><td>FP ADDRE </td> <td>                   </td> <td>1                 </td></tr>
 *     </table>
 *
 * @param[in] a - addend, the first double-precision number to be added.
 * @param[in] b - augend, the second double-precision number to be added.
 * @return The sum of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddaddl(const double a, const double b)
{
	doubledouble sum;
	sum.hi = twosum(a, b, &sum.lo);
	return sum;
}

/**
 * @ingroup DD
 * @brief Wide addition of double-precision number to a double-double number.
 * @details Adds double-precision number to a double-double number and produces a double-double result.
 *
 * Implementation follows @cite QD2000, Figure 7.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th> <th>Count (default ISA)</th> <th>Count (with ADDRE)</th></tr>
 *         <tr><td>FP ADD   </td> <td>10                 </td> <td>3                 </td></tr>
 *         <tr><td>FP ADDRE </td> <td>                   </td> <td>2                 </td></tr>
 *     </table>
 *
 * @param[in] a - addend, the double-double number to be added to.
 * @param[in] b - augend, the double-precision number to be added.
 * @return The sum of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddaddw(const doubledouble a, const double b)
{
	doubledouble sum = ddaddl(a.lo, b);
	double e;
	/* QD uses efaddord here. I think it is a bug (what if b > a.hi -> sum.hi > a.hi ?). */
	sum.hi = twosum(a.hi, sum.hi, &e);
#ifdef __CUDA_ARCH__
	sum.lo = __dadd_rn(sum.lo, e);
#else
	sum.lo += e;
#endif
	return sum;
}

FPPLUS_STATIC_INLINE doubledouble ddsubw(const doubledouble a, const double b)
{
	doubledouble sum = ddaddl(a.lo, -b);
	double e;
	/* QD uses efaddord here. I think it is a bug (what if b > a.hi -> sum.hi > a.hi ?). */
	sum.hi = twosum(a.hi, sum.hi, &e);
#ifdef __CUDA_ARCH__
	sum.lo = __dadd_rn(sum.lo, e);
#else
	sum.lo += e;
#endif
	return sum;
}

/**
 * @ingroup DD
 * @brief Addition of two double-double numbers.
 * @details Adds two double-double numbers and produces a double-double result.
 *
 * According to a comment in the source of @cite QD2000, the algorithm due to Briggs and Kahan.
 * Implementation follows @cite FPHandbook2009.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count</th></tr>
 *         <tr><td>FP ADD</td><td>20</td></tr>
 *     </table>
 *
 * @param[in] a - addend, the first double-double number to be added.
 * @param[in] b - augend, the second double-double number to be added.
 * @return The sum of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddadd(const doubledouble a, const doubledouble b)
{
#if 0
	const doubledouble s = ddaddl(a.hi, b.hi);
	const doubledouble t = ddaddl(a.lo, b.lo);
	doubledouble v;
#ifdef __CUDA_ARCH__
	v.hi = efaddord(s.hi, __dadd_rn(s.lo, t.hi), &v.lo);
#else
	v.hi = twosumfast(s.hi, s.lo + t.hi, &v.lo);
#endif
	doubledouble z;
#ifdef __CUDA_ARCH__
	z.hi = efaddord(v.hi, __dadd_rn(t.lo, v.lo), &z.lo);
#else
	z.hi = twosumfast(v.hi, t.lo + v.lo, &z.lo);
#endif
	return z;
#else
	double z1, z2, z3, z4;

	z1 = twosum(a.hi, b.hi, &z2);
	z2 += a.lo + b.lo;
	z3 = twosum(z1, z2, &z4);

	return { z3, z4 };
#endif
}

FPPLUS_STATIC_INLINE doubledouble ddsub(const doubledouble a, const doubledouble b)
{
#if 0
	const doubledouble s = ddaddl(a.hi, -b.hi);
	const doubledouble t = ddaddl(a.lo, b.lo);
	doubledouble v;
#ifdef __CUDA_ARCH__
	v.hi = efaddord(s.hi, __dadd_rn(s.lo, t.hi), &v.lo);
#else
	v.hi = twosumfast(s.hi, s.lo - t.hi, &v.lo);
#endif
	doubledouble z;
#ifdef __CUDA_ARCH__
	z.hi = efaddord(v.hi, __dadd_rn(t.lo, v.lo), &z.lo);
#else
	z.hi = twosumfast(v.hi, t.lo + v.lo, &z.lo);
#endif
	return z;
#else
	double z1, z2, z3, z4;

	z1 = twosum(a.hi, -b.hi, &z2);
	z2 += a.lo - b.lo;
	z3 = twosum(z1, z2, &z4);

	return { z3, z4 };
#endif
}

/**
 * @ingroup DD
 * @brief Fast addition of two double-double numbers with weaker error guarantees.
 * @details Adds two double-double numbers and produces a double-double result.
 *
 * Implementation based on @cite Dekker1971, Section 8, function add2.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count (default ISA)</th><th>Count (with ADDRE)</th></tr>
 *         <tr><td>FP ADD  </td> <td>11</td> <td>4</td> </tr>
 *         <tr><td>FP ADDRE</td> <td>  </td> <td>2</td> </tr>
 *     </table>
 *
 * @param[in] a - addend, the first double-double number to be added.
 * @param[in] b - augend, the second double-double number to be added.
 * @return The sum of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddadd_fast(const doubledouble a, const doubledouble b)
{
	doubledouble sum = ddaddl(a.hi, b.hi);
#ifdef __CUDA_ARCH__
	sum.lo = __dadd_rn(sum.lo, __dadd_rn(a.lo, b.lo));
#else
	sum.lo += a.lo + b.lo;
#endif
	sum.hi = twosumfast(sum.hi, sum.lo, &sum.lo);
	return sum;
}

/**
 * @ingroup DD
 * @brief Long multiplication of double-precision numbers.
 * @details Multiplies two double-precision numbers and produces double-double result.
 * The algorith is a version of error-free multiplication.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count (default ISA)</th></tr>
 *         <tr><td>FP MUL</td><td>1</td></tr>
 *         <tr><td>FP FMA</td><td>1</td></tr>
 *     </table>
 *
 * @param[in] a - multiplicand, the double-precision number to be multiplied.
 * @param[in] b - multiplier, the double-precision number to multipliy by.
 * @return The product of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddmull(const double a, const double b)
{
	doubledouble product;
	product.hi = efmul(a, b, &product.lo);
	return product;
}

/**
 * @ingroup DD
 * @brief Multiplication of double-double numbers.
 * @details Multiplies two double-double numbers and produces double-double result.
 *
 * Implementation mostly follows @cite Dekker1971, Section 8, function mul2.
 *
 * @par	Computational complexity
 *     <table>
 *         <tr><th>Operation</th><th>Count (default ISA)</th></tr>
 *         <tr><td>FP ADD</td><td>3</td></tr>
 *         <tr><td>FP MUL</td><td>1</td></tr>
 *         <tr><td>FP FMA</td><td>3</td></tr>
 *     </table>
 *
 * @param[in] a - multiplicand, the double-double number to be multiplied.
 * @param[in] b - multiplier, the double-double number to multipliy by.
 * @return The product of @b a and @b b as a double-double number.
 */
FPPLUS_STATIC_INLINE doubledouble ddmul(const doubledouble a, const doubledouble b)
{
	doubledouble product = ddmull(a.hi, b.hi);

	/*
	 * Dekker's paper used product.lo += (a.lo * b.hi) + (a.hi * b.lo) here,
	 * but FMA-based implementation should be slightly faster and more accurate
	 */
#if defined(__CUDA_ARCH__)
	product.lo = __fma_rn(a.lo, b.hi, product.lo);
	product.lo = __fma_rn(a.hi, b.lo, product.lo);
#elif defined(__GNUC__)
	product.lo = __builtin_fma(a.lo, b.hi, product.lo);
	product.lo = __builtin_fma(a.hi, b.lo, product.lo);
#else
#if 0
	 //product.lo += a.lo * b.hi + a.hi * b.lo;//fast vaer
	product.lo = fma(a.lo, b.hi, product.lo);
	product.lo = fma(a.hi, b.lo, product.lo);
#else
	 //product.lo += a.lo * b.hi + a.hi * b.lo  + a.lo * b.lo; //accurate ver
	product.lo = fma(a.lo, b.lo, product.lo);
	product.lo = fma(a.lo, b.hi, product.lo);
	product.lo = fma(a.hi, b.lo, product.lo);

#endif

#endif

	//product.hi = twosumfast(product.hi, product.lo, &product.lo);
	product.hi = twosum(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE doubledouble dddiv(const doubledouble a, const double b)
{
	const double z1 = a.hi / b;
	double z4;
	double z3 = efmul(-z1, b, &z4);
	//z2 = ((((z3 + a.hi) - z1 * b.lo) + a.lo) + z4) / b.hi;
	const double z2 = (((a.hi + z3) + a.lo) + z4) / b;

	//z3 = twosumfast(z1, z2, &z4);
	z3 = twosum(z1, z2, &z4);
	return { z3, z4 };
}

FPPLUS_STATIC_INLINE doubledouble dddiv(const doubledouble a, const doubledouble b)
{
	const double z1 = a.hi / b.hi;
	double z4;
	double z3 = efmul(-z1, b.hi, &z4);
	//z2 = ((((z3 + a.hi) - z1 * b.lo) + a.lo) + z4) / b.hi;
	//const double z2 = ((((a.hi + z3) - z1 * b.lo) + a.lo) + z4) / b.hi;
	const double z2 = ((fma(-z1, b.lo, (a.hi + z3)) + a.lo) + z4) / b.hi;

	//z3 = twosumfast(z1, z2, &z4);
	z3 = twosum(z1, z2, &z4);
	return { z3, z4 };
}

FPPLUS_STATIC_INLINE doubledouble ddsqrt(const doubledouble a)
{
	double z1, z2, z3, z4;

	z1 = std::sqrt(a.lo);
	z4 = efmul(-z1, z1, &z3);

	z2 = ((z3 + a.lo) + a.hi + z4) / (2.0 * z1);
	z4 = twosum(z1, z2, &z3);

	return { z3, z4 };
}

FPPLUS_STATIC_INLINE doubledouble round(const doubledouble x)
{
	double z1, z2, z3, z4;
	z1 = std::round(x.hi);
	if (z1 != x.hi)
	{
		return doubledouble{ z1, 0.0 };
	}
	else
	{
		z2 = std::round(x.lo);
		z3 = twosum(z1, z2, &z4);
		return doubledouble{ z3, z4 };
	}
}

FPPLUS_STATIC_INLINE doubledouble floor(const doubledouble x)
{
	double z1, z2, z3, z4;
	z1 = std::floor(x.hi);
	if (z1 != x.hi)
	{
		return doubledouble{ z1, 0.0 };
	}
	else
	{
		z2 = std::floor(x.lo);
		z3 = twosum(z1, z2, &z4);
		return doubledouble{ z3, z4 };
	}
}

FPPLUS_STATIC_INLINE doubledouble ddminus(const doubledouble x)
{
	return ddmul({ -1.0, 0.0 }, x);
}

FPPLUS_STATIC_INLINE doubledouble ipower(const doubledouble x, double i)
{
	if (i != i) return { i, 0.0 }; // NaN check

	doubledouble xp = x;
	doubledouble r = { 1.0, 0.0 };

	while (i != 0.0)
	{
		i *= 0.5;
		const double tmp = std::floor(i);
		if (tmp != i)
		{
			i = tmp;
			r = ddmul(r, xp);
		}
		xp = ddmul(xp, xp);
	}

	return r;
}


FPPLUS_STATIC_INLINE doubledouble ddexp(const doubledouble x)
{
	if (x.hi == 0.0 && x.lo == 0.0)return { 1.0, 0.0 };

	//return { exp(x.hi),0.0 };

	doubledouble x_i, x_f;

#if 0
	doubledouble log2e;
	stringtodd("1.44269504088896340735992468100189213742664595415300", log2e.hi, log2e.lo);
	doubledouble ln2;//log_e(2)
	stringtodd("0.69314718055994530941723212145817656807550013436026", ln2.hi, ln2.lo);

	doubledouble d = ddmul(x, log2e);

	if (x.hi >= 0.0)
	{
		x_i = round(d);
		x_f = ddsub(d, x_i);
	}
	else
	{
		x_i = ddminus(round(ddminus(d)));
		x_f = ddsub(d, x_i);
	}

	x_f = ddmul(x_f, ln2);
	doubledouble r = { 1.0, 0.0 };
	doubledouble y = { 1.0, 0.0 };
	//Taylor
	for (int i = 1; i <= 25; i++)
	{
		y = ddmul(y, x_f);
		y = dddiv(y, (double)i);
		r = ddadd(r, y);
	}
	double w = double(1 << (int)x_i.hi);
	return ddmul(r, { w,0.0 });

#else

	//#define M_E   2.71828182845904523536   // e

	if (x.hi >= 0.0)
	{
		//std::cout << "+" << std::endl;
		x_i = floor(x);
		x_f = ddsub(x, x_i);
		if (x_f.hi >= 0.5)
		{
			x_f = ddsub(x_f, { 1.0, 0.0 });
			x_i = ddadd(x_i, { 1.0, 0.0 });
		}
	}
	else
	{
		//std::cout << "-" << std::endl;
		doubledouble x_ii = floor(ddminus(x));
		x_i = { -x_ii.hi,0.0 };
		x_f = ddsub(x, x_i);
		if (x_f.hi <= -0.5)
		{
			x_f = ddadd(x_f, { 1.0, 0.0 });
			x_i = ddsub(x_i, { 1.0, 0.0 });
		}
	}

	doubledouble r = { 1.0, 0.0 };
	doubledouble y = { 1.0, 0.0 };

	//Taylor
	for (int i = 1; i <= 25; i++)
	{
		y = ddmul(y, x_f);
		y = dddiv(y, (double)i);
		r = ddadd(r, y);
	}

	if (x_i.hi >= 0.0)
	{
		//cout << "2 x_i.hi "<< x_i.hi << endl;
		// r *= pow(constants<dd>::e(), (int)x_i);
		r = ddmul(r, ipower(M_E_DD, x_i.hi));
	}
	else
	{
		// r /= pow(constants<dd>::e(), -(int)x_i);
		r = dddiv(r, ipower(M_E_DD, -x_i.hi));
	}
#endif
	return r;
}
#if defined(__AVX__) && (defined(__FMA__) || defined(__FMA4__) || defined(__AVX2__))

typedef struct {
	__m128d hi;
	__m128d lo;
} __m128dd;

FPPLUS_STATIC_INLINE __m128dd _mm_setzero_pdd(void)
{
	__m128dd ret;
	ret.hi = _mm_setzero_pd();
	ret.lo = _mm_setzero_pd();
	return ret;
}

FPPLUS_STATIC_INLINE __m128dd _mm_broadcast_sdd(
	const doubledouble FPPLUS_NONNULL_POINTER(pointer))
{
	__m128dd ret;
	ret.hi = _mm_loaddup_pd(&pointer->hi);
	ret.lo = _mm_loaddup_pd(&pointer->lo);
	return ret;
}

FPPLUS_STATIC_INLINE __m128dd _mm_loaddeinterleave_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 2))
{
	const __m128d number0 = _mm_load_pd(&pointer[0].hi);
	const __m128d number1 = _mm_load_pd(&pointer[1].hi);

	__m128dd ret;
	ret.hi = _mm_unpacklo_pd(number0, number1);
	ret.lo = _mm_unpackhi_pd(number0, number1);
	return ret;
}

FPPLUS_STATIC_INLINE __m128dd _mm_loaddeinterleaveu_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 2))
{
	const __m128d number0 = _mm_loadu_pd(&pointer[0].hi);
	const __m128d number1 = _mm_loadu_pd(&pointer[1].hi);

	__m128dd ret;
	ret.hi = _mm_unpacklo_pd(number0, number1);
	ret.lo = _mm_unpackhi_pd(number0, number1);
	return ret;
}

FPPLUS_STATIC_INLINE __m128dd _mm_addl_sd(const __m128d a, const __m128d b) {
	__m128dd sum;
	sum.hi = _mm_efadd_sd(a, b, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_addl_pd(const __m128d a, const __m128d b) {
	__m128dd sum;
	sum.hi = _mm_efadd_pd(a, b, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_addw_sdd(const __m128dd a, const __m128d b) {
	__m128dd sum = _mm_addl_sd(a.lo, b);
	__m128d e;
	sum.hi = _mm_efadd_sd(a.hi, sum.hi, &e);
	sum.lo = _mm_add_sd(sum.lo, e);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_addw_pdd(const __m128dd a, const __m128d b) {
	__m128dd sum = _mm_addl_pd(a.lo, b);
	__m128d e;
	sum.hi = _mm_efadd_pd(a.hi, sum.hi, &e);
	sum.lo = _mm_add_pd(sum.lo, e);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_add_sdd(const __m128dd a, const __m128dd b) {
	const __m128dd s = _mm_addl_sd(a.hi, b.hi);
	const __m128dd t = _mm_addl_sd(a.lo, b.lo);
	__m128dd v;
	v.hi = _mm_efaddord_sd(s.hi, _mm_add_pd(s.lo, t.hi), &v.lo);
	__m128dd z;
	z.hi = _mm_efaddord_sd(v.hi, _mm_add_pd(t.lo, v.lo), &z.lo);
	return z;
}

FPPLUS_STATIC_INLINE __m128dd _mm_add_pdd(const __m128dd a, const __m128dd b) {
	const __m128dd s = _mm_addl_pd(a.hi, b.hi);
	const __m128dd t = _mm_addl_pd(a.lo, b.lo);
	__m128dd v;
	v.hi = _mm_efaddord_pd(s.hi, _mm_add_pd(s.lo, t.hi), &v.lo);
	__m128dd z;
	z.hi = _mm_efaddord_pd(v.hi, _mm_add_pd(t.lo, v.lo), &z.lo);
	return z;
}

FPPLUS_STATIC_INLINE __m128dd _mm_add_fast_sdd(const __m128dd a, const __m128dd b) {
	__m128dd sum = _mm_addl_sd(a.hi, b.hi);
	sum.lo = _mm_add_pd(_mm_add_pd(a.lo, b.lo), sum.lo);
	sum.hi = _mm_efaddord_sd(sum.hi, sum.lo, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_add_fast_pdd(const __m128dd a, const __m128dd b) {
	__m128dd sum = _mm_addl_pd(a.hi, b.hi);
	sum.lo = _mm_add_pd(_mm_add_pd(a.lo, b.lo), sum.lo);
	sum.hi = _mm_efaddord_pd(sum.hi, sum.lo, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m128dd _mm_mull_sd(const __m128d a, const __m128d b) {
	__m128dd product;
	product.hi = _mm_efmul_sd(a, b, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m128dd _mm_mull_pd(const __m128d a, const __m128d b) {
	__m128dd product;
	product.hi = _mm_efmul_pd(a, b, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m128dd _mm_mul_sdd(const __m128dd a, const __m128dd b) {
	__m128dd product = _mm_mull_sd(a.hi, b.hi);
#if defined(__FMA__) || defined(__AVX2__)
	product.lo = _mm_fmadd_sd(a.lo, b.hi, product.lo);
	product.lo = _mm_fmadd_sd(a.hi, b.lo, product.lo);
#else
	product.lo = _mm_macc_sd(a.lo, b.hi, product.lo);
	product.lo = _mm_macc_sd(a.hi, b.lo, product.lo);
#endif
	product.hi = _mm_efaddord_sd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m128dd _mm_mul_pdd(const __m128dd a, const __m128dd b) {
	__m128dd product = _mm_mull_pd(a.hi, b.hi);
#if defined(__FMA__) || defined(__AVX2__)
	product.lo = _mm_fmadd_pd(a.lo, b.hi, product.lo);
	product.lo = _mm_fmadd_pd(a.hi, b.lo, product.lo);
#else
	product.lo = _mm_macc_pd(a.lo, b.hi, product.lo);
	product.lo = _mm_macc_pd(a.hi, b.lo, product.lo);
#endif
	product.hi = _mm_efaddord_pd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE doubledouble _mm_cvtsdd_f64dd(const __m128dd x)
{
	doubledouble ret;
	ret.hi = _mm_cvtsd_f64(x.hi);
	ret.lo = _mm_cvtsd_f64(x.lo);
	return ret;
}

FPPLUS_STATIC_INLINE doubledouble _mm_reduce_add_pdd(const __m128dd x) {
	const __m128dd x1 = {
		_mm_unpackhi_pd(x.hi, x.hi),
		_mm_unpackhi_pd(x.lo, x.lo)
	};
	return _mm_cvtsdd_f64dd(_mm_add_sdd(x, x1));
}

typedef struct {
	__m256d hi;
	__m256d lo;
} __m256dd;

FPPLUS_STATIC_INLINE __m256dd _mm256_setzero_pdd(void)
{
	__m256dd ret;
	ret.hi = _mm256_setzero_pd();
	ret.lo = _mm256_setzero_pd();
	return ret;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_set1_pdd(const doubledouble val)
{
	__m256dd ret;
	ret.hi = _mm256_set1_pd(val.hi);
	ret.lo = _mm256_set1_pd(val.lo);
	return ret;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_broadcast_sdd(
	const doubledouble FPPLUS_NONNULL_POINTER(pointer))
{
	__m256dd ret;
	ret.hi = _mm256_broadcast_sd(&pointer->hi);
	ret.lo = _mm256_broadcast_sd(&pointer->lo);
	return ret;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_loaddeinterleave_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 4))
{
	const __m256d numbers01 = _mm256_loadu_pd(&pointer[0].hi);
	const __m256d numbers23 = _mm256_loadu_pd(&pointer[2].hi);
	const __m256d numbers12 = _mm256_permute2f128_pd(numbers01, numbers23, 0x21);
	const __m256d numbers02 = _mm256_blend_pd(numbers01, numbers12, 0xC);
	const __m256d numbers13 = _mm256_blend_pd(numbers23, numbers12, 0x3);

	__m256dd ret;
	ret.hi = _mm256_unpacklo_pd(numbers02, numbers13);
	ret.lo = _mm256_unpackhi_pd(numbers02, numbers13);
	return ret;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_loaddeinterleaveu_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 4))
{
	const __m256d numbers01 = _mm256_loadu_pd(&pointer[0].hi);
	const __m256d numbers23 = _mm256_loadu_pd(&pointer[2].hi);
	const __m256d numbers12 = _mm256_permute2f128_pd(numbers01, numbers23, 0x21);
	const __m256d numbers02 = _mm256_blend_pd(numbers01, numbers12, 0xC);
	const __m256d numbers13 = _mm256_blend_pd(numbers23, numbers12, 0x3);

	__m256dd ret;
	ret.hi = _mm256_unpacklo_pd(numbers02, numbers13);
	ret.lo = _mm256_unpackhi_pd(numbers02, numbers13);
	return ret;
}

FPPLUS_STATIC_INLINE void _mm256_interleavestore_pdd(
	doubledouble FPPLUS_ARRAY_POINTER(pointer, 4),
	__m256dd numbers)
{
	const __m256d numbers02 = _mm256_unpacklo_pd(numbers.lo, numbers.hi);
	const __m256d numbers13 = _mm256_unpackhi_pd(numbers.lo, numbers.hi);
	const __m256d numbers21 = _mm256_permute2f128_pd(numbers02, numbers13, 0x21);
	const __m256d numbers01 = _mm256_blend_pd(numbers02, numbers21, 0xC);
	const __m256d numbers23 = _mm256_blend_pd(numbers13, numbers21, 0x3);
	_mm256_store_pd(&pointer[0].hi, numbers01);
	_mm256_store_pd(&pointer[2].hi, numbers23);
}

FPPLUS_STATIC_INLINE void _mm256_interleavestoreu_pdd(
	doubledouble FPPLUS_ARRAY_POINTER(pointer, 4),
	__m256dd numbers)
{
	const __m256d numbers02 = _mm256_unpacklo_pd(numbers.lo, numbers.hi);
	const __m256d numbers13 = _mm256_unpackhi_pd(numbers.lo, numbers.hi);
	const __m256d numbers21 = _mm256_permute2f128_pd(numbers02, numbers13, 0x21);
	const __m256d numbers01 = _mm256_blend_pd(numbers02, numbers21, 0xC);
	const __m256d numbers23 = _mm256_blend_pd(numbers13, numbers21, 0x3);
	_mm256_storeu_pd(&pointer[0].hi, numbers01);
	_mm256_storeu_pd(&pointer[2].hi, numbers23);
}

FPPLUS_STATIC_INLINE __m256dd _mm256_addl_pd(const __m256d a, const __m256d b)
{
	__m256dd sum;
	sum.hi = _mm256_twosum_pd(a, b, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_subl_pd(const __m256d a, const __m256d b)
{
	__m256dd sum;
	sum.hi = _mm256_twonsum_pd(a, b, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_addw_pdd(const __m256dd a, const __m256d b)
{
	__m256dd sum = _mm256_addl_pd(a.lo, b);
	__m256d e;
	sum.hi = _mm256_twosum_pd(a.hi, sum.hi, &e);
	sum.lo = _mm256_add_pd(sum.lo, e);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_addw_pdd(const __m256d a, const __m256dd b)
{
	__m256dd sum = _mm256_addl_pd(a, b.lo);
	__m256d e;
	sum.hi = _mm256_twosum_pd(b.hi, sum.hi, &e);
	sum.lo = _mm256_add_pd(sum.lo, e);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_subw_pdd(const __m256dd a, const __m256d b)
{
	__m256dd sum = _mm256_subl_pd(a.lo, b);
	__m256d e;
	sum.hi = _mm256_twosum_pd(a.hi, sum.hi, &e);
	sum.lo = _mm256_add_pd(sum.lo, e);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_add_pdd(const __m256dd a, const __m256dd b)
{
	const __m256dd s = _mm256_addl_pd(a.hi, b.hi);
	const __m256dd t = _mm256_addl_pd(a.lo, b.lo);
	__m256dd v;
	v.hi = _mm256_twosumfast_pd(s.hi, _mm256_add_pd(s.lo, t.hi), &v.lo);
	__m256dd z;
	z.hi = _mm256_twosumfast_pd(v.hi, _mm256_add_pd(t.lo, v.lo), &z.lo);
	return z;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_add_fast_pdd(const __m256dd a, const __m256dd b)
{
	__m256dd sum = _mm256_addl_pd(a.hi, b.hi);
	sum.lo = _mm256_add_pd(_mm256_add_pd(a.lo, b.lo), sum.lo);
	sum.hi = _mm256_twosumfast_pd(sum.hi, sum.lo, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_mull_pd(const __m256d a, const __m256d b)
{
	__m256dd product;
	product.hi = _mm256_twoproduct_pd(a, b, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_mulw_pdd(const __m256dd a, const __m256d b)
{
	__m256dd product;
	product.hi = _mm256_twoproduct_pd(a.hi, b, &product.lo);
	product.lo = _mm256_fmadd_pd(a.lo, b, product.lo);

	//product.hi = _mm256_twosumfast_pd(product.hi, product.lo, &product.lo);
	product.hi = _mm256_twosum_pd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_mulw_pdd(const __m256d a, const __m256dd b)
{
	__m256dd product;
	product.hi = _mm256_twoproduct_pd(a, b.hi, &product.lo);
	product.lo = _mm256_fmadd_pd(a, b.lo, product.lo);

	//product.hi = _mm256_twosumfast_pd(product.hi, product.lo, &product.lo);
	product.hi = _mm256_twosum_pd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_mul_pdd(const __m256dd a, const __m256dd b)
{
	__m256dd product = _mm256_mull_pd(a.hi, b.hi);
#if defined(__FMA__) || defined(__AVX2__)
	product.lo = _mm256_fmadd_pd(a.lo, b.lo, product.lo);
	product.lo = _mm256_fmadd_pd(a.lo, b.hi, product.lo);
	product.lo = _mm256_fmadd_pd(a.hi, b.lo, product.lo);
#else
	product.lo = _mm256_macc_pd(a.lo, b.hi, product.lo);
	product.lo = _mm256_macc_pd(a.hi, b.lo, product.lo);
#endif
	//product.hi = _mm256_twosumfast_pd(product.hi, product.lo, &product.lo);
	product.hi = _mm256_twosum_pd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m256dd _mm256_divw_pdd(const __m256d a, const __m256d b)
{
	__m256d mz1 = _mm256_div_pd(a, b);
	__m256d mz4;
	__m256d mz3 = _mm256_twoproduct_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), mz1), b, &mz4);
	__m256d mz2 = _mm256_div_pd(_mm256_add_pd(_mm256_add_pd(a, mz3), mz4), b);
	//return __m256dd{ mz1, mz2 };
	mz3 = _mm256_twosum_pd(mz1, mz2, &mz4);
	return __m256dd{ mz3, mz4 };
}

FPPLUS_STATIC_INLINE __m256dd _mm256_div_pdd(const __m256dd a, const __m256dd b)
{
	__m256d mz1 = _mm256_div_pd(a.hi, b.hi);
	__m256d mz4;
	__m256d mz3 = _mm256_twoproduct_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), mz1), b.hi, &mz4);
	__m256d mz2 = _mm256_div_pd(_mm256_add_pd(_mm256_add_pd(_mm256_fnmadd_pd(mz1, b.lo, _mm256_add_pd(a.hi, mz3)), a.lo), mz4), b.hi);
	//return __m256dd{ mz1, mz2 };
	mz3 = _mm256_twosum_pd(mz1, mz2, &mz4);
	return __m256dd{ mz3, mz4 };
}

FPPLUS_STATIC_INLINE __m256dd _mm256_divw_pdd(const __m256d a, const __m256dd b)
{
	__m256d mz1 = _mm256_div_pd(a, b.hi);
	__m256d mz4;
	__m256d mz3 = _mm256_twoproduct_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), mz1), b.hi, &mz4);
	__m256d mz2 = _mm256_div_pd(_mm256_add_pd(_mm256_fnmadd_pd(mz1, b.lo, _mm256_add_pd(a, mz3)), mz4), b.hi);
	//return __m256dd{ mz1, mz2 };
	mz3 = _mm256_twosum_pd(mz1, mz2, &mz4);
	return __m256dd{ mz3, mz4 };
}

FPPLUS_STATIC_INLINE doubledouble _mm256_reduce_add_pdd(const __m256dd x)
{
	const __m128dd x01 =
	{
		_mm256_castpd256_pd128(x.hi),
		_mm256_castpd256_pd128(x.lo)
	};
	const __m128dd x23 =
	{
		_mm256_extractf128_pd(x.hi, 1),
		_mm256_extractf128_pd(x.lo, 1)
	};
	return _mm_reduce_add_pdd(_mm_add_pdd(x01, x23));
}

#endif /* AVX */

#undef __AVX512F__
#if defined(__AVX512F__) || defined(__KNC__)

typedef struct {
	__m512d hi;
	__m512d lo;
} __m512dd;

FPPLUS_STATIC_INLINE __m512dd _mm512_setzero_pdd(void) {
	return (__m512dd) { _mm512_setzero_pd(), _mm512_setzero_pd() };
}

FPPLUS_STATIC_INLINE __m512dd _mm512_broadcast_sdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 8))
{
	return (__m512dd) {
		_mm512_extload_pd(&pointer->hi, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE),
			_mm512_extload_pd(&pointer->lo, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE),
	};
}

FPPLUS_STATIC_INLINE __m512dd _mm512_loaddeinterleave_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 8))
{
	const __m512d numbers0123 = _mm512_load_pd(&pointer[0].hi);
	const __m512d numbers4567 = _mm512_load_pd(&pointer[4].hi);
	const __mmask16 mask_lo = _mm512_int2mask(0xAAAA);
	const __mmask16 mask_hi = _mm512_knot(mask_hi);
	const __m512d hi04152637 = _mm512_mask_swizzle_pd(numbers0123, mask_lo, numbers4567, _MM_SWIZ_REG_CDAB);
	const __m512d lo04152637 = _mm512_mask_swizzle_pd(numbers4567, mask_hi, numbers0123, _MM_SWIZ_REG_CDAB);
	const __m512i mask_shuffle = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
	const __m512d hi01234567 = _mm512_castsi512_pd(_mm512_permutevar_epi32(mask_shuffle, _mm512_castpd_si512(hi04152637)));
	const __m512d lo01234567 = _mm512_castsi512_pd(_mm512_permutevar_epi32(mask_shuffle, _mm512_castpd_si512(lo04152637)));
	return (__m512dd) { hi01234567, lo01234567 };
}

FPPLUS_STATIC_INLINE __m512dd _mm512_loaddeinterleaveu_pdd(
	const doubledouble FPPLUS_ARRAY_POINTER(pointer, 8))
{
	const __m512i index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112, 0, 16, 32, 48, 64, 80, 96, 112);
	return (__m512dd) {
		_mm512_i32loextgather_pd(index, &pointer->hi, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE),
			_mm512_i32loextgather_pd(index, &pointer->lo, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE)
	};
}

FPPLUS_STATIC_INLINE void _mm512_interleavestore_pdd(
	doubledouble FPPLUS_ARRAY_POINTER(pointer, 8),
	__m512dd numbers)
{
	const __m512i mask_shuffle = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
	const __m512d hi04152637 = _mm512_castsi512_pd(_mm512_permutevar_epi32(mask_shuffle, _mm512_castpd_si512(numbers.hi)));
	const __m512d lo04152637 = _mm512_castsi512_pd(_mm512_permutevar_epi32(mask_shuffle, _mm512_castpd_si512(numbers.lo)));

	const __mmask16 mask_lo = _mm512_int2mask(0xAAAA);
	const __mmask16 mask_hi = _mm512_knot(mask_hi);
	const __m512d numbers0123 = _mm512_mask_swizzle_pd(hi04152637, mask_lo, lo04152637, _MM_SWIZ_REG_CDAB);
	const __m512d numbers4567 = _mm512_mask_swizzle_pd(lo04152637, mask_hi, hi04152637, _MM_SWIZ_REG_CDAB);
	_mm512_store_pd(&pointer[0].hi, numbers0123);
	_mm512_store_pd(&pointer[4].hi, numbers4567);
}

FPPLUS_STATIC_INLINE void _mm512_interleavestoreu_pdd(
	doubledouble FPPLUS_ARRAY_POINTER(pointer, 8),
	__m512dd numbers)
{
	const __m512i index = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112, 0, 16, 32, 48, 64, 80, 96, 112);
	_mm512_i32loextscatter_pd(&pointer->hi, index, numbers.hi, _MM_DOWNCONV_PD_NONE, 1, _MM_HINT_NONE);
	_mm512_i32loextscatter_pd(&pointer->lo, index, numbers.lo, _MM_DOWNCONV_PD_NONE, 1, _MM_HINT_NONE);
}

FPPLUS_STATIC_INLINE __m512dd _mm512_addl_pd(const __m512d a, const __m512d b) {
	__m512dd sum;
	sum.hi = _mm512_efadd_pd(a, b, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m512dd _mm512_addw_pdd(const __m512dd a, const __m512d b) {
	__m512dd sum = _mm512_addl_pd(a.lo, b);
	__m512d e;
	sum.hi = _mm512_efadd_pd(a.hi, sum.hi, &e);
	sum.lo = _mm512_add_round_pd(sum.lo, e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	return sum;
}

FPPLUS_STATIC_INLINE __m512dd _mm512_add_pdd(const __m512dd a, const __m512dd b) {
	const __m512dd s = _mm512_addl_pd(a.hi, b.hi);
	const __m512dd t = _mm512_addl_pd(a.lo, b.lo);
	__m512dd v;
	v.hi = _mm512_efaddord_pd(s.hi, _mm512_add_round_pd(s.lo, t.hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), &v.lo);
	__m512dd z;
	z.hi = _mm512_efaddord_pd(v.hi, _mm512_add_round_pd(t.lo, v.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), &z.lo);
	return z;
}

FPPLUS_STATIC_INLINE __m512dd _mm512_add_fast_pdd(const __m512dd a, const __m512dd b) {
	__m512dd sum = _mm512_addl_pd(a.hi, b.hi);
	sum.lo = _mm512_add_round_pd(sum.lo, _mm512_add_round_pd(a.lo, b.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	sum.hi = _mm512_efaddord_pd(sum.hi, sum.lo, &sum.lo);
	return sum;
}

FPPLUS_STATIC_INLINE __m512dd _mm512_mull_pd(const __m512d a, const __m512d b) {
	__m512dd product;
	product.hi = _mm512_efmul_pd(a, b, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE __m512dd _mm512_mul_pdd(const __m512dd a, const __m512dd b) {
	__m512dd product = _mm512_mull_pd(a.hi, b.hi);
	product.lo = _mm512_fmadd_round_pd(a.lo, b.hi, product.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	product.lo = _mm512_fmadd_round_pd(a.hi, b.lo, product.lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	product.hi = _mm512_efaddord_pd(product.hi, product.lo, &product.lo);
	return product;
}

FPPLUS_STATIC_INLINE doubledouble _mm512_reduce_add_pdd(const __m512dd x) {
	const __m512dd x01234567 = x;
	const __m512dd x45670123 = {
		_mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(x01234567.hi), _MM_PERM_BADC)),
		_mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(x01234567.lo), _MM_PERM_BADC))
	};

	const __m512dd y0123 = _mm512_add_pdd(x01234567, x45670123);
	const __m512dd y2301 = {
		_mm512_swizzle_pd(y0123.hi, _MM_SWIZ_REG_BADC),
		_mm512_swizzle_pd(y0123.lo, _MM_SWIZ_REG_BADC)
	};


	const __m512dd z01 = _mm512_add_pdd(y0123, y2301);
	const __m512dd z10 = {
		_mm512_swizzle_pd(z01.hi, _MM_SWIZ_REG_CDAB),
		_mm512_swizzle_pd(z01.lo, _MM_SWIZ_REG_CDAB),
	};

	const __m512dd r = _mm512_add_pdd(z01, z10);

	union {
		__m512d as_vector;
		double as_scalar;
	} hi, lo;

	hi.as_vector = r.hi;
	lo.as_vector = r.lo;
	return (doubledouble) { hi.as_scalar, lo.as_scalar };
}

#endif /* Intel KNC or AVX-512 */

static inline void _mm256_addkahan_pdd(const __m256d a, __m256dd& b)
{
	__m256d y = _mm256_sub_pd(a, b.lo);
	__m256d t = _mm256_add_pd(b.hi, y);
	b.lo = _mm256_sub_pd(_mm256_sub_pd(t, b.hi), y);
	b.hi = t;
}

static inline __m256dd _mm256_addkahan_pdd(const __m256d a, const __m256d bhi, const __m256d blo)
{
	__m256d y = _mm256_sub_pd(a, blo);
	__m256d t = _mm256_add_pd(bhi, y);
	return { t, _mm256_sub_pd(_mm256_sub_pd(t, bhi), y) };
}

static inline void _mm256_fmakahan_pdd(const __m256d a, const __m256d x, __m256dd& b)
{
	__m256d y = _mm256_fmsub_pd(a, x, b.lo);
	__m256d t = _mm256_add_pd(b.hi, y);
	b.lo = _mm256_sub_pd(_mm256_sub_pd(t, b.hi), y);
	b.hi = t;
}

static inline __m256dd _mm256_fmakahan_pdd(const __m256d a, const __m256d x, const __m256d bhi, const __m256d blo)
{
	__m256d y = _mm256_fmsub_pd(a, x, blo);
	__m256d t = _mm256_add_pd(bhi, y);
	return { t,_mm256_sub_pd(_mm256_sub_pd(t, bhi), y) };
}

#endif /* FPPLUS_DD_H */
