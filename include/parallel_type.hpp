#pragma once

#include <string>
#include <omp.h>

const int OMP_THREADS_MAX = omp_get_max_threads();

enum ParallelTypes
{
	NAIVE,
	OMP,
	PARALLEL_FOR_,

	NumParallelTypes // num of parallelTypes. must be last element
};

inline std::string getParallelType(int parallelType)
{
	std::string type;
	switch (parallelType)
	{
	case NAIVE:			type = "NAIVE";	break;
	case OMP:			type = "OMP";	break;
	case PARALLEL_FOR_:	type = "PARALLEL_FOR_";	break;
	default:			type = "UNDEFINED_METHOD";	break;
	}

	return type;
}