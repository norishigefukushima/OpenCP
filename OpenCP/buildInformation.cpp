#include "buildInformation.hpp"
#include "inlineSIMDFunctions.hpp"
using namespace cv;
using namespace std;
namespace cp
{
	void printBuildInformation()
	{
#ifdef _OPENMP
#ifdef _OPENMP_LLVM_RUNTIME
		std::cout << "OPENMP(llvm) support" << std::endl;
#else
		std::cout << "OPENMP support" << std::endl;
#endif
#else 
		std::cout << "OPENMP off" << std::endl;
#endif
#ifdef __AVX512F__
		std::cout << "AVX512 support" << std::endl;
#endif
#ifdef __AVX2__
		std::cout << "AVX2 support" << std::endl;
#endif
#ifdef __AVX__
		std::cout << "AVX support" << std::endl;
#endif

#ifdef USE_SET4GATHER
		std::cout << "USE_SET4GATHER on" << std::endl;
#endif
#ifdef UNUSE_FMA
		std::cout << "UNUSE_FMA on" << std::endl;
#endif
	}
}