#include "general_utils.hpp"


namespace mcrt {

	GeneralUtils::GeneralUtils(){}

	int GeneralUtils::pow2roundup(int x)
	{
    // ==========================================================================================================================
    // https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
    // ==========================================================================================================================
        if (x < 0)
            return 0;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
	}
}