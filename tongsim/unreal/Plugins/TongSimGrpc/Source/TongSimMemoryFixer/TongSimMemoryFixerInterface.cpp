#include "TongSimMemoryFixerInterface.h"

std::string TongSimAllocateCString(const char* InTChar)
{
	return std::string{InTChar};
}

void TongSimDeallocateCString(std::string InCString)
{
}
