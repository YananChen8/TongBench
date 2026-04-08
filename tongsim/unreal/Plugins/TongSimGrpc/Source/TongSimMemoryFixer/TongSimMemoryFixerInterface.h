#pragma once
#include <string>

FORCENOINLINE std::string TONGSIMMEMORYFIXER_API TongSimAllocateCString(const char* InTChar);
FORCENOINLINE void TONGSIMMEMORYFIXER_API TongSimDeallocateCString(std::string InCString);
