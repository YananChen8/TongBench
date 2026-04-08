#include "util/thread.h"

#ifdef __unix__
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "Windows/WindowsPlatformTLS.h"
#endif

namespace tongos
{
	uint64_t getThreadId()
	{
#ifdef __unix__
  // return gettid();
  return syscall(__NR_gettid);
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32)
		return FWindowsPlatformTLS::GetCurrentThreadId();
#endif
	}
}
