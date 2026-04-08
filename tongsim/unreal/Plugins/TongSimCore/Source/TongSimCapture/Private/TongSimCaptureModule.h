#pragma once

#include "Modules/ModuleManager.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

class FTongSimCaptureModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;

	virtual void ShutdownModule() override
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_ModuleShutdown);
	}
};
