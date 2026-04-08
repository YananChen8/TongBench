#include "TongSimCaptureModule.h"
#include "Modules/ModuleManager.h"
#include "ShaderCore.h"
#include "Interfaces/IPluginManager.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

void FTongSimCaptureModule::StartupModule()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_ModuleStartup);
    if (const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("TongSimCore")))
    {
        const FString ShaderDir = Plugin->GetBaseDir() / TEXT("Shaders");
        AddShaderSourceDirectoryMapping(TEXT("/Plugin/TongSimCapture"), ShaderDir);
    }
}

IMPLEMENT_MODULE(FTongSimCaptureModule, TongSimCapture)
