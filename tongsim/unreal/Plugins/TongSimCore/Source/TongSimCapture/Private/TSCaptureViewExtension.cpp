#include "TSCaptureViewExtension.h"
#include "TSCaptureSubsystem.h"

#include "RenderGraphBuilder.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "ScreenPass.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

void FTSCaptureViewExtension::SubscribeToPostProcessingPass(EPostProcessingPass Pass, const FSceneView& InView, FPostProcessingPassDelegateArray& InOutPassCallbacks, bool bIsPassEnabled)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SubscribeToPPP);
    if (!Owner || !bIsPassEnabled || !InView.bIsSceneCapture)
    {
        return;
    }

	if (Pass == EPostProcessingPass::Tonemap)
	{
		InOutPassCallbacks.Add(FAfterPassCallbackDelegate::CreateRaw(this, &FTSCaptureViewExtension::PostProcessPassAfterTonemap_RenderThread));
	}
}

FScreenPassTexture FTSCaptureViewExtension::PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_VE_AfterTonemap);
    if (Owner)
    {
        Owner->ProcessViewAfterTonemap_RenderThread(GraphBuilder, View, Inputs);
    }

    return Inputs.ReturnUntouchedSceneColorForPostProcessing(GraphBuilder);
}
