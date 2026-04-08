#include "TSCaptureDepthCompute.h"

#include "TSCaptureLinearDepthCS.h"

#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "SceneRendering.h"
#include "SceneRenderTargetParameters.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

FTSCaptureDepthComputeDevice::FTSCaptureDepthComputeDevice(EShaderPlatform InShaderPlatform, ERHIFeatureLevel::Type InFeatureLevel)
	: ShaderPlatform(InShaderPlatform)
	  , FeatureLevel(InFeatureLevel)
{
	// Defer final support check to AddDepthPass where a valid View.ShaderMap is guaranteed
	bSupported = (GetGlobalShaderMap(FeatureLevel) != nullptr);
}

bool FTSCaptureDepthComputeDevice::IsValid() const
{
	return bSupported;
}

FRDGTextureRef FTSCaptureDepthComputeDevice::AddDepthPass(
    FRDGBuilder& GraphBuilder,
    const FViewInfo& ViewInfo,
    FRDGTextureRef SceneDepthTexture,
    int32 Width,
    int32 Height,
    ETSCaptureDepthMode DepthMode,
    float NearPlane,
    float FarPlane) const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_AddDepthPass);
    // Re-evaluate shader availability on the current view's shader map to avoid stale state
    bool bHasShader = false;
    if (ViewInfo.ShaderMap)
    {
        TShaderMapRef<FTSCaptureLinearDepthCS> TestShader(ViewInfo.ShaderMap);
		bHasShader = TestShader.IsValid();
	}

	if (!bHasShader || !SceneDepthTexture)
	{
		return nullptr;
	}

	const FIntPoint OutputSize(FMath::Max(1, Width), FMath::Max(1, Height));

	FRDGTextureDesc Desc = FRDGTextureDesc::Create2D(
		OutputSize,
		PF_R32_FLOAT,
		FClearValueBinding::White,
		TexCreate_ShaderResource | TexCreate_UAV);

	FRDGTextureRef OutputTexture = GraphBuilder.CreateTexture(Desc, TEXT("TSCapture.LinearDepth"));

	auto* Parameters = GraphBuilder.AllocParameters<FTSCaptureLinearDepthCS::FParameters>();
	Parameters->View = ViewInfo.ViewUniformBuffer;
	Parameters->SceneTexturesStruct = CreateSceneTextureUniformBuffer(GraphBuilder, ViewInfo, ESceneTextureSetupMode::SceneDepth);
	Parameters->OutLinearDepth = GraphBuilder.CreateUAV(OutputTexture);
	Parameters->OutputSize = OutputSize;
	Parameters->ViewRectMin = ViewInfo.ViewRect.Min;
	Parameters->DepthMode = static_cast<uint32>(DepthMode);
	Parameters->DepthNear = NearPlane;
	Parameters->DepthFar = FarPlane;
	Parameters->InvDepthRange = (FarPlane > NearPlane) ? 1.0f / (FarPlane - NearPlane) : 0.0f;

	TShaderMapRef<FTSCaptureLinearDepthCS> ComputeShader(ViewInfo.ShaderMap);
	if (!ComputeShader.IsValid())
	{
		return nullptr;
	}

	const FIntVector GroupCount = FComputeShaderUtils::GetGroupCount(OutputSize, FIntPoint(8, 8));
	FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("TSCapture.LinearDepth"), ComputeShader, Parameters, GroupCount);

	return OutputTexture;
}
