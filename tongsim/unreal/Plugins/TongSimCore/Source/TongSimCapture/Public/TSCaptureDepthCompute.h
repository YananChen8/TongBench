#pragma once

#include "CoreMinimal.h"
#include "RHIShaderPlatform.h"
#include "RenderGraphFwd.h"
#include "TSCaptureTypes.h"
#include "SceneRenderTargetParameters.h"

class FTSCaptureDepthComputeDevice
{
public:
	FTSCaptureDepthComputeDevice(EShaderPlatform InShaderPlatform, ERHIFeatureLevel::Type InFeatureLevel);

	bool IsValid() const;

	FRDGTextureRef AddDepthPass(
		FRDGBuilder& GraphBuilder,
		const class FViewInfo& ViewInfo,
		FRDGTextureRef SceneDepthTexture,
		int32 Width,
		int32 Height,
		ETSCaptureDepthMode DepthMode,
		float NearPlane,
		float FarPlane) const;

private:
	bool bSupported = false;
	EShaderPlatform ShaderPlatform;
	ERHIFeatureLevel::Type FeatureLevel;
};
