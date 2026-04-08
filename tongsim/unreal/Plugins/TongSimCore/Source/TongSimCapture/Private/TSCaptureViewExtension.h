#pragma once

#include "CoreMinimal.h"
#include "SceneViewExtension.h"
#include "Templates/SharedPointer.h"

struct FScreenPassTexture;
struct FPostProcessMaterialInputs;

class UTSCaptureSubsystem;
class FTSCaptureDepthComputeDevice;

// Minimal stub for future RDG integration.
class FTSCaptureViewExtension : public FSceneViewExtensionBase
{
public:
	FTSCaptureViewExtension(const FAutoRegister& AutoRegister, UTSCaptureSubsystem* InOwner)
		: FSceneViewExtensionBase(AutoRegister)
		  , Owner(InOwner)
	{
	}

	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override
	{
	}

	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override
	{
	}

	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override
	{
	}

	virtual void PreRenderViewFamily_RenderThread(FRDGBuilder& GraphBuilder, FSceneViewFamily& InViewFamily) override
	{
	}

	virtual void SubscribeToPostProcessingPass(EPostProcessingPass Pass, const FSceneView& InView, FPostProcessingPassDelegateArray& InOutPassCallbacks, bool bIsPassEnabled) override;

private:
	FScreenPassTexture PostProcessPassAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs);

	UTSCaptureSubsystem* Owner = nullptr;
};
