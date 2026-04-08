#pragma once

#include "CoreMinimal.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "SceneRenderTargetParameters.h"

class FTSCaptureLinearDepthCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FTSCaptureLinearDepthCS);
	SHADER_USE_PARAMETER_STRUCT(FTSCaptureLinearDepthCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters,)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, OutLinearDepth)
		SHADER_PARAMETER(FIntPoint, OutputSize)
		SHADER_PARAMETER(FIntPoint, ViewRectMin)
		SHADER_PARAMETER(uint32, DepthMode)
		SHADER_PARAMETER(float, DepthNear)
		SHADER_PARAMETER(float, DepthFar)
		SHADER_PARAMETER(float, InvDepthRange)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return true;
	}
};
