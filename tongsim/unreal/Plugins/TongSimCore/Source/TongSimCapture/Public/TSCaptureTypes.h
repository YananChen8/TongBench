#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TSCaptureTypes.generated.h"

// Intrinsics for a pinhole camera (assuming square pixels unless specified)
USTRUCT(BlueprintType)
struct FTSCameraIntrinsics
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly)
	float Fx = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	float Fy = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	float Cx = 0.0f;

	UPROPERTY(BlueprintReadOnly)
	float Cy = 0.0f;
};

USTRUCT(BlueprintType)
struct FTSCaptureFrame
{
	GENERATED_BODY()

	// Not exposed to Blueprint due to unsupported type
	UPROPERTY()
	uint64 FrameId = 0;

	// Timestamp when GPU readback becomes ready (seconds)
	// Not exposed to Blueprint (double not supported); use helper if needed
	UPROPERTY()
	double GpuReadyTimestamp = 0.0;

	// Game thread time when the capture request was issued
	// Not exposed to Blueprint (double not supported)
	UPROPERTY()
	double GameTimeSeconds = 0.0;

	UPROPERTY(BlueprintReadOnly)
	int32 Width = 0;

	UPROPERTY(BlueprintReadOnly)
	int32 Height = 0;

	UPROPERTY(BlueprintReadOnly)
	FTransform Pose;

	UPROPERTY(BlueprintReadOnly)
	FTSCameraIntrinsics Intrinsics;

	// Raw RGBA8 color buffer (Width*Height*4 bytes)
	UPROPERTY()
	TArray<uint8> Rgba8;

	// Raw Depth buffer as 32-bit float (Width*Height elements)
	UPROPERTY()
	TArray<float> DepthR32;
};

UENUM(BlueprintType)
enum class ETSRgbCodec : uint8
{
	None UMETA(DisplayName="None"),
	JPEG UMETA(DisplayName="JPEG"),
};

UENUM(BlueprintType)
enum class ETSDepthCodec : uint8
{
	None UMETA(DisplayName="None"),
	EXR UMETA(DisplayName="EXR"),
};

USTRUCT(BlueprintType)
struct FTSCaptureCompressedFrame
{
	GENERATED_BODY()

	UPROPERTY()
	uint64 FrameId = 0;

	UPROPERTY()
	int32 Width = 0;

	UPROPERTY()
	int32 Height = 0;

	UPROPERTY()
	TArray<uint8> RgbJpeg;

	UPROPERTY()
	TArray<uint8> DepthExr;
};

// Depth mode selection for future extensibility
UENUM(BlueprintType)
enum class ETSCaptureDepthMode : uint8
{
    None UMETA(DisplayName = "None"),
    LinearDepth UMETA(DisplayName = "LinearDepth (R32F)"),
    DeviceZ UMETA(DisplayName = "DeviceZ (0..1)"),
    ViewSpaceZ UMETA(DisplayName = "ViewSpace Z (R32F)"),
    Normalized01 UMETA(DisplayName = "Normalized [0,1]")
};

USTRUCT(BlueprintType)
struct FTSCaptureStatus
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    bool bCapturing = false;

    UPROPERTY(BlueprintReadOnly)
    int32 QueueCount = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 CompressedQueueCount = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 Width = 0;

    UPROPERTY(BlueprintReadOnly)
    int32 Height = 0;

    UPROPERTY(BlueprintReadOnly)
    float FovDegrees = 0.f;

    UPROPERTY(BlueprintReadOnly)
    ETSCaptureDepthMode DepthMode = ETSCaptureDepthMode::None;
};

USTRUCT(BlueprintType)
struct FTSCaptureCameraParams
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    int32 Width = 640;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    int32 Height = 480;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    float FovDegrees = 90.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    float Qps = 30.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    bool bEnableDepth = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    TEnumAsByte<ESceneCaptureSource> ColorCaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    TEnumAsByte<ETextureRenderTargetFormat> ColorRenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    bool bEnablePostProcess = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    bool bEnableTemporalAA = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    float DepthNearPlane = 10.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    float DepthFarPlane = 5000.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    ETSCaptureDepthMode DepthMode = ETSCaptureDepthMode::LinearDepth;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    ETSRgbCodec RgbCodec = ETSRgbCodec::None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    ETSDepthCodec DepthCodec = ETSDepthCodec::None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture", meta=(ClampMin=1, ClampMax=100))
    int32 JpegQuality = 90;
};
