#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/EngineTypes.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TSDevCaptureActor.generated.h"

struct FTSCaptureFrame;
class UTSCaptureSubsystem;

UENUM(BlueprintType)
enum class ETSCaptureSaveMode : uint8
{
	ColorOnly,
	DepthOnly,
	ColorAndDepth
};

UENUM(BlueprintType)
enum class ETSCaptureDepthFileFormat : uint8
{
	Binary UMETA(DisplayName="Binary (.depth.bin)"),
	Exr UMETA(DisplayName="EXR (.depth.exr)"),
	BinaryAndExr UMETA(DisplayName="Binary + EXR")
};

UCLASS(BlueprintType)
class TONGSIMCAPTURE_API ATSDevCaptureActor : public AActor
{
    GENERATED_BODY()

public:
	ATSDevCaptureActor();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	bool bAutoStart = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	FName CaptureId = TEXT("DevCapture");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	int32 Width = 640;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	int32 Height = 480;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	float FovDegrees = 90.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	float Qps = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	bool bEnableDepth = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	bool bSyncTransform = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	ETSCaptureSaveMode SaveMode = ETSCaptureSaveMode::ColorAndDepth;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	FString SaveSubDir = TEXT("DevCapture");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	int32 SaveEveryNFrames = 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	int32 MaxFramesToSave = 100;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
	ETSCaptureDepthFileFormat DepthFileFormat = ETSCaptureDepthFileFormat::Exr;

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


	UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
	bool Start();

	UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
	bool Stop();

private:
    int64 SavedCount = 0;
    int64 FrameCounter = 0;
    float LastAppliedNear = -1.f;
    float LastAppliedFar = -1.f;
    UPROPERTY(Transient)
    class ATSCaptureCameraActor* ManagedCamera = nullptr;

	void SaveColorPNG(const FTSCaptureFrame& Frame, const FString& BasePath);
	void SaveDepthBIN(const FTSCaptureFrame& Frame, const FString& BasePath);
	void SaveDepthEXR(const FTSCaptureFrame& Frame, const FString& BasePath);

    UTSCaptureSubsystem* ResolveSubsystem() const;
};
