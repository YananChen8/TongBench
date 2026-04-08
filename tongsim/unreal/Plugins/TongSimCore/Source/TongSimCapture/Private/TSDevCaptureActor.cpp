#include "TSDevCaptureActor.h"
#include "TSCaptureSubsystem.h"
#include "TSCaptureTypes.h"
#include "TSCaptureBPLibrary.h"
#include "TSCaptureCameraActor.h"

#include "Components/SceneComponent.h"
#include "Engine/World.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Modules/ModuleManager.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"

ATSDevCaptureActor::ATSDevCaptureActor()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("DevCaptureRoot"));
	Root->SetMobility(EComponentMobility::Movable);
	RootComponent = Root;
}

void ATSDevCaptureActor::BeginPlay()
{
	Super::BeginPlay();
	if (bAutoStart)
	{
		Start();
	}
}

void ATSDevCaptureActor::Tick(float DeltaSeconds)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_Tick);
	Super::Tick(DeltaSeconds);

	FrameCounter++;

	if (SavedCount >= MaxFramesToSave)
	{
		return;
	}

    if (ManagedCamera)
    {
        if (bSyncTransform)
        {
            UTSCaptureBPLibrary::SetCaptureCameraPose(ManagedCamera, GetActorTransform());
        }

        FTSCaptureFrame Frame;
        if (UTSCaptureBPLibrary::GetLatestFrame(ManagedCamera, Frame))
        {
			TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_SaveFrame);
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("DevCapture[%s] consuming FrameId=%llu RgbaBytes=%d DepthCount=%d"),
			       *CaptureId.ToString(),
			       (unsigned long long)Frame.FrameId,
			       Frame.Rgba8.Num(),
			       Frame.DepthR32.Num());
			if (SaveEveryNFrames <= 1 || (Frame.FrameId % SaveEveryNFrames) == 0)
			{
				const FString BaseDir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("TSCapture"), SaveSubDir);
				IFileManager::Get().MakeDirectory(*BaseDir, true);
				const FString BaseName = FString::Printf(TEXT("%s_%llu_%dx%d"), *CaptureId.ToString(), (unsigned long long)Frame.FrameId, Frame.Width, Frame.Height);
				const FString BasePath = FPaths::Combine(BaseDir, BaseName);

				switch (SaveMode)
				{
				case ETSCaptureSaveMode::ColorOnly:
					SaveColorPNG(Frame, BasePath);
					break;
				case ETSCaptureSaveMode::DepthOnly:
					if (DepthFileFormat == ETSCaptureDepthFileFormat::Binary || DepthFileFormat == ETSCaptureDepthFileFormat::BinaryAndExr)
					{
						SaveDepthBIN(Frame, BasePath);
					}
					if (DepthFileFormat == ETSCaptureDepthFileFormat::Exr || DepthFileFormat == ETSCaptureDepthFileFormat::BinaryAndExr)
					{
						SaveDepthEXR(Frame, BasePath);
					}
					break;
				case ETSCaptureSaveMode::ColorAndDepth:
					SaveColorPNG(Frame, BasePath);
					if (DepthFileFormat == ETSCaptureDepthFileFormat::Binary || DepthFileFormat == ETSCaptureDepthFileFormat::BinaryAndExr)
					{
						SaveDepthBIN(Frame, BasePath);
					}
					if (DepthFileFormat == ETSCaptureDepthFileFormat::Exr || DepthFileFormat == ETSCaptureDepthFileFormat::BinaryAndExr)
					{
						SaveDepthEXR(Frame, BasePath);
					}
					break;
				}
				SavedCount++;
        }
    }
	}
}

bool ATSDevCaptureActor::Start()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_Start);
    if (CaptureId.IsNone())
    {
        CaptureId = FName(*FString::Printf(TEXT("Dev_%s"), *GetFName().ToString()));
    }
    FTSCaptureCameraParams Params;
    Params.Width = Width;
    Params.Height = Height;
    Params.FovDegrees = FovDegrees;
    Params.Qps = Qps;
    Params.bEnableDepth = bEnableDepth;
    Params.ColorCaptureSource = ColorCaptureSource;
    Params.ColorRenderTargetFormat = ColorRenderTargetFormat;
    Params.bEnablePostProcess = bEnablePostProcess;
    Params.bEnableTemporalAA = bEnableTemporalAA;
    Params.DepthNearPlane = DepthNearPlane;
    Params.DepthFarPlane = DepthFarPlane;
    ManagedCamera = UTSCaptureBPLibrary::CreateCaptureCamera(this, CaptureId, GetActorTransform(), Params);
    if (!ManagedCamera)
    {
        return false;
    }
    if (bSyncTransform)
    {
        UTSCaptureBPLibrary::SetCaptureCameraPose(ManagedCamera, GetActorTransform());
    }
    return UTSCaptureBPLibrary::StartCapture(ManagedCamera);
}

bool ATSDevCaptureActor::Stop()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_Stop);
    bool bOk = false;
    if (ManagedCamera)
    {
        bOk = UTSCaptureBPLibrary::StopCapture(ManagedCamera);
        UTSCaptureBPLibrary::DestroyCaptureCamera(ManagedCamera);
        ManagedCamera = nullptr;
    }
    return bOk;
}

void ATSDevCaptureActor::SaveColorPNG(const FTSCaptureFrame& Frame, const FString& BasePath)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_SavePNG);
	if (Frame.Rgba8.Num() != Frame.Width * Frame.Height * 4)
	{
		UE_LOG(LogTongSimCapture, Warning, TEXT("DevCapture[%s] skipping color save FrameId=%llu invalid buffer Num=%d expected=%d"),
		       *CaptureId.ToString(),
		       (unsigned long long)Frame.FrameId,
		       Frame.Rgba8.Num(),
		       Frame.Width * Frame.Height * 4);
		return;
	}
	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
	TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
	if (!Wrapper.IsValid()) return;
	Wrapper->SetRaw(Frame.Rgba8.GetData(), Frame.Rgba8.Num(), Frame.Width, Frame.Height, ERGBFormat::BGRA, 8);
	const TArray64<uint8>& Comp = Wrapper->GetCompressed(100);
	TArray<uint8> Out;
	Out.Append(Comp.GetData(), Comp.Num());
	FFileHelper::SaveArrayToFile(Out, *(BasePath + TEXT(".png")));
}

void ATSDevCaptureActor::SaveDepthBIN(const FTSCaptureFrame& Frame, const FString& BasePath)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_SaveDepthBIN);
	if (Frame.DepthR32.Num() != Frame.Width * Frame.Height)
	{
		UE_LOG(LogTongSimCapture, Warning, TEXT("DevCapture[%s] skipping depth BIN FrameId=%llu invalid buffer Num=%d expected=%d"),
		       *CaptureId.ToString(),
		       (unsigned long long)Frame.FrameId,
		       Frame.DepthR32.Num(),
		       Frame.Width * Frame.Height);
		return;
	}
	const uint8* Ptr = reinterpret_cast<const uint8*>(Frame.DepthR32.GetData());
	const int64 NumBytes = static_cast<int64>(Frame.DepthR32.Num()) * sizeof(float);
	TArray<uint8> Bytes;
	Bytes.Append(Ptr, NumBytes);
	const FString Path = BasePath + TEXT(".depth.bin");
	FFileHelper::SaveArrayToFile(Bytes, *Path);
}

void ATSDevCaptureActor::SaveDepthEXR(const FTSCaptureFrame& Frame, const FString& BasePath)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSDevCapture_SaveDepthEXR);
	if (Frame.DepthR32.Num() != Frame.Width * Frame.Height)
	{
		UE_LOG(LogTongSimCapture, Warning, TEXT("DevCapture[%s] skipping depth EXR FrameId=%llu invalid buffer Num=%d expected=%d"),
		       *CaptureId.ToString(),
		       (unsigned long long)Frame.FrameId,
		       Frame.DepthR32.Num(),
		       Frame.Width * Frame.Height);
		return;
	}

	IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
	TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::EXR);
	if (!Wrapper.IsValid())
	{
		return;
	}

	TArray<float> RGBA;
	RGBA.SetNumUninitialized(Frame.Width * Frame.Height * 4);
	for (int32 Index = 0; Index < Frame.Width * Frame.Height; ++Index)
	{
		const float DepthValue = Frame.DepthR32[Index];
		RGBA[Index * 4 + 0] = DepthValue;
		RGBA[Index * 4 + 1] = DepthValue;
		RGBA[Index * 4 + 2] = DepthValue;
		RGBA[Index * 4 + 3] = 1.f;
	}

	Wrapper->SetRaw(RGBA.GetData(), RGBA.Num() * sizeof(float), Frame.Width, Frame.Height, ERGBFormat::RGBAF, 32);
	const TArray64<uint8>& Comp = Wrapper->GetCompressed(0);
	TArray<uint8> Out;
	Out.Append(Comp.GetData(), Comp.Num());
	FFileHelper::SaveArrayToFile(Out, *(BasePath + TEXT(".depth.exr")));
}

UTSCaptureSubsystem* ATSDevCaptureActor::ResolveSubsystem() const
{
	if (UWorld* World = GetWorld())
	{
		if (UGameInstance* GI = World->GetGameInstance())
		{
			return GI->GetSubsystem<UTSCaptureSubsystem>();
		}
	}
	return nullptr;
}
