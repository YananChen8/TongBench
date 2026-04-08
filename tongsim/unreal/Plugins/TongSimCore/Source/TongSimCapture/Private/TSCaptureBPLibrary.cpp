#include "TSCaptureBPLibrary.h"

#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Kismet/GameplayStatics.h"
#include "TSCaptureSubsystem.h"
#include "TSCaptureCameraActor.h"

namespace
{
    static UTSCaptureSubsystem* GetSubsystemFrom(UObject* WorldContextObject)
    {
        if (!WorldContextObject) return nullptr;
        UWorld* World = WorldContextObject->GetWorld();
        if (!World) return nullptr;
        UGameInstance* GI = World->GetGameInstance();
        return GI ? GI->GetSubsystem<UTSCaptureSubsystem>() : nullptr;
    }

    static UTSCaptureSubsystem* GetSubsystemFromActor(AActor* Actor)
    {
        if (!Actor) return nullptr;
        UWorld* World = Actor->GetWorld();
        if (!World) return nullptr;
        UGameInstance* GI = World->GetGameInstance();
        return GI ? GI->GetSubsystem<UTSCaptureSubsystem>() : nullptr;
    }
}

ATSCaptureCameraActor* UTSCaptureBPLibrary::CreateCaptureCamera(UObject* WorldContextObject, FName CaptureId, const FTransform& WorldTransform, const FTSCaptureCameraParams& Params)
{
    if (!WorldContextObject) return nullptr;
    UWorld* World = WorldContextObject->GetWorld();
    if (!World) return nullptr;
    FActorSpawnParameters SpawnParams;
    SpawnParams.Name = CaptureId.IsNone() ? NAME_None : CaptureId;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    ATSCaptureCameraActor* Actor = World->SpawnActor<ATSCaptureCameraActor>(ATSCaptureCameraActor::StaticClass(), WorldTransform, SpawnParams);
    if (Actor)
    {
        Actor->CaptureId = CaptureId.IsNone() ? Actor->GetFName() : CaptureId;
        Actor->Params = Params;
    }
    return Actor;
}

bool UTSCaptureBPLibrary::DestroyCaptureCamera(ATSCaptureCameraActor* CameraActor)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        if (SS->IsCapturing(CameraActor->CaptureId))
        {
            SS->StopCapture(CameraActor->CaptureId);
        }
    }
    return CameraActor->Destroy();
}

bool UTSCaptureBPLibrary::SetCaptureCameraPose(ATSCaptureCameraActor* CameraActor, const FTransform& WorldTransform)
{
    if (!CameraActor) return false;
    CameraActor->SetActorTransform(WorldTransform);
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        if (SS->IsCapturing(CameraActor->CaptureId))
        {
            SS->SetCaptureTransform(CameraActor->CaptureId, WorldTransform);
        }
    }
    return true;
}

bool UTSCaptureBPLibrary::UpdateCameraParams(ATSCaptureCameraActor* CameraActor, const FTSCaptureCameraParams& Params)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        if (SS->IsCapturing(CameraActor->CaptureId))
        {
            return false; // cannot modify while capturing
        }
    }
    CameraActor->Params = Params;
    return true;
}

bool UTSCaptureBPLibrary::AttachCaptureCamera(ATSCaptureCameraActor* CameraActor, AActor* ParentActor, FName SocketName, bool bKeepWorld)
{
    if (!CameraActor || !ParentActor) return false;
    USceneComponent* ParentRoot = ParentActor->GetRootComponent();
    if (!ParentRoot) return false;
    USceneComponent* Root = CameraActor->GetRootComponent();
    if (!Root) return false;
    Root->AttachToComponent(ParentRoot, FAttachmentTransformRules(bKeepWorld ? EAttachmentRule::KeepWorld : EAttachmentRule::SnapToTarget, true), SocketName);
    return true;
}

bool UTSCaptureBPLibrary::StartCapture(ATSCaptureCameraActor* CameraActor)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        if (SS->IsCapturing(CameraActor->CaptureId))
        {
            return false;
        }
        const FTSCaptureCameraParams& P = CameraActor->Params;
        const bool bStarted = SS->StartCaptureOnActor(CameraActor->CaptureId, CameraActor, P.Width, P.Height, P.FovDegrees, P.Qps, P.bEnableDepth);
        if (bStarted)
        {
            SS->SetColorCaptureSettings(CameraActor->CaptureId, (ESceneCaptureSource)P.ColorCaptureSource.GetValue(), (ETextureRenderTargetFormat)P.ColorRenderTargetFormat.GetValue(), P.bEnablePostProcess, P.bEnableTemporalAA);
            SS->SetDepthRange(CameraActor->CaptureId, P.DepthNearPlane, P.DepthFarPlane);
            SS->SetDepthMode(CameraActor->CaptureId, P.DepthMode);
            SS->SetCompression(CameraActor->CaptureId, P.RgbCodec, P.DepthCodec, P.JpegQuality);
            SS->SetCaptureTransform(CameraActor->CaptureId, CameraActor->GetActorTransform());
        }
        return bStarted;
    }
    return false;
}

bool UTSCaptureBPLibrary::StopCapture(ATSCaptureCameraActor* CameraActor)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        return SS->StopCapture(CameraActor->CaptureId);
    }
    return false;
}

bool UTSCaptureBPLibrary::GetLatestFrame(ATSCaptureCameraActor* CameraActor, FTSCaptureFrame& OutFrame)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        return SS->GetLatestFrame(CameraActor->CaptureId, OutFrame);
    }
    return false;
}

bool UTSCaptureBPLibrary::GetLatestColor(ATSCaptureCameraActor* CameraActor, int32& OutWidth, int32& OutHeight, TArray<uint8>& OutRgba8)
{
    FTSCaptureFrame Frame;
    if (!GetLatestFrame(CameraActor, Frame)) return false;
    OutWidth = Frame.Width;
    OutHeight = Frame.Height;
    OutRgba8 = MoveTemp(Frame.Rgba8);
    return (OutWidth > 0 && OutHeight > 0 && OutRgba8.Num() == OutWidth * OutHeight * 4);
}

bool UTSCaptureBPLibrary::GetLatestDepth(ATSCaptureCameraActor* CameraActor, int32& OutWidth, int32& OutHeight, TArray<float>& OutDepth)
{
    FTSCaptureFrame Frame;
    if (!GetLatestFrame(CameraActor, Frame)) return false;
    OutWidth = Frame.Width;
    OutHeight = Frame.Height;
    OutDepth = MoveTemp(Frame.DepthR32);
    return (OutWidth > 0 && OutHeight > 0 && OutDepth.Num() == OutWidth * OutHeight);
}

bool UTSCaptureBPLibrary::CaptureSnapshot(ATSCaptureCameraActor* CameraActor, FTSCaptureFrame& OutFrame, float TimeoutSeconds)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        const FTSCaptureCameraParams& P = CameraActor->Params;
        return SS->CaptureSnapshotOnActor(CameraActor->CaptureId, CameraActor, P.Width, P.Height, P.FovDegrees, P.bEnableDepth, OutFrame, TimeoutSeconds);
    }
    return false;
}

bool UTSCaptureBPLibrary::IsCapturing(ATSCaptureCameraActor* CameraActor)
{
    if (!CameraActor) return false;
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        return SS->IsCapturing(CameraActor->CaptureId);
    }
    return false;
}

bool UTSCaptureBPLibrary::GetStatus(ATSCaptureCameraActor* CameraActor, FTSCaptureStatus& OutStatus)
{
    if (!CameraActor) { OutStatus = FTSCaptureStatus(); return false; }
    if (UTSCaptureSubsystem* SS = GetSubsystemFromActor(CameraActor))
    {
        return SS->GetStatus(CameraActor->CaptureId, OutStatus);
    }
    OutStatus = FTSCaptureStatus();
    return false;
}
