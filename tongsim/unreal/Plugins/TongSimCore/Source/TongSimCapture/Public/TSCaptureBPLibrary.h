#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TSCaptureTypes.h"
#include "TSCaptureBPLibrary.generated.h"

class ATSCaptureCameraActor;

UCLASS()
class TONGSIMCAPTURE_API UTSCaptureBPLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    // Camera management
    UFUNCTION(BlueprintCallable, Category="TongSim|Capture", meta=(WorldContext="WorldContextObject"))
    static ATSCaptureCameraActor* CreateCaptureCamera(UObject* WorldContextObject, FName CaptureId, const FTransform& WorldTransform, const FTSCaptureCameraParams& Params);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool DestroyCaptureCamera(ATSCaptureCameraActor* CameraActor);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool SetCaptureCameraPose(ATSCaptureCameraActor* CameraActor, const FTransform& WorldTransform);

    // Update all parameters in one call; returns false if currently capturing under this id
    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool UpdateCameraParams(ATSCaptureCameraActor* CameraActor, const FTSCaptureCameraParams& Params);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool AttachCaptureCamera(ATSCaptureCameraActor* CameraActor, AActor* ParentActor, FName SocketName, bool bKeepWorld = true);

    // Per-camera capture control
    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool StartCapture(ATSCaptureCameraActor* CameraActor);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool StopCapture(ATSCaptureCameraActor* CameraActor);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool GetLatestFrame(ATSCaptureCameraActor* CameraActor, FTSCaptureFrame& OutFrame);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool GetLatestColor(ATSCaptureCameraActor* CameraActor, int32& OutWidth, int32& OutHeight, TArray<uint8>& OutRgba8);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool GetLatestDepth(ATSCaptureCameraActor* CameraActor, int32& OutWidth, int32& OutHeight, TArray<float>& OutDepth);

    // Snapshot: if currently capturing, returns false
    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool CaptureSnapshot(ATSCaptureCameraActor* CameraActor, FTSCaptureFrame& OutFrame, float TimeoutSeconds = 0.5f);

    // Status helpers
    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool IsCapturing(ATSCaptureCameraActor* CameraActor);

    UFUNCTION(BlueprintCallable, Category="TongSim|Capture")
    static bool GetStatus(ATSCaptureCameraActor* CameraActor, FTSCaptureStatus& OutStatus);
};
