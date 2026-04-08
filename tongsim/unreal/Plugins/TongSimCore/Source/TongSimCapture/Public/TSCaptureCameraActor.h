#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TSCaptureTypes.h"
#include "TSCaptureCameraActor.generated.h"

UCLASS(BlueprintType, Blueprintable)
class TONGSIMCAPTURE_API ATSCaptureCameraActor : public AActor
{
    GENERATED_BODY()

public:
    ATSCaptureCameraActor();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    FName CaptureId = NAME_None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="TongSim|Capture")
    FTSCaptureCameraParams Params;
};
