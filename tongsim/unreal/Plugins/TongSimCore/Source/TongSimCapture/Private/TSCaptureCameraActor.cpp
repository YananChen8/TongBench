#include "TSCaptureCameraActor.h"

#include "Components/SceneComponent.h"

ATSCaptureCameraActor::ATSCaptureCameraActor()
{
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("TSCameraRoot"));
    Root->SetMobility(EComponentMobility::Movable);
    RootComponent = Root;
}
