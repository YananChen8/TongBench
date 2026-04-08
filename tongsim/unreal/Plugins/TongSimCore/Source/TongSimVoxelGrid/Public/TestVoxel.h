#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TestVoxel.generated.h"

UCLASS()
class TONGSIMVOXELGRID_API ATestVoxel : public AActor
{
	GENERATED_BODY()

public:
	ATestVoxel();

protected:
	virtual void BeginPlay() override;

	UPROPERTY(EditAnywhere,BlueprintReadOnly)
	float BoxSize{100.0};

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	TObjectPtr<class UBoxComponent> BoxComponent;
};
