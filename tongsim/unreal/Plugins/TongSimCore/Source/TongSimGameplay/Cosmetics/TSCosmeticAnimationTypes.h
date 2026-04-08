#pragma once

#include "GameplayTagContainer.h"
#include "Templates/SubclassOf.h"

#include "TSCosmeticAnimationTypes.generated.h"

class UAnimInstance;
class UPhysicsAsset;
class USkeletalMesh;

USTRUCT(BlueprintType)
struct FTongSimAnimBodyStyleSelectionEntry
{
	GENERATED_BODY()

	// Layer to apply if the tag matches
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TObjectPtr<USkeletalMesh> Mesh = nullptr;

	// TODO: Array?
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	TSubclassOf<UAnimInstance> AnimLayer;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float MaxWalkSpeed = 200.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float CapsuleHalfHeight = 88.f;

	// If set, ensures this physics asset is always used
	UPROPERTY(EditAnywhere)
	TObjectPtr<UPhysicsAsset> ForcedPhysicsAsset = nullptr;

	// Cosmetic tags required (all of these must be present to be considered a match)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta=(Categories="Cosmetic"))
	FGameplayTagContainer RequiredTags;

	FORCEINLINE bool IsValid() const
	{
		return Mesh && AnimLayer;
	}
};

USTRUCT(BlueprintType)
struct FTongSimAnimBodyStyleSelectionSet
{
	GENERATED_BODY()

	// List of layer rules to apply, first one that matches will be used
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta=(TitleProperty=Mesh))
	TArray<FTongSimAnimBodyStyleSelectionEntry> MeshRules;

	// The layer to use if none of the LayerRules matches
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FTongSimAnimBodyStyleSelectionEntry DefaultBodyStyle;

	// Choose the best body style skeletal mesh given the rules
	const FTongSimAnimBodyStyleSelectionEntry& SelectBestBodyStyle(const FGameplayTagContainer& CosmeticTags) const;
};
