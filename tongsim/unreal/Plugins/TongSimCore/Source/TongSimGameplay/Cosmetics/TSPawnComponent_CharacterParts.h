// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Cosmetics/TSCosmeticAnimationTypes.h"
#include "TSPawnComponent_CharacterParts.generated.h"

struct FGameplayTag;
// TODO: Refactor to PawnComponent
UCLASS(meta=(BlueprintSpawnableComponent))
class TONGSIMGAMEPLAY_API UTSPawnComponent_CharacterParts : public UActorComponent
{
	GENERATED_BODY()

public:
	UTSPawnComponent_CharacterParts(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	// ~UActorComponent interface
	virtual void BeginPlay() override;
	// ~End of UActorComponent interface

	// TODO: Refactor to AddCharacterPart
	UFUNCTION(BlueprintCallable, BlueprintAuthorityOnly, Category=Cosmetics)
	void SetCurrentCharacterTypeTag(FGameplayTag NewTag);

	FORCEINLINE const FGameplayTag& GetCurrentCharacterTypeTag() const {return CurrentCharacterTypeTag;}

	void ChangeCharacterMesh();

protected:
	// If the parent actor is derived from ACharacter, returns the Mesh component, otherwise nullptr
	USkeletalMeshComponent* GetParentMeshComponent() const;

	ACharacter* GetParentCharacter() const;
private:
	// List of character parts
	// TODO: Refactor to CharacterPartList, and gather tag from all parts
	UPROPERTY(ReplicatedUsing = OnRep_CurrentCharacterTag, Transient)
	FGameplayTag CurrentCharacterTypeTag;

	UFUNCTION()
	void OnRep_CurrentCharacterTag();

	// Rules for how to pick a body style mesh for animation to play on, based on character part cosmetics tags
	UPROPERTY(EditAnywhere, Category=Cosmetics)
	FTongSimAnimBodyStyleSelectionSet BodyMeshes;
};
