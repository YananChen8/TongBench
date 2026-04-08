// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameplayTagAssetInterface.h"
#include "ModularCharacter.h"
#include "TSPawnExtComponent.h"
#include "TSCharacterBase.generated.h"

class UTSPawnExtComponent;
/*
 * The base character pawn class used by TongSim.
 * Responsible for sending events to pawn components.
 * TODO by wukunlun: add BindAbility with PawnExt Comp
 */
UCLASS()
class TONGSIMGAMEPLAY_API ATSCharacterBase : public AModularCharacter, public IGameplayTagAssetInterface //,   IAbilitySystemInterface
{
	GENERATED_BODY()

public:
	ATSCharacterBase(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());


	/* GameplayTag Asset Interface */
	virtual void GetOwnedGameplayTags(FGameplayTagContainer& TagContainer) const override;
	FGameplayTagContainer& GetGameplayTags() const { return PawnExtComponent->GetGameplayTags(); }
	/* ~GameplayTag Asset Interface */

protected:
	virtual void PossessedBy(AController* NewController) override;
	virtual void UnPossessed() override;

	virtual void OnRep_Controller() override;
	virtual void OnRep_PlayerState() override;

	virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;

private:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TongSim|Character", Meta = (AllowPrivateAccess = "true"))
	TObjectPtr<UTSPawnExtComponent> PawnExtComponent;
};
