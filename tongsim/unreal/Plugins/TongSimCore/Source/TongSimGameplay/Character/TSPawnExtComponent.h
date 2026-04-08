#pragma once

#include "GameplayTagAssetInterface.h"
#include "Components/GameFrameworkInitStateInterface.h"
#include "Components/PawnComponent.h"

#include "TSPawnExtComponent.generated.h"

struct FGameplayTag;

/**
 * Component that adds functionality to all Pawn classes so it can be used for characters/vehicles/etc.
 * This coordinates the initialization of other components.
 * TODO: AbilityComponent
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSPawnExtComponent : public UPawnComponent, public IGameFrameworkInitStateInterface, public IGameplayTagAssetInterface
{
	GENERATED_BODY()

public:
	UTSPawnExtComponent(const FObjectInitializer& ObjectInitializer);

	/** The name of this overall feature, this one depends on the other named component features */
	static const FName NAME_ActorFeatureName;

	/* GameplayTag Asset Interface */
	virtual void GetOwnedGameplayTags(FGameplayTagContainer& TagContainer) const override;

	// TODO: change to ability subsystem
	/** Tags that are set on this object */
	UPROPERTY(EditAnywhere, Replicated, BlueprintReadWrite, Category = "TongSim|GameplayTags")
	FGameplayTagContainer GameplayTags;

	FGameplayTagContainer& GetGameplayTags() {return  GameplayTags;}

	/* ~GameplayTag Asset Interface */

	//~ Begin IGameFrameworkInitStateInterface interface
	virtual FName GetFeatureName() const override { return NAME_ActorFeatureName; }
	virtual bool CanChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) const override;
	virtual void HandleChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) override;
	virtual void OnActorInitStateChanged(const FActorInitStateChangedParams& Params) override;
	virtual void CheckDefaultInitialization() override;
	//~ End IGameFrameworkInitStateInterface interface

	/** Should be called by the owning pawn when the pawn's controller changes. */
	void HandleControllerChanged();

	/** Should be called by the owning pawn when the player state has been replicated. */
	void HandlePlayerStateReplicated();

	/** Should be called by the owning pawn when the input component is setup. */
	void SetupPlayerInputComponent();

protected:
	virtual void OnRegister() override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
};
