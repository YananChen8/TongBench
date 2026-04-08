// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/PawnComponent.h"
#include "Components/GameFrameworkInitStateInterface.h"
#include "TSInputBinderBase.generated.h"

class UTSInputComponent;
class UTSInputConfig;
class UPlayerMappableInputConfig;
class UInputMappingContext;

UCLASS(Blueprintable, meta=(BlueprintSpawnableComponent))
class TONGSIMGAMEPLAY_API UTSInputBinderBase : public UPawnComponent, public IGameFrameworkInitStateInterface
{
	GENERATED_BODY()

public:
	UTSInputBinderBase(const FObjectInitializer& ObjectInitializer);

	UFUNCTION(BlueprintCallable, Category= "TongSim|Input")
	virtual void InitializePlayerInput(UInputComponent* PlayerInputComponent);

	virtual void OnRegister() override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	/** The name of this component-implemented feature */
	static const FName NAME_InputBinderFeatureName;

	//~ Begin IGameFrameworkInitStateInterface interface
	virtual FName GetFeatureName() const override { return NAME_InputBinderFeatureName; }
	virtual bool CanChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) const override;
	virtual void HandleChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) override;
	virtual void OnActorInitStateChanged(const FActorInitStateChangedParams& Params) override;
	virtual void CheckDefaultInitialization() override;
	//~ End IGameFrameworkInitStateInterface interface

protected:
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "TongSim|Input")
	TSoftObjectPtr<UInputMappingContext> InputMapping;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "TongSim|Input")
	UTSInputConfig* InputConfig;

	virtual void BindInputEvent(UTSInputComponent* InputComponent) {PURE_VIRTUAL( UTSInputBinderBase::BindInputEvent, );}

	/** True when player input bindings have been applied, will never be true for non - players */
	bool bReadyToBindInputs = false;
};
