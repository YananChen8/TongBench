// Fill out your copyright notice in the Description page of Project Settings.


#include "TSInputBinderBase.h"

#include "TSGameplayTags.h"
#include "TSInputComponent.h"
#include "TSInputConfig.h"
#include "GameFramework/PlayerState.h"
#include "Player/TSPlayerControllerBase.h"
#include "PlayerMappableInputConfig.h"
#include "Character/TSPawnExtComponent.h"
#include "Components/GameFrameworkComponentDelegates.h"
#include "TSLogChannels.h"

const FName UTSInputBinderBase::NAME_InputBinderFeatureName("InputBinder");

UTSInputBinderBase::UTSInputBinderBase(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	bReadyToBindInputs = false;
	SetIsReplicatedByDefault(true);
}

void UTSInputBinderBase::InitializePlayerInput(UInputComponent* PlayerInputComponent)
{
	APawn* OwnerPawn = Cast<APawn>(GetOwner());
	const ATSPlayerControllerBase* PC = Cast<ATSPlayerControllerBase>(OwnerPawn->GetController());
	UTSInputComponent* IC = Cast<UTSInputComponent>(PlayerInputComponent);

	if (IsValid(InputConfig) && IC && PC)
	{
		PC->SetPlayerMappableInputConfig(InputMapping);
		BindInputEvent(IC);
		UE_LOG(LogTongSimCore, Log, TEXT("Initialize input %s."), *GetNameSafe(this));
		return;
	}
	UE_LOG(LogTongSimCore, Error, TEXT("Initialize input %s error."), *GetNameSafe(this));
}

void UTSInputBinderBase::OnRegister()
{
	Super::OnRegister();

	if (APawn* Pawn = Cast<APawn>(GetOwner()))
	{
		RegisterInitStateFeature();
	}
}

void UTSInputBinderBase::BeginPlay()
{
	Super::BeginPlay();
	// Listen for when the pawn extension component changes init state
	BindOnActorInitStateChanged(UTSPawnExtComponent::NAME_ActorFeatureName, FGameplayTag(), false);
	// Notifies that we are done spawning, then try the rest of initialization
	ensure(TryToChangeInitState(TongSimGameplayTags::InitState_Spawned));
	CheckDefaultInitialization();
}

void UTSInputBinderBase::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	UnregisterInitStateFeature();
	Super::EndPlay(EndPlayReason);
}

bool UTSInputBinderBase::CanChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) const
{
	check(Manager)
	APawn* Pawn = Cast<APawn>(GetOwner());
	if (!CurrentState.IsValid() && DesiredState == TongSimGameplayTags::InitState_Spawned)
	{
		// As long as we have a real pawn, let us transition
		if (Pawn)
		{
			return true;
		}
	}
	else if (CurrentState == TongSimGameplayTags::InitState_Spawned && DesiredState == TongSimGameplayTags::InitState_DataAvailable)
	{
		// If we're authority or autonomous, we need to wait for a controller with registered ownership of the player state.
		if (Pawn->GetLocalRole() != ROLE_SimulatedProxy)
		{
			AController* Controller = Pawn->GetController<AController>();

			const bool bHasControllerPairedWithPS = (Controller != nullptr) &&
				(Controller->PlayerState != nullptr) &&
				(Controller->PlayerState->GetOwner() == Controller);

			if (!bHasControllerPairedWithPS)
			{
				return false;
			}
		}

		const bool bIsLocallyControlled = Pawn->IsLocallyControlled();
		const bool bIsBot = Pawn->IsBotControlled();

		if (bIsLocallyControlled && !bIsBot)
		{
			ATSPlayerControllerBase* TongSimPC = Pawn->GetController<ATSPlayerControllerBase>();

			// The input component and local player is required when locally controlled.
			if (!Pawn->InputComponent || !TongSimPC || !TongSimPC->GetLocalPlayer())
			{
				return false;
			}
		}
		return true;
	}
	else if (CurrentState == TongSimGameplayTags::InitState_DataAvailable && DesiredState == TongSimGameplayTags::InitState_GameplayReady)
	{
		// TODO add ability initialization checks?
		return true;
	}

	return false;
}

void UTSInputBinderBase::HandleChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState)
{
	APawn* Pawn = Cast<APawn>(GetOwner());
	if (CurrentState == TongSimGameplayTags::InitState_DataAvailable && DesiredState == TongSimGameplayTags::InitState_GameplayReady)
	{
		if (!ensure(Pawn))
		{
			return;
		}

		// TODO: Ability??
		if (ATSPlayerControllerBase* TongSimPC = Pawn->GetController<ATSPlayerControllerBase>())
		{
			if (Pawn->InputComponent != nullptr)
			{
				InitializePlayerInput(Pawn->InputComponent);
			}
		}
	}
	UE_LOG(LogTongSimCore, Log, TEXT("%s-%s HandleChangeInitState from %s to %s"), *GetNameSafe(GetOwner()), *GetNameSafe(this), *CurrentState.ToString(), *DesiredState.ToString());
}

void UTSInputBinderBase::OnActorInitStateChanged(const FActorInitStateChangedParams& Params)
{
	if (Params.FeatureName == UTSPawnExtComponent::NAME_ActorFeatureName)
	{
		if (Params.FeatureState == TongSimGameplayTags::InitState_DataAvailable)
		{
			// If the extension component says all other components are initialized, try to progress to next state
			CheckDefaultInitialization();
		}
	}
}

void UTSInputBinderBase::CheckDefaultInitialization()
{
	static const TArray<FGameplayTag> StateChain = {TongSimGameplayTags::InitState_Spawned, TongSimGameplayTags::InitState_DataAvailable, TongSimGameplayTags::InitState_GameplayReady};

	// This will try to progress from spawned (which is only set in BeginPlay) through the data initialization stages until it gets to gameplay ready
	ContinueInitStateChain(StateChain);
}
