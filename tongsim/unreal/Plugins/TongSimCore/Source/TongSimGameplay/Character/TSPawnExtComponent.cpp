#include "TSPawnExtComponent.h"

#include "TSGameplayTags.h"
#include "Components/GameFrameworkComponentManager.h"
#include "Net/UnrealNetwork.h"
#include "TSLogChannels.h"

const FName UTSPawnExtComponent::NAME_ActorFeatureName("PawnExtension");

UTSPawnExtComponent::UTSPawnExtComponent(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	PrimaryComponentTick.bStartWithTickEnabled = false;
	PrimaryComponentTick.bCanEverTick = false;

	SetIsReplicatedByDefault(true);
}

// TODO: Change to Ability subsystem
void UTSPawnExtComponent::GetOwnedGameplayTags(FGameplayTagContainer& TagContainer) const
{
	TagContainer = GameplayTags;
}

bool UTSPawnExtComponent::CanChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState) const
{
	check(Manager);

	APawn* Pawn = GetPawn<APawn>();
	if (!CurrentState.IsValid() && DesiredState == TongSimGameplayTags::InitState_Spawned)
	{
		// As long as we are on a valid pawn, we count as spawned
		if (Pawn)
		{
			return true;
		}
	}
	if (CurrentState == TongSimGameplayTags::InitState_Spawned && DesiredState == TongSimGameplayTags::InitState_DataAvailable)
	{
		const bool bHasAuthority = Pawn->HasAuthority();
		const bool bIsLocallyControlled = Pawn->IsLocallyControlled();

		if (bHasAuthority || bIsLocallyControlled)
		{
			// Check for being possessed by a controller.
			if (!GetController<AController>())
			{
				return false;
			}
		}

		return true;
	}
	else if (CurrentState == TongSimGameplayTags::InitState_DataAvailable && DesiredState == TongSimGameplayTags::InitState_GameplayReady)
	{
		// Transition to initialize if all features have their data available
		return Manager->HaveAllFeaturesReachedInitState(Pawn, TongSimGameplayTags::InitState_DataAvailable);
	}
	return false;
}

void UTSPawnExtComponent::HandleChangeInitState(UGameFrameworkComponentManager* Manager, FGameplayTag CurrentState, FGameplayTag DesiredState)
{
	UE_LOG(LogTongSimCore, Log, TEXT("%s-%s HandleChangeInitState from %s to %s"), *GetNameSafe(GetOwner()), *GetNameSafe(this), *CurrentState.ToString(), *DesiredState.ToString());
}

void UTSPawnExtComponent::OnActorInitStateChanged(const FActorInitStateChangedParams& Params)
{
	// If another feature is now in DataAvailable, see if we should transition to DataInitialized
	if (Params.FeatureName != NAME_ActorFeatureName)
	{
		if (Params.FeatureState == TongSimGameplayTags::InitState_DataAvailable)
		{
			CheckDefaultInitialization();
		}
	}
}

void UTSPawnExtComponent::CheckDefaultInitialization()
{
	// Before checking our progress, try progressing any other features we might depend on
	CheckDefaultInitializationForImplementers();

	static const TArray<FGameplayTag> StateChain = {TongSimGameplayTags::InitState_Spawned, TongSimGameplayTags::InitState_DataAvailable, TongSimGameplayTags::InitState_GameplayReady };

	// This will try to progress from spawned (which is only set in BeginPlay) through the data initialization stages until it gets to gameplay ready
	ContinueInitStateChain(StateChain);
}

void UTSPawnExtComponent::HandleControllerChanged()
{
	CheckDefaultInitialization();
}

void UTSPawnExtComponent::HandlePlayerStateReplicated()
{
	CheckDefaultInitialization();
}

void UTSPawnExtComponent::SetupPlayerInputComponent()
{
	CheckDefaultInitialization();
}

void UTSPawnExtComponent::OnRegister()
{
	Super::OnRegister();
	const APawn* Pawn = GetPawn<APawn>();
	check(Pawn);

	// Register with the init state system early, this will only work if this is a game world
	RegisterInitStateFeature();
}

void UTSPawnExtComponent::BeginPlay()
{
	Super::BeginPlay();

	// Listen for changes to all features
	BindOnActorInitStateChanged(NAME_None, FGameplayTag(), false);

	// Notifies state manager that we have spawned, then try rest of default initialization
	ensure(TryToChangeInitState(TongSimGameplayTags::InitState_Spawned));
	CheckDefaultInitialization();
}

void UTSPawnExtComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	UnregisterInitStateFeature();

	Super::EndPlay(EndPlayReason);
}

void UTSPawnExtComponent::GetLifetimeReplicatedProps(TArray< class FLifetimeProperty > & OutLifetimeProps) const
{
	DOREPLIFETIME(ThisClass, GameplayTags);
}
