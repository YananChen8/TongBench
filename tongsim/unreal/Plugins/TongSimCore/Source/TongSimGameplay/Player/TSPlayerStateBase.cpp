// Fill out your copyright notice in the Description page of Project Settings.


#include "TSPlayerStateBase.h"

#include "Common/TSGameplaySettings.h"
#include "TSLogChannels.h"
#include "Net/UnrealNetwork.h"

void ATSPlayerStateBase::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);

	DOREPLIFETIME_CONDITION(ThisClass, PlayerType, COND_OwnerOnly)
	DOREPLIFETIME_CONDITION(ThisClass, OwnedAgents, COND_OwnerOnly)
}

void ATSPlayerStateBase::PreInitializeComponents()
{
	if (GetNetMode() != NM_Client)
	{
		OwnedAgents.Reset();
	}

	Super::PreInitializeComponents();
}

void ATSPlayerStateBase::SetPlayerType(ETSPlayerType NewType)
{
	PlayerType = NewType;
	// TODO:
	MaxAgentNumPerClient = GetSelfMaxAgentNumber();
}

int32 ATSPlayerStateBase::GetNumberOfAvailableAgents() const
{
	return MaxAgentNumPerClient - GetCurrentAgentNumber();
}

int32 ATSPlayerStateBase::GetCurrentAgentNumber() const
{
	int32 Ret = 0;
	for (AActor* Agent : OwnedAgents)
	{
		if (IsValid(Agent))
		{
			++Ret;
		}
	}
	return Ret;
}

bool ATSPlayerStateBase::AddNewAgent(AActor* NewAgent)
{
	if (GetLocalRole() < ROLE_Authority)
	{
		return false;
	}

	if (IsValid(NewAgent) && GetNumberOfAvailableAgents() > 0)
	{
		OwnedAgents.Emplace(NewAgent);

		if (GetPlayerController()->IsLocalController())
		{
			OnRep_OwnedAgents();
		}
		return true;
	}
	return false;
}

void ATSPlayerStateBase::RemoveAgent(AActor* InAgent)
{
	if (GetLocalRole() < ROLE_Authority)
	{
		return;
	}

	OwnedAgents.Remove(InAgent);
}

bool ATSPlayerStateBase::OwnAgent(const AActor* InAgent) const
{
	if (GetLocalRole() == ROLE_Authority)
	{
		return OwnedAgents.Contains(InAgent);
	}

	if (HasLocalNetOwner())
	{
		return OwnedAgents.Contains(InAgent);
	}
	return false;
}

void ATSPlayerStateBase::NotifyAgentAdded(AActor* NewAgent)
{
	if (NewAgent)
	{
		OnOwnedAgentAdded().Broadcast(NewAgent);
		UE_LOG(LogTongSimCore, Log, TEXT("Notify distribution agent added, agent name: %s"), *GetNameSafe(NewAgent));
	}
}

int32 ATSPlayerStateBase::GetSelfMaxAgentNumber() const
{
	// dedicated server owned player cant have any available agent.
	if (GetPlayerController()->IsLocalPlayerController() && GetNetMode() == NM_DedicatedServer)
	{
		return 0;
	}

	if (PlayerType != ETSPlayerType::HumanPlayer && PlayerType != ETSPlayerType::TongSimAutoManaged)
	{
		return 0;
	}

	if (const UTSGameplaySettings* GameSettings = GetDefault<UTSGameplaySettings>())
	{
		return FMath::Clamp(GameSettings->MaxAgentNumber, 1, 10);
	}

	return 0;
}

void ATSPlayerStateBase::OnRep_OwnedAgents()
{
	// TODO:
	if (OwnedAgents.Num() == 1 && IsValid(OwnedAgents[0]))
	{
		NotifyAgentAdded(OwnedAgents[0]);
	}
}
