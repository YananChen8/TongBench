#include "TSGameModeBase.h"
#include "Common/TongSimGameplayFunc.h"
#include "Core/TSCommandLineParams.h"
#include "GameFramework/GameSession.h"
#include "Kismet/GameplayStatics.h"
#include "Net/OnlineEngineInterface.h"
#include "Online/TSGameSessionBase.h"
#include "Player/TSPlayerControllerBase.h"
#include "Player/TSPlayerStateBase.h"
#include "TSLogChannels.h"

FString ATSGameModeBase::InitNewPlayer(APlayerController* NewPlayerController, const FUniqueNetIdRepl& UniqueId, const FString& Options, const FString& Portal)
{
	if (ATSPlayerStateBase* PlayerState = NewPlayerController->GetPlayerState<ATSPlayerStateBase>())
	{
		FString PlayerTypeName = UGameplayStatics::ParseOption(Options, TEXT("PlayerType"));

		if (PlayerTypeName.Equals(FString(TEXT("VR")), ESearchCase::IgnoreCase))
		{
			PlayerState->SetPlayerType(ETSPlayerType::VR);
		}
		else if (PlayerTypeName.Equals(FString(TEXT("HumanPlayer")), ESearchCase::IgnoreCase))
		{
			PlayerState->SetPlayerType(ETSPlayerType::HumanPlayer);
		}
		else
		{
			UE_LOG(LogTongSimCore, Error, TEXT("PlayerType parse error!"));
		}
	}
	else
	{
		UE_LOG(LogTongSimCore, Warning, TEXT("Can't get TongSimpPlayer state. Check!"));
	}

	return Super::InitNewPlayer(NewPlayerController, UniqueId, Options, Portal);
}

UClass* ATSGameModeBase::GetDefaultPawnClassForController_Implementation(AController* InController)
{
	const ETSPlayerType PlayerType = GetPlayerTypeForController(InController);
	if (PawnClassMap.Contains(PlayerType))
	{
		const TSoftClassPtr<APawn> PawnSoftClassPtr = PawnClassMap[PlayerType];
		if (!PawnSoftClassPtr.IsNull())
		{
			return PawnSoftClassPtr.LoadSynchronous();
		}
	}
	return Super::GetDefaultPawnClassForController_Implementation(InController);
}

void ATSGameModeBase::PostLogin(APlayerController* NewPlayer)
{
	Super::PostLogin(NewPlayer);

}

void ATSGameModeBase::Logout(AController* Exiting)
{
	Super::Logout(Exiting);
}

void ATSGameModeBase::BeginPlay()
{
	Super::BeginPlay();

	// if (GetNetMode() != NM_Standalone)
	// {
	// 	HostSessionServer();
	// }
}

ATSPlayerControllerBase* ATSGameModeBase::DistributeAgent(AActor* InAgent) const
{
	if (InAgent)
	{
		// Try find :
		for (FConstPlayerControllerIterator Iterator = GetWorld()->GetPlayerControllerIterator(); Iterator; ++Iterator)
		{
			if (ATSPlayerControllerBase* PC = Cast<ATSPlayerControllerBase>(Iterator->Get()))
			{
				const ATSPlayerStateBase* PlayerState = PC->GetPlayerState<ATSPlayerStateBase>();
				if (PlayerState->OwnAgent(InAgent))
				{
					return PC;
				}
			}
		}

		// Assign a new valid player state for agent:
		for (FConstPlayerControllerIterator Iterator = GetWorld()->GetPlayerControllerIterator(); Iterator; ++Iterator)
		{
			if (ATSPlayerControllerBase* PC = Cast<ATSPlayerControllerBase>(Iterator->Get()))
			{
				ATSPlayerStateBase* PlayerState = PC->GetPlayerState<ATSPlayerStateBase>();
				if (PlayerState->GetNumberOfAvailableAgents() > 0 && PlayerState->AddNewAgent(InAgent))
				{
					return PC;
				}
			}
		}
	}
	return nullptr;
}

void ATSGameModeBase::HostSessionServer()
{
	if (ATSGameSessionBase* TSGameSessionBase = Cast<ATSGameSessionBase>(GameSession))
	{
		if (!UOnlineEngineInterface::Get()->DoesSessionExist(GetWorld(), GameSession->SessionName))
		{
			if (ULocalPlayer* LP = GetGameInstance()->GetFirstGamePlayer())
			{
				int32 GroupId = FTSCommandLineParams::Get().TongSimDistributionGroupID;
				TSGameSessionBase->HostSession(LP->GetPreferredUniqueNetId().GetUniqueNetId(), NAME_GameSession, true, GroupId);
			}
		}
	}
}

ETSPlayerType ATSGameModeBase::GetPlayerTypeForController(const AController* InController) const
{
	if (InController != nullptr)
	{
		if (const ATSPlayerStateBase* TongSimPS = InController->GetPlayerState<ATSPlayerStateBase>())
		{
			return TongSimPS->GetPlayerType();
		}
	}

	return ETSPlayerType::Inactive;
}
