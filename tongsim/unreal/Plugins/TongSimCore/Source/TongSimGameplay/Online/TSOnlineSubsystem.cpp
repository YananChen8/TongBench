// Fill out your copyright notice in the Description page of Project Settings.


#include "TSOnlineSubsystem.h"
#include "OnlineSubsystemUtils.h"
#include "OnlineSessionSettings.h"
#include "OnlineSubsystem.h"
#include "TSGameSessionBase.h"
#include "TSLogChannels.h"
#include "Core/TSCommandLineParams.h"
#include "GameFramework/GameModeBase.h"

UTSOnlineSubsystem* UTSOnlineSubsystem::Instance = nullptr;

void UTSOnlineSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	AddToRoot();
	Instance = this;
}

void UTSOnlineSubsystem::Deinitialize()
{
	Instance = nullptr;
	RemoveFromRoot();
	Super::Deinitialize();
}

void UTSOnlineSubsystem::Tick(float DeltaTime)
{
}

TStatId UTSOnlineSubsystem::GetStatId() const
{
	RETURN_QUICK_DECLARE_CYCLE_STAT(UTSOnlineSubsystem, STATGROUP_Tickables);
}

bool UTSOnlineSubsystem::IsTickableInEditor() const
{
	return false;
}

ETickableTickType UTSOnlineSubsystem::GetTickableTickType() const
{
	return ETickableTickType::Conditional;
}

bool UTSOnlineSubsystem::IsTickable() const
{
	return !HasAnyFlags(RF_ClassDefaultObject);
}

ATSGameSessionBase* UTSOnlineSubsystem::GetGameSession() const
{
	if (UWorld* World = GetWorld())
	{
		if (AGameModeBase* GameMode = World->GetAuthGameMode())
		{
			return Cast<ATSGameSessionBase>(GameMode->GameSession);
		}
	}
	return nullptr;
}

bool UTSOnlineSubsystem::FindSessionsWithTongSimGroupID(ULocalPlayer* PlayerOwner, bool bIsPreScene, bool bLANMatch)
{
	if (PlayerOwner)
	{
		if (ATSGameSessionBase* const GameSession = GetGameSession())
		{
			GameSession->OnFindSessionsComplete().RemoveAll(this);
			OnSearchSessionsCompleteDelegateHandle = GameSession->OnFindSessionsComplete().AddUObject(this, &UTSOnlineSubsystem::OnSearchSessionsComplete);
			GameSession->FindSessions(PlayerOwner->GetPreferredUniqueNetId().GetUniqueNetId(), NAME_GameSession, bLANMatch, bIsPreScene, TongSimClientGroupID);
			return true;
		}
	}
	return false;
}

bool UTSOnlineSubsystem::JoinSession(ULocalPlayer* LocalPlayer, const FOnlineSessionSearchResult& SearchResult)
{
	if (LocalPlayer)
	{
		if (ATSGameSessionBase* const GameSession = GetGameSession())
		{
			OnJoinSessionCompleteDelegateHandle = GameSession->OnJoinSessionComplete().AddUObject(this, &UTSOnlineSubsystem::OnJoinSessionComplete);
			return GameSession->JoinSession(LocalPlayer->GetPreferredUniqueNetId().GetUniqueNetId(), NAME_GameSession, SearchResult);
		}
	}

	return false;
}

void UTSOnlineSubsystem::UpdateSearchStatus()
{
	if (ATSGameSessionBase* GameSession = GetGameSession())
	{
		int32 CurrentSearchIdx, NumSearchResults;
		EOnlineAsyncTaskState::Type SearchState = GameSession->GetSearchResultStatus(CurrentSearchIdx, NumSearchResults);

		switch (SearchState)
		{
		case EOnlineAsyncTaskState::InProgress:
			break;
		case EOnlineAsyncTaskState::Done:
			{
				CachedSearchResults = GameSession->GetSearchResults();
				check(CachedSearchResults.Num() == NumSearchResults);
			}
			break;
		case EOnlineAsyncTaskState::Failed:
		case EOnlineAsyncTaskState::NotStarted:
		default:
			{
			}
			break;
		}
	}
}

void UTSOnlineSubsystem::OnJoinSessionComplete(EOnJoinSessionCompleteResult::Type Result)
{
	ATSGameSessionBase* const GameSession = GetGameSession();
	if (GameSession)
	{
		GameSession->OnJoinSessionComplete().Remove(OnJoinSessionCompleteDelegateHandle);
	}

	if (Result != EOnJoinSessionCompleteResult::Success)
	{
		UE_LOG(LogTongSimCore, Warning, TEXT("Join Session Failed"));
	}

	InternalTravelToSession(NAME_GameSession);
}

void UTSOnlineSubsystem::OnSearchSessionsComplete(bool bWasSuccessful)
{
	ATSGameSessionBase* const Session = GetGameSession();
	if (Session)
	{
		Session->OnFindSessionsComplete().Remove(OnSearchSessionsCompleteDelegateHandle);
		UpdateSearchStatus();

		// TODO:
		if (TongSimClientGroupID >= 0)
		{
			TryConnectToSameGroupServer();
		}
	}
}

void UTSOnlineSubsystem::InternalTravelToSession(const FName& SessionName)
{
	APlayerController* const PlayerController = GetGameInstance()->GetFirstLocalPlayerController();
	if (!PlayerController)
	{
		UE_LOG(LogTongSimCore, Error, TEXT("Travel to session failed, Player controller is null"));
	}

	IOnlineSubsystem* OnlineSub = Online::GetSubsystem(GetWorld());
	if (!OnlineSub)
	{
		UE_LOG(LogTongSimCore, Error, TEXT("Travel to session failed, Online subsystem is null"));
	}

	FString URL;
	IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
	if (!Sessions.IsValid() || !Sessions->GetResolvedConnectString(SessionName, URL))
	{
		UE_LOG(LogTongSimCore, Error, TEXT("Travel to session failed, Session is not valid"));
	}

	PlayerController->ClientTravel(URL, TRAVEL_Absolute);
}


bool UTSOnlineSubsystem::TryConnectToSameGroupServer()
{
	for (FOnlineSessionSearchResult& CachedSearchResult : CachedSearchResults)
	{
		if (CachedSearchResult.IsValid())
		{
			int32 ServerGroupId = -1;
			CachedSearchResult.Session.SessionSettings.Get(FName(TEXT("TongSimGroupID")), ServerGroupId);

			if (ServerGroupId == TongSimClientGroupID)
			{
				return JoinSession(GetGameInstance()->GetFirstGamePlayer(), CachedSearchResult);
			}
		}
	}
	return false;
}
