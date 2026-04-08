// Fill out your copyright notice in the Description page of Project Settings.


#include "TSGameSessionBase.h"
#include "OnlineSubsystemSessionSettings.h"
#include "OnlineSubsystemUtils.h"
#include "Online/TSOnlineSettings.h"
#include "Interfaces/OnlineSessionInterface.h"
#include "Online/OnlineSessionNames.h"
#include "TSLogChannels.h"

namespace TongSimSession
{
	const FName GROUP_ID_NAME = FName(TEXT("TongSimGroupID"));
}

ATSGameSessionBase::ATSGameSessionBase(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer)
{
	if (!HasAnyFlags(RF_ClassDefaultObject))
	{
		OnFindSessionsCompleteDelegate = FOnFindSessionsCompleteDelegate::CreateUObject(this, &ATSGameSessionBase::OnFindSessionsComplete);
		OnJoinSessionCompleteDelegate = FOnJoinSessionCompleteDelegate::CreateUObject(this, &ATSGameSessionBase::OnJoinSessionComplete);
	}
}

void ATSGameSessionBase::OnFindSessionsComplete(bool bWasSuccessful)
{
	IOnlineSubsystem* const OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
		if (Sessions.IsValid())
		{
			UE_LOG(LogTongSimCore, Log, TEXT("Find Sessions Complete Result Number: %d"),  SearchSettings->SearchResults.Num());
			Sessions->ClearOnFindSessionsCompleteDelegate_Handle(OnFindSessionsCompleteDelegateHandle);
			OnFindSessionsComplete().Broadcast(bWasSuccessful);
		}
	}
}

void ATSGameSessionBase::OnJoinSessionComplete(FName InSessionName, EOnJoinSessionCompleteResult::Type Result)
{
	UE_LOG(LogTongSimCore, Log, TEXT("OnJoinSessionComplete %s bSuccess: %d"), *InSessionName.ToString(), static_cast<int32>(Result));

	IOnlineSubsystem* OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
		if (Sessions.IsValid())
		{
			Sessions->ClearOnJoinSessionCompleteDelegate_Handle(OnJoinSessionCompleteDelegateHandle);
		}
	}

	OnJoinSessionComplete().Broadcast(Result);
}

void ATSGameSessionBase::RegisterServer()
{
	return;
	IOnlineSubsystem* OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		IOnlineSessionPtr SessionInt = OnlineSub->GetSessionInterface();
		if (SessionInt.IsValid())
		{
			// TODO: Remove group id:
			int32 GroupId = -1;
			FParse::Value(FCommandLine::Get(), TEXT("TongSimServer"), GroupId);

			// TODO: Fix is LAN:
			// TODO: Fix listened server
			TSharedPtr<class FTSOnlineSessionSettings> DedicatedServerHostSettings = MakeShareable(new FTSOnlineSessionSettings(true, false, 16));
			DedicatedServerHostSettings->Set(SETTING_MAPNAME, GetWorld()->GetMapName(), EOnlineDataAdvertisementType::ViaOnlineService);

			DedicatedServerHostSettings->Set(FName(TEXT("TongSimGroupID")), GroupId);
			DedicatedServerHostSettings->bAllowInvites = true;
			DedicatedServerHostSettings->bIsDedicated = true;

			UE_LOG(LogTongSimCore, Log, TEXT("[%lld] Registering a LAN Server, Group Id is %d"), GFrameCounter, GroupId);

			HostSettings = DedicatedServerHostSettings;
			SessionInt->CreateSession(0, NAME_GameSession, *HostSettings);
		}
	}
}

bool ATSGameSessionBase::HostSession(TSharedPtr<const FUniqueNetId> UserId, FName InSessionName, bool bIsLAN, int32 TongSimGroupId)
{
	IOnlineSubsystem* const OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
		if (Sessions.IsValid() && UserId.IsValid())
		{
			CurrentSessionParams.SessionName = InSessionName;
			CurrentSessionParams.bIsLAN = bIsLAN;
			CurrentSessionParams.bIsPresence = false;
			CurrentSessionParams.UserId = UserId;
			MaxPlayers = DEFAULT_NUM_PLAYERS;

			HostSettings = MakeShareable(new FTSOnlineSessionSettings(bIsLAN, false, MaxPlayers));

			HostSettings->Set(SETTING_MAPNAME, GetWorld()->GetMapName(), EOnlineDataAdvertisementType::ViaOnlineService);
			HostSettings->Set(TongSimSession::GROUP_ID_NAME, TongSimGroupId, EOnlineDataAdvertisementType::ViaOnlineService);

			UE_LOG(LogTongSimCore, Log, TEXT("[%lld] Registering a LAN Server, Group Id is %d"), GFrameCounter, TongSimGroupId);
			return Sessions->CreateSession(*CurrentSessionParams.UserId, CurrentSessionParams.SessionName, *HostSettings);
		}
	}
	return false;
}

void ATSGameSessionBase::FindSessions(TSharedPtr<const FUniqueNetId> UserId, FName InSessionName, bool bIsLAN, bool bIsPresence, int32 TongSimGroupId)
{
	IOnlineSubsystem* OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		CurrentSessionParams.SessionName = InSessionName;
		CurrentSessionParams.bIsLAN = bIsLAN;
		CurrentSessionParams.bIsPresence = bIsPresence;
		CurrentSessionParams.UserId = UserId;
		IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
		if (Sessions.IsValid() && CurrentSessionParams.UserId.IsValid())
		{
			SearchSettings = MakeShareable(new FTSOnlineSearchSettings(bIsLAN, bIsPresence));
			SearchSettings->QuerySettings.Set(TongSimSession::GROUP_ID_NAME, TongSimGroupId, EOnlineComparisonOp::Equals);

			TSharedRef<FOnlineSessionSearch> SearchSettingsRef = SearchSettings.ToSharedRef();

			OnFindSessionsCompleteDelegateHandle = Sessions->AddOnFindSessionsCompleteDelegate_Handle(OnFindSessionsCompleteDelegate);
			Sessions->FindSessions(*CurrentSessionParams.UserId, SearchSettingsRef);
		}
	}
}

bool ATSGameSessionBase::JoinSession(TSharedPtr<const FUniqueNetId> UserId, FName InSessionName, const FOnlineSessionSearchResult& SearchResult)
{
	bool bResult = false;

	IOnlineSubsystem* OnlineSub = Online::GetSubsystem(GetWorld());
	if (OnlineSub)
	{
		IOnlineSessionPtr Sessions = OnlineSub->GetSessionInterface();
		if (Sessions.IsValid() && UserId.IsValid())
		{
			OnJoinSessionCompleteDelegateHandle = Sessions->AddOnJoinSessionCompleteDelegate_Handle(OnJoinSessionCompleteDelegate);
			bResult = Sessions->JoinSession(*UserId, SessionName, SearchResult);
		}
	}

	return bResult;
}

EOnlineAsyncTaskState::Type ATSGameSessionBase::GetSearchResultStatus(int32& SearchResultIdx, int32& NumSearchResults)
{
	SearchResultIdx = 0;
	NumSearchResults = 0;

	if (SearchSettings.IsValid())
	{
		if (SearchSettings->SearchState == EOnlineAsyncTaskState::Done)
		{
			SearchResultIdx = CurrentSessionParams.BestSessionIdx;
			NumSearchResults = SearchSettings->SearchResults.Num();
		}
		return SearchSettings->SearchState;
	}

	return EOnlineAsyncTaskState::NotStarted;
}

const TArray<FOnlineSessionSearchResult>& ATSGameSessionBase::GetSearchResults() const
{
	return SearchSettings->SearchResults;
}
