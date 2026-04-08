// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OnlineSubsystemTypes.h"
#include "GameFramework/GameSession.h"
#include "Interfaces/OnlineSessionInterface.h"
#include "TSGameSessionBase.generated.h"

struct FTongSimGameSessionParams
{
	/** Name of session settings are stored with */
	FName SessionName;
	/** LAN Match */
	bool bIsLAN;
	/** Presence enabled session */
	bool bIsPresence;
	/** Id of player initiating lobby */
	TSharedPtr<const FUniqueNetId> UserId;
	/** Current search result choice to join */
	int32 BestSessionIdx;

	FTongSimGameSessionParams()
		: SessionName(NAME_None)
		  , bIsLAN(false)
		  , bIsPresence(false)
		  , BestSessionIdx(0)
	{
	}
};

UCLASS()
class TONGSIMGAMEPLAY_API ATSGameSessionBase : public AGameSession
{
	GENERATED_BODY()

public:
	ATSGameSessionBase(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	/** Default number of players allowed in a game */
	static const int32 DEFAULT_NUM_PLAYERS = 50;

protected:
	/** Delegate for searching for sessions */
	FOnFindSessionsCompleteDelegate OnFindSessionsCompleteDelegate;
	/** Delegate after joining a session */
	FOnJoinSessionCompleteDelegate OnJoinSessionCompleteDelegate;

	/** Transient properties of a session during game creation/matchmaking */
	FTongSimGameSessionParams CurrentSessionParams;
	/** Current host settings */
	TSharedPtr<class FTSOnlineSessionSettings> HostSettings;
	/** Current search settings */
	TSharedPtr<class FTSOnlineSearchSettings> SearchSettings;

	void OnFindSessionsComplete(bool bWasSuccessful);
	void OnJoinSessionComplete(FName InSessionName, EOnJoinSessionCompleteResult::Type Result);

	/**
	 * Called when this instance is starting up as a dedicated server
	 */
	virtual void RegisterServer() override;

	/*
	 * Event triggered when a session is joined
	 *
	 * @param SessionName name of session that was joined
	 * @param bWasSuccessful was the create successful
	 */
	DECLARE_EVENT_OneParam(ATSGameSessionBase, FOnJoinSessionComplete, EOnJoinSessionCompleteResult::Type /*Result*/)
	FOnJoinSessionComplete JoinSessionCompleteEvent;

	/*
	 * Event triggered after session search completes
	 */
	DECLARE_EVENT_OneParam(ATSGameSessionBase, FOnFindSessionsComplete, bool /*bWasSuccessful*/)
	FOnFindSessionsComplete FindSessionsCompleteEvent;

public:
	/**
	 * Find an online session
	 *
	 * @param UserId user that initiated the request
	 * @param SessionName name of session
	 * @param bIsLAN is this going to hosted over LAN
	 * @param TongSimGroupId for TongSim managed client
	 */
	bool HostSession(TSharedPtr<const FUniqueNetId> UserId, FName SessionName, bool bIsLAN, int32 TongSimGroupId = -1);

	/**
	 * Find an online session
	 *
	 * @param UserId user that initiated the request
	 * @param InSessionName name of session this search will generate
	 * @param bIsLAN are we searching LAN matches
	 * @param bIsPresence are we searching presence sessions
	 * @param TongSimGroupId for TongSim managed client
	 */
	void FindSessions(TSharedPtr<const FUniqueNetId> UserId, FName InSessionName, bool bIsLAN, bool bIsPresence, int32 TongSimGroupId = -1);

	/**
	 * Joins a session via a search result
	 *
	 * @param InSessionName name of session
	 * @param SearchResult Session to join
	 *
	 * @return bool true if successful, false otherwise
	 */
	bool JoinSession(TSharedPtr<const FUniqueNetId> UserId, FName InSessionName, const FOnlineSessionSearchResult& SearchResult);

	/**
	* Get the search results found and the current search result being probed
	*
	* @param SearchResultIdx idx of current search result accessed
	* @param NumSearchResults number of total search results found in FindGame()
	*
	* @return State of search result query
	*/
	EOnlineAsyncTaskState::Type GetSearchResultStatus(int32& SearchResultIdx, int32& NumSearchResults);

	/**
	 * Get the search results.
	 *
	 * @return Search results
	 */
	const TArray<FOnlineSessionSearchResult>& GetSearchResults() const;

	/** @return the delegate fired when joining a session */
	FOnJoinSessionComplete& OnJoinSessionComplete() { return JoinSessionCompleteEvent; }

	/** @return the delegate fired when search of session completes */
	FOnFindSessionsComplete& OnFindSessionsComplete() { return FindSessionsCompleteEvent; }

	FDelegateHandle OnFindSessionsCompleteDelegateHandle;
	FDelegateHandle OnJoinSessionCompleteDelegateHandle;
};
