#pragma once
#include "OnlineSessionSettings.h"
#include "Online/OnlineSessionNames.h"

/**
 * General session settings
 */
class FTSOnlineSessionSettings : public FOnlineSessionSettings
{
public:
	FTSOnlineSessionSettings(bool bIsLAN = false, bool bIsPresence = false, int32 MaxNumPlayers = 32)
	{
		NumPublicConnections = MaxNumPlayers;
		if (NumPublicConnections < 0)
		{
			NumPublicConnections = 0;
		}
		NumPrivateConnections = 0;
		bIsLANMatch = bIsLAN;
		bShouldAdvertise = true;
		bAllowJoinInProgress = true;
		bAllowInvites = true;
		bUsesPresence = bIsPresence;
		bAllowJoinViaPresence = true;
		bAllowJoinViaPresenceFriendsOnly = false;
	}

	virtual ~FTSOnlineSessionSettings() {}
};

/**
 * General search setting for a Shooter game
 */
class FTSOnlineSearchSettings : public FOnlineSessionSearch
{
public:
	FTSOnlineSearchSettings(bool bSearchingLAN = false, bool bSearchingPresence = false)
	{
		bIsLanQuery = bSearchingLAN;
		MaxSearchResults = 10;
		PingBucketSize = 50;

		if (bSearchingPresence)
		{
			QuerySettings.Set(SEARCH_LOBBIES, true, EOnlineComparisonOp::Equals);
		}
	}

	virtual ~FTSOnlineSearchSettings() {}
};
