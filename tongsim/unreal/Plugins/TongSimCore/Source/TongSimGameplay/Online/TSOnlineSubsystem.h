// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TSOnlineSettings.h"
#include "Interfaces/OnlineSessionInterface.h"
#include "TSOnlineSubsystem.generated.h"

class ATSGameSessionBase;
class FOnlineSessionSearchResult;
/**
 *
 */
UCLASS

()
class TONGSIMGAMEPLAY_API UTSOnlineSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// Begin FTickableGameObject Interface.
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override;
	virtual bool IsTickableInEditor() const override;
	virtual ETickableTickType GetTickableTickType() const override;
	virtual bool IsTickable() const override;
	// End FTickableGameObject Interface.

	// Online Session API:
	ATSGameSessionBase* GetGameSession() const;

	const TArray<FOnlineSessionSearchResult>& GetSearchResults() const {return  CachedSearchResults;}

	bool FindSessionsWithTongSimGroupID(ULocalPlayer* PlayerOwner, bool bIsPreScene, bool bLANMatch);
	bool JoinSession(ULocalPlayer* LocalPlayer, const FOnlineSessionSearchResult& SearchResult);

	static UTSOnlineSubsystem* GetInstance() {return  Instance;}

private:
	TArray<FOnlineSessionSearchResult> CachedSearchResults;

	FDelegateHandle OnJoinSessionCompleteDelegateHandle;
	FDelegateHandle OnSearchSessionsCompleteDelegateHandle;

	void UpdateSearchStatus();

	void OnJoinSessionComplete(EOnJoinSessionCompleteResult::Type Result);
	void OnSearchSessionsComplete(bool bWasSuccessful);

	static UTSOnlineSubsystem* Instance;

	/** Travel directly to the named session */
	void InternalTravelToSession(const FName& SessionName);

	/*
	 *  Tong Sim Managed Server-Client
	 */
public:
	int32 TongSimClientGroupID = -1;

private:
	bool TryConnectToSameGroupServer();

	float TotalTime = 0.f;
};
