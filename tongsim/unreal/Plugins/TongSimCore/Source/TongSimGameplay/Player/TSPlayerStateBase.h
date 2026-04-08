// Fill out your copyright notice in the Description page of Project Settings.
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerState.h"
#include "TSPlayerStateBase.generated.h"

UENUM(BlueprintType)
enum class ETSPlayerType : uint8
{
	HumanPlayer = 0,
	VR,
	VisionMoCap,
	TongSimAutoManaged,
	Inactive
};

/*
 * Manage agent for distribution.
 */
UCLASS()
class TONGSIMGAMEPLAY_API ATSPlayerStateBase : public APlayerState
{
	GENERATED_BODY()

	/* AActor Interface */
protected:
	virtual void PreInitializeComponents() override;

	/* ~AActor Interface */

	/* Tong Sim Player Type */
public:
	void SetPlayerType(ETSPlayerType NewType);
	ETSPlayerType GetPlayerType() const { return PlayerType; }

private:
	UPROPERTY(VisibleAnywhere, Replicated, BlueprintReadWrite, Category = "TongSim|Player", meta = (AllowPrivateAccess))
	ETSPlayerType PlayerType = ETSPlayerType::Inactive;
	/* ~Tong Sim Player Type */

	/* Agent Manager */
public:
	int32 GetNumberOfAvailableAgents() const;

	int32 GetCurrentAgentNumber() const;

	bool AddNewAgent(AActor* NewAgent);
	void RemoveAgent(AActor* InAgent);
	bool OwnAgent(const AActor* InAgent) const;

	DECLARE_EVENT_OneParam(ATSPlayerStateBase, FOnOwnedAgentAdded, AActor*);
	FOnOwnedAgentAdded& OnOwnedAgentAdded() const { return OnOwnedAgentAddedEvent; }

protected:
	// TODO: Fix multi agent in one client
	virtual void NotifyAgentAdded(AActor* NewAgent);

	int32 GetSelfMaxAgentNumber() const;

private:
	mutable FOnOwnedAgentAdded OnOwnedAgentAddedEvent;

	int32 MaxAgentNumPerClient = 0;

	UPROPERTY(VisibleAnywhere, ReplicatedUsing=OnRep_OwnedAgents)
	TArray<TObjectPtr<AActor>> OwnedAgents;

	UFUNCTION()
	void OnRep_OwnedAgents();
	/* ~Agent Manager */
};
