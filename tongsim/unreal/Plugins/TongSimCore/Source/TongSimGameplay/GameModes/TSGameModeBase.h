// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "TSGameModeBase.generated.h"

enum class ETSPlayerType : uint8;
class ATSPlayerControllerBase;

UCLASS()
class TONGSIMGAMEPLAY_API ATSGameModeBase : public AGameModeBase
{
	GENERATED_BODY()

public:

	/* AGameModeBase override */
	virtual FString InitNewPlayer(APlayerController* NewPlayerController, const FUniqueNetIdRepl& UniqueId, const FString& Options, const FString& Portal) override;
	virtual UClass* GetDefaultPawnClassForController_Implementation(AController* InController) override;
	virtual void PostLogin(APlayerController* NewPlayer) override;
	virtual void Logout(AController* Exiting) override;

protected:
	virtual void BeginPlay() override;
	/* ~AGameModeBase override */


	/* Distribution Client-Server Interface */
public:
	ATSPlayerControllerBase* DistributeAgent(AActor* InAgent) const;

private:
	void HostSessionServer();
	/* ~Distribution Client-Server Interface */

	/* Pawn */
public:
	UFUNCTION(BlueprintCallable, Category = "TongSim|Pawn")
	ETSPlayerType GetPlayerTypeForController(const AController* InController) const;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TongSim|Pawn")
	TMap<ETSPlayerType, TSoftClassPtr<APawn>> PawnClassMap;
	/* ~Pawn */
};
