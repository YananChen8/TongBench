// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TSUISubsystem.generated.h"

class UTSPrimaryLayout;
/**
 *
 */
UCLASS()
class TONGSIMGAMEPLAY_API UTSUISubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual UWorld* GetWorld() const override;

	static UTSUISubsystem* GetInstance() { return Instance; }

	UTSPrimaryLayout* GetCurrentPrimaryLayout() const { return CurrentPrimaryLayout; }

protected:
	virtual void NotifyPlayerAdded(ULocalPlayer* InLocalPlayer);
	virtual void NotifyPlayerRemoved(ULocalPlayer* InLocalPlayer);

private:
	void CreatePrimaryLayoutWidget(ULocalPlayer* InLocalPlayer);
	void RemovePrimaryLayoutWidget(ULocalPlayer* InLocalPlayer);

	void AddLayoutToViewport(ULocalPlayer* LocalPlayer, UTSPrimaryLayout* Layout);

	UPROPERTY(Transient)
	TObjectPtr<UTSPrimaryLayout> CurrentPrimaryLayout = nullptr;

	UPROPERTY(EditAnywhere)
	TSoftClassPtr<UTSPrimaryLayout> LayoutClass;

	static UTSUISubsystem* Instance;
};
