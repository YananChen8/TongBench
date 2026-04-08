// Fill out your copyright notice in the Description page of Project Settings.


#include "TSUISubsystem.h"
#include "Common/TSGameplaySettings.h"
#include "Layer/TSPrimaryLayout.h"
#include "Player/TSLocalPlayerBase.h"
#include "TSLogChannels.h"

namespace TSUISubsystemHelper
{
	constexpr int32 PrimaryLayoutZOrder = 1000;
}

UTSUISubsystem* UTSUISubsystem::Instance = nullptr;

void UTSUISubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	if (const UTSGameplaySettings* GameSettings = GetDefault<UTSGameplaySettings>())
	{
		LayoutClass = GameSettings->LayoutClass;
	}

	// Different from Lyra here, we bind delegates to default game instance for PrimaryLayout lifetime.
	if (UGameInstance* GameInstance = GetGameInstance())
	{
		GameInstance->OnLocalPlayerAddedEvent.AddUObject(this, &ThisClass::NotifyPlayerAdded);
		GameInstance->OnLocalPlayerRemovedEvent.AddUObject(this, &ThisClass::NotifyPlayerRemoved);
	}

	Instance = this;
}

void UTSUISubsystem::Deinitialize()
{
	Instance = nullptr;
	Super::Deinitialize();
}

bool UTSUISubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	if (!CastChecked<UGameInstance>(Outer)->IsDedicatedServerInstance())
	{
		TArray<UClass*> ChildClasses;
		GetDerivedClasses(GetClass(), ChildClasses, false);

		// Only create an instance if there is no override implementation defined elsewhere
		return ChildClasses.Num() == 0;
	}
	return false;
}

UWorld* UTSUISubsystem::GetWorld() const
{
	if (GetGameInstance())
	{
		return GetGameInstance()->GetWorld();
	}
	return nullptr;
}

void UTSUISubsystem::NotifyPlayerAdded(ULocalPlayer* InLocalPlayer)
{
	if (UTSLocalPlayerBase* LocalPlayer = Cast<UTSLocalPlayerBase>(InLocalPlayer))
	{
		// Player controller of LocalPlayer is null here, dont create widget directly here, using delegate ~~.
		LocalPlayer->OnPlayerControllerSet.AddWeakLambda(this, [this](UTSLocalPlayerBase* LocalPlayer, APlayerController* PlayerController)
		{
			RemovePrimaryLayoutWidget(LocalPlayer);
			if (CurrentPrimaryLayout)
			{
				AddLayoutToViewport(LocalPlayer, CurrentPrimaryLayout);
			}
			else
			{
				CreatePrimaryLayoutWidget(LocalPlayer);
			}
		});

		if (CurrentPrimaryLayout)
		{
			AddLayoutToViewport(InLocalPlayer, CurrentPrimaryLayout);
		}
		else
		{
			CreatePrimaryLayoutWidget(InLocalPlayer);
		}
	}
}

void UTSUISubsystem::NotifyPlayerRemoved(ULocalPlayer* InLocalPlayer)
{
	if (UTSLocalPlayerBase* LocalPlayer = Cast<UTSLocalPlayerBase>(InLocalPlayer))
	{
		RemovePrimaryLayoutWidget(LocalPlayer);
		LocalPlayer->OnPlayerControllerSet.RemoveAll(this);
		if (CurrentPrimaryLayout != nullptr)
		{
			CurrentPrimaryLayout = nullptr;
		}
	}
}

void UTSUISubsystem::CreatePrimaryLayoutWidget(ULocalPlayer* InLocalPlayer)
{
	if (!LayoutClass.IsNull())
	{
		if (APlayerController* PlayerController = InLocalPlayer->GetPlayerController(GetWorld()))
		{
			TSubclassOf<UTSPrimaryLayout> LayoutAssetClass = LayoutClass.LoadSynchronous();
			if (LayoutAssetClass && !LayoutAssetClass->HasAnyClassFlags(CLASS_Abstract))
			{
				CurrentPrimaryLayout = CreateWidget<UTSPrimaryLayout>(PlayerController, LayoutAssetClass);
				AddLayoutToViewport(InLocalPlayer, CurrentPrimaryLayout);
			}
		}
	}
	else
	{
		UE_LOG(LogTongSimCore, Error, TEXT("Tong Sim Primary Layout Class is Null."));
	}
}

void UTSUISubsystem::RemovePrimaryLayoutWidget(ULocalPlayer* InLocalPlayer)
{
	if (CurrentPrimaryLayout)
	{
		TWeakPtr<SWidget> LayoutSlateWidget = CurrentPrimaryLayout->GetCachedWidget();
		if (LayoutSlateWidget.IsValid())
		{
			UE_LOG(LogTongSimCore, Log, TEXT("[%s] is removing player [%s]'s root layout [%s] from the viewport"), *GetName(), *GetNameSafe(InLocalPlayer), *GetNameSafe(CurrentPrimaryLayout));
			CurrentPrimaryLayout->RemoveFromParent();
			if (LayoutSlateWidget.IsValid())
			{
				UE_LOG(LogTongSimCore, Log, TEXT("root layout [%s] has been removed from the viewport, but other references to its underlying Slate widget still exist."), *GetNameSafe(InLocalPlayer));
			}
			CurrentPrimaryLayout = nullptr;
		}
	}
}

void UTSUISubsystem::AddLayoutToViewport(ULocalPlayer* LocalPlayer, UTSPrimaryLayout* Layout)
{
	CurrentPrimaryLayout->SetPlayerContext(FLocalPlayerContext(LocalPlayer));
	CurrentPrimaryLayout->AddToPlayerScreen(TSUISubsystemHelper::PrimaryLayoutZOrder);
	UE_LOG(LogTongSimCore, Log, TEXT("[%s] is adding player [%s]'s root layout [%s] to the viewport"), *GetName(), *GetNameSafe(LocalPlayer), *GetNameSafe(CurrentPrimaryLayout));
}
