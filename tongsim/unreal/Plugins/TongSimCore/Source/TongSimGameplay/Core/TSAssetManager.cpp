// Fill out your copyright notice in the Description page of Project Settings.


#include "TSAssetManager.h"

#include "TSCommandLineParams.h"
#include "TSGameplayTags.h"
#include "TSGlobalDataAsset.h"
#include "TSLogChannels.h"
#include "Engine/DataAsset.h"
#include "Misc/App.h"
#include "Stats/StatsMisc.h"
#include "Engine/Engine.h"
#include "Misc/ScopedSlowTask.h"

UTSAssetManager& UTSAssetManager::Get()
{
	check(GEngine);

	if (UTSAssetManager* Singleton = Cast<UTSAssetManager>(GEngine->AssetManager))
	{
		return *Singleton;
	}

	UE_LOG(LogTongSimCore, Fatal, TEXT("Invalid AssetManagerClassName in DefaultEngine.ini. It must be set to UTTAssetManager!"));

	// Fatal error above prevents this from being called.
	return *NewObject<UTSAssetManager>();
}

const UTSGlobalDataAsset& UTSAssetManager::GetGlobalDataAsset()
{
	return GetOrLoadTypedGameData<UTSGlobalDataAsset>(GlobalDataAssetPath);
}

UObject* UTSAssetManager::SynchronousLoadAsset(const FSoftObjectPath& AssetPath)
{
	if (AssetPath.IsValid())
	{
		if (IsInitialized())
		{
			return GetStreamableManager().LoadSynchronous(AssetPath, false);
		}

		// Use LoadObject if asset manager isn't ready yet.
		return AssetPath.TryLoad();
	}

	return nullptr;
}

TSharedPtr<FStreamableHandle> UTSAssetManager::AsynchronousLoadAssetAndKeepInMemory(const FSoftObjectPath& AssetPath)
{
	if (AssetPath.IsValid())
	{
		TWeakObjectPtr<UTSAssetManager> WeakThis(this); // using weak ptr in case 'this' has gone out of scope by the time this lambda is called
		const auto KeepInMemoryLambda = [WeakThis, AssetPath]()
		{
			if (UTSAssetManager* StrongThis = WeakThis.Get())
			{
				UObject* LoadedAsset = AssetPath.TryLoad();
				ensureMsgf(LoadedAsset, TEXT("Failed to load %s"), *AssetPath.ToString());
				StrongThis->AddLoadedAsset(Cast<UObject>(LoadedAsset));
				UE_LOG(LogTongSimCore, Log, TEXT("Async loading asset is complete, asset name %s"), *GetNameSafe(LoadedAsset));
			}
		};
		UE_LOG(LogTongSimCore, Log, TEXT("Start async loading asset, asset name %s"), *AssetPath.ToString());
		return GetStreamableManager().RequestAsyncLoad(AssetPath, KeepInMemoryLambda);
	}

	return nullptr;
}

void UTSAssetManager::AddLoadedAsset(const UObject* Asset)
{
	if (ensureAlways(Asset))
	{
		LoadedAssets.Add(Asset);
	}
}

void UTSAssetManager::StartInitialLoading()
{
	SCOPED_BOOT_TIMING("UTSAssetManager::StartInitialLoading");
	Super::StartInitialLoading();

	// Command Line
	FTSCommandLineParams::InitializeCommandLineParams();

	// TODO by wukunlun: Use stream handle for Progress or LoadingScreen
	const UTSGlobalDataAsset& GlobalDataAsset = GetGlobalDataAsset();
	if (!FTSCommandLineParams::Get().bDisablePreLoadedAsset)
	{
		// Async Load Asset:
		for (const TSoftClassPtr<UObject>& AsyncLoadAssetPtr : GlobalDataAsset.AsyncLoadAssets)
		{
			AsynchronousLoadAssetAndKeepInMemory(AsyncLoadAssetPtr.ToSoftObjectPath());
		}

		// Sync Load Asset:
		for (const TSoftClassPtr<UObject>& SyncLoadAssetPtr : GlobalDataAsset.SyncLoadAssets)
		{
			GetSubclass<UObject>(SyncLoadAssetPtr, true);
		}
	}
}

#if WITH_EDITOR
void UTSAssetManager::PreBeginPIE(bool bStartSimulate)
{
	Super::PreBeginPIE(bStartSimulate);

	FScopedSlowTask SlowTask(0, NSLOCTEXT("TongSim Editor", "BeginLoading TongSim Global DataAsset", "Loading TongSim Global Data"));
	const bool bShowCancelButton = false;
	const bool bAllowInPIE = true;
	SlowTask.MakeDialog(bShowCancelButton, bAllowInPIE);

	const UTSGlobalDataAsset& LocalGameDataCommon = GetGlobalDataAsset();

	// Intentionally after GetGameData to avoid counting GameData time in this timer
	SCOPE_LOG_TIME_IN_SECONDS(TEXT("PreBeginPIE asset preloading complete"), nullptr);
}
#endif

UPrimaryDataAsset* UTSAssetManager::LoadGameDataOfClass(TSubclassOf<UPrimaryDataAsset> DataClass, const TSoftObjectPtr<UPrimaryDataAsset>& DataClassPath, FPrimaryAssetType PrimaryAssetType)
{
	UPrimaryDataAsset* Asset = nullptr;

	DECLARE_SCOPE_CYCLE_COUNTER(TEXT("Loading TongSim Global DataAsset Object"), STAT_GameData, STATGROUP_LoadTime);
	if (!DataClassPath.IsNull())
	{
#if WITH_EDITOR
		FScopedSlowTask SlowTask(0, FText::Format(NSLOCTEXT("TongSim Editor", "BeginLoadingGameDataTask", "Loading GameData {0}"), FText::FromName(DataClass->GetFName())));
		const bool bShowCancelButton = false;
		const bool bAllowInPIE = true;
		SlowTask.MakeDialog(bShowCancelButton, bAllowInPIE);
#endif
		UE_LOG(LogTongSimCore, Log, TEXT("Loading TongSim Global DataAsset: %s ..."), *DataClassPath.ToString());
		// TODO: Figure out GetPrimaryAssetIdList return empty?
		Asset = GetAsset(DataClassPath);
		SCOPE_LOG_TIME_IN_SECONDS(TEXT("    ... TongSim Global DataAsset Loaded!"), nullptr);
		/*if (GIsEditor)
		{
			Asset = DataClassPath.LoadSynchronous();
			LoadPrimaryAssetsWithType(PrimaryAssetType);
		}
		else
		{
			TSharedPtr<FStreamableHandle> Handle = LoadPrimaryAssetsWithType(PrimaryAssetType);
			if (Handle.IsValid())
			{
				Handle->WaitUntilComplete(0.0f, false);

				Asset = Cast<UPrimaryDataAsset>(Handle->GetLoadedAsset());
			}
		}*/
	}

	if (Asset)
	{
		GameDataMap.Add(DataClass, Asset);
	}
	else
	{
		// It is not acceptable to fail to load any GameData asset. It will result in soft failures that are hard to diagnose.
		UE_LOG(LogTongSimCore, Fatal, TEXT("Failed to load GameData asset at %s. Type %s. This is not recoverable and likely means you do not have the correct data to run %s."), *DataClassPath.ToString(), *PrimaryAssetType.ToString(), FApp::GetProjectName());
	}

	return Asset;
}
