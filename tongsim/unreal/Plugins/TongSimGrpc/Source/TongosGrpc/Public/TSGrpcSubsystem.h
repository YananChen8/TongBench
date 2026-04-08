#pragma once

#include <string>
#include "CoreMinimal.h"
#include "rpc_router.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TSGrpcSubsystem.generated.h"

namespace tongos
{
	template <typename T>
	class Channel;

	class RpcServer;
	class RpcEvent;
}

UCLASS()
class TONGOSGRPC_API UTSGrpcSubsystem final : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	virtual void Tick(float DeltaTime) override;
	virtual bool IsTickable() const override { return !IsTemplate(); }
	virtual TStatId GetStatId() const override{RETURN_QUICK_DECLARE_CYCLE_STAT(UTSGrpcSubsystem, STATGROUP_Tickables);}

	FORCEINLINE static UTSGrpcSubsystem* GetInstance() { return Instance; }

private:
	static UTSGrpcSubsystem* Instance;

	/**
	 * gRPC Server
	 */
public:
	template <typename Handler>
	void RegisterUnaryHandler(const std::string& method, Handler handler)
	{
		RpcRouter->registerUnaryHandler(method, handler);
	}

	template <typename Reactor>
	void RegisterReactor(const std::string& method)
	{
		RpcRouter->registerReactor<Reactor>(method);
	}

	void RefreshActorMappings();

private:
	void StartGrpcServer();
	void StopGrpcServer();
	void UpdateRpcRouter();
	TSharedPtr<tongos::RpcRouter> RpcRouter;
	TSharedPtr<tongos::RpcServer> RpcServer;
	TSharedPtr<tongos::Channel<tongos::RpcEvent>> EventChannel;
	/**
	 * ~gRPC Server
	 */

	/**
	 * gRPC ObjectID Mapping
	 */
public:
	/** 查 GUID -> Actor（可能返回空） */
	AActor* FindActorByGuid(const FGuid& Id) const;

	/** 查 Actor -> GUID（可能返回无效 GUID） */
	FGuid FindGuidByActor(AActor* Actor) const;

	const TMap<FGuid, TWeakObjectPtr<AActor>>& GetIdToActorMap() const {return IdToActor;}
	const TMap<TWeakObjectPtr<AActor>, FGuid>& GetActorToIdMap() const {return ActorToId;}

	const TSet<FGuid>& GetDestroyedIds() const { return DestroyedIds; }
private:
	bool ShouldAddressActor(const AActor* Actor) const;
	FGuid RegisterActor(AActor* Actor);
	void UnregisterActor(AActor* Actor);
	void PurgeInvalidActor();

	/** GUID <-> Actor（弱引用） */
	TMap<FGuid, TWeakObjectPtr<AActor>> IdToActor;
	TMap<TWeakObjectPtr<AActor>, FGuid> ActorToId;
	// 被销毁的 GUID 集合
	TSet<FGuid> DestroyedIds;


	void HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues IVS);
	void HandleWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources);
	void HandleActorSpawned(AActor* Actor);
	UFUNCTION()
	void HandleActorEndPlay(AActor* Actor, EEndPlayReason::Type Reason);
	UFUNCTION()
	void HandleActorDestroyed(AActor* Actor);
	FDelegateHandle ActorSpawnedDelegateHandle;
	/**
	 * ~gRPC ObjectID Mapping
	 */
};
