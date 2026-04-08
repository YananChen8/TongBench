#include "TSGrpcSubsystem.h"

#include "EngineUtils.h"
#include "rpc_event.h"
#include "rpc_server.h"
#include "util/channel.h"

DECLARE_STATS_GROUP(TEXT("TongSim gRPC"), STATGROUP_gRPC, STATCAT_Advanced);

DECLARE_CYCLE_STAT(TEXT("TongSim gRPC Tick Handling"), STAT_UTSGrpcSubsystem_UpdateRpcRouter, STATGROUP_gRPC)

UTSGrpcSubsystem* UTSGrpcSubsystem::Instance = nullptr;

void UTSGrpcSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	StartGrpcServer();

	Instance = this;

	// hooks
	FWorldDelegates::OnPostWorldInitialization.AddUObject(this, &ThisClass::HandlePostWorldInit);
	FWorldDelegates::OnWorldCleanup.AddUObject(this, &ThisClass::HandleWorldCleanup);
}

void UTSGrpcSubsystem::Deinitialize()
{
	FWorldDelegates::OnPostWorldInitialization.RemoveAll(this);
	FWorldDelegates::OnWorldCleanup.RemoveAll(this);

	Instance = nullptr;

	StopGrpcServer();
	Super::Deinitialize();
}

void UTSGrpcSubsystem::Tick(float DeltaTime)
{
	UpdateRpcRouter();
}

void UTSGrpcSubsystem::StartGrpcServer()
{
	// server的回调，消息会被投递到管道里
	auto CallBack = [this](tongos::RpcEvent RpcEvent)
	{
		EventChannel->send(std::move(RpcEvent));
	};

	EventChannel = MakeShareable(new tongos::Channel<tongos::RpcEvent>);
	RpcRouter = MakeShareable(new tongos::RpcRouter());
	const FString IP = "0.0.0.0:5726"; // TODO by @wukunlun
	RpcServer = MakeShareable(new tongos::RpcServer(TCHAR_TO_UTF8(*IP), *RpcRouter));
	RpcServer->addWorker(CallBack);
	RpcServer->start();
}

void UTSGrpcSubsystem::StopGrpcServer()
{
	RpcServer.Reset();
	EventChannel->close();
	EventChannel.Reset();
}

void UTSGrpcSubsystem::UpdateRpcRouter()
{
	SCOPE_CYCLE_COUNTER(STAT_UTSGrpcSubsystem_UpdateRpcRouter);
	while (true)
	{
		// 处理管道里的消息
		auto OptionalValue = EventChannel->try_receive();
		if (!OptionalValue)
		{
			break;
		}
		RpcRouter->handle(std::move(OptionalValue).value());
	}
}

AActor* UTSGrpcSubsystem::FindActorByGuid(const FGuid& Id) const
{
	if (const TWeakObjectPtr<AActor>* P = IdToActor.Find(Id))
	{
		return P->Get();
	}
	return nullptr;
}

FGuid UTSGrpcSubsystem::FindGuidByActor(AActor* Actor) const
{
	if (!Actor)
	{
		return FGuid();
	}
	if (const FGuid* Found = ActorToId.Find(Actor))
	{
		return *Found;
	}
	return FGuid();
}

bool UTSGrpcSubsystem::ShouldAddressActor(const AActor* Actor) const
{
	if (!IsValid(Actor))
	{
		return false;
	}

	// 排除模板/默认对象/被销毁中
	if (Actor->HasAnyFlags(RF_ClassDefaultObject | RF_ArchetypeObject) || Actor->IsActorBeingDestroyed() || Actor->IsPendingKillPending())
	{
		return false;
	}

	// TODO: 黑名单/白名单等（当前：默认所有有效 Actor 都注册）
	return true;
}

FGuid UTSGrpcSubsystem::RegisterActor(AActor* Actor)
{
	check(IsInGameThread());

	if (!ShouldAddressActor(Actor))
	{
		return FGuid();
	}

	// 已有则复用
	if (const FGuid* Existing = ActorToId.Find(Actor))
	{
		return *Existing;
	}

	FGuid NewId = FGuid::NewGuid();
	// TODO: 是否需要检查重试？
	// while (IdToActor.Contains(NewId))
	// {
	// 	NewId = FGuid::NewGuid();
	// }

	IdToActor.Add(NewId, Actor);
	ActorToId.Add(Actor, NewId);

	// EndPlay
	Actor->OnEndPlay.AddUniqueDynamic(this, &ThisClass::HandleActorEndPlay);
	// Destroyed（有些情况下先触发 Destroyed）
	Actor->OnDestroyed.AddUniqueDynamic(this, &ThisClass::HandleActorDestroyed);

	UE_LOG(LogTongSimGRPC, Verbose, TEXT("Registered Actor %s -> %s"), *GetNameSafe(Actor), *NewId.ToString(EGuidFormats::DigitsWithHyphensInBraces));
	return NewId;
}

void UTSGrpcSubsystem::UnregisterActor(AActor* Actor)
{
	check(IsInGameThread());
	if (!Actor)
	{
		return;
	}

	if (const FGuid* Found = ActorToId.Find(Actor))
	{
		const FGuid Id = *Found;

		// 解绑事件
		Actor->OnEndPlay.RemoveAll(this);
		Actor->OnDestroyed.RemoveAll(this);

		// 仅从 Actor->Id 侧移除；Id->Actor 保留（弱指针将失效）
		ActorToId.Remove(Actor);
		// 标记为 Destroyed
		DestroyedIds.Add(Id);

		UE_LOG(LogTongSimGRPC, Verbose, TEXT("Marked Destroyed Actor %s <- %s"), *GetNameSafe(Actor), *Id.ToString(EGuidFormats::DigitsWithHyphensInBraces));
	}
}

void UTSGrpcSubsystem::PurgeInvalidActor()
{
	// 清理失效弱引用
	TArray<FGuid> DeadIds;
	for (const auto& Pair : IdToActor)
	{
		if (!Pair.Value.IsValid() && !DestroyedIds.Contains(Pair.Key))
		{
			DeadIds.Add(Pair.Key);
		}
	}
	for (const FGuid& Id : DeadIds)
	{
		IdToActor.Remove(Id);
	}

	TArray<TWeakObjectPtr<AActor>> DeadObjs;
	for (const auto& Pair : ActorToId)
	{
		if (!Pair.Key.IsValid())
		{
			DeadObjs.Add(Pair.Key);
		}
	}
	for (const auto& K : DeadObjs)
	{
		ActorToId.Remove(K);
	}

	if (DeadIds.Num() || DeadObjs.Num())
	{
		UE_LOG(LogTongSimGRPC, Verbose, TEXT("Purged invalid entries: %d ids, %d objs"), DeadIds.Num(), DeadObjs.Num());
	}
}

void UTSGrpcSubsystem::RefreshActorMappings()
{
	DestroyedIds.Empty();
	PurgeInvalidActor();

	// 扫描现存 Actor
	for (TActorIterator<AActor> It(GetWorld()); It; ++It)
	{
		RegisterActor(*It);
	}
}

void UTSGrpcSubsystem::HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues IVS)
{
	if (!World || World->IsPreviewWorld())
	{
		UE_LOG(LogTongSimGRPC, Warning, TEXT("[HandlePostWorldInit] World is not valid."));
		return;
	}

	// 做一次清理
	DestroyedIds.Empty();
	PurgeInvalidActor();

	// 扫描现存 Actor
	for (TActorIterator<AActor> It(World); It; ++It)
	{
		RegisterActor(*It);
	}

	ActorSpawnedDelegateHandle = World->AddOnActorSpawnedHandler(FOnActorSpawned::FDelegate::CreateUObject(this, &ThisClass::HandleActorSpawned));

	UE_LOG(LogTongSimGRPC, Log, TEXT("[HandlePostWorldInit] PostWorldInit scan complete. Current registered: %d"), IdToActor.Num());
}

void UTSGrpcSubsystem::HandleWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources)
{
	// 做一次清理
	DestroyedIds.Empty();
	PurgeInvalidActor();
}

void UTSGrpcSubsystem::HandleActorSpawned(AActor* Actor)
{
	if (!Actor)
	{
		UE_LOG(LogTongSimGRPC, Error, TEXT("[HandleActorSpawned] Actor is not valid."));
		return;
	}
	RegisterActor(Actor);
}

void UTSGrpcSubsystem::HandleActorEndPlay(AActor* Actor, EEndPlayReason::Type Reason)
{
	UnregisterActor(Actor);
}

void UTSGrpcSubsystem::HandleActorDestroyed(AActor* Actor)
{
	UnregisterActor(Actor);
}
