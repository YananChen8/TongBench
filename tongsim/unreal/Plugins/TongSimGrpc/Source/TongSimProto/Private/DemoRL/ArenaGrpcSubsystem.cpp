#include "DemoRL/ArenaGrpcSubsystem.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "TSGrpcSubsystem.h" // 复用你的注册中心
#include "UObject/SoftObjectPath.h"

using namespace tongos;
UArenaGrpcSubsystem* UArenaGrpcSubsystem::Instance = nullptr;
static TMap<FGuid, TWeakObjectPtr<ULevelStreamingDynamic>> GArena_OldStreaming;
static TMap<FGuid, TWeakObjectPtr<ULevel>> GArena_OldLevel;
static TMap<FGuid, float> GArena_FlushAccum;
static TSet<FGuid> GArena_DidGC;

static bool IsOldArenaFullyUnloaded(const FGuid& ArenaId)
{
	if (const TWeakObjectPtr<ULevelStreamingDynamic>* P = GArena_OldStreaming.Find(ArenaId))
	{
		if (ULevelStreamingDynamic* LSD = P->Get())
		{
			// 以 Streaming 为准：既不 loaded 且无 LoadedLevel 视为完全卸载
			return (!LSD->IsLevelLoaded()) && (LSD->GetLoadedLevel() == nullptr);
		}
	}
	// 若 Streaming 已无效，也可视作已卸载；额外兜底：旧 Level 指针也应无效
	if (const TWeakObjectPtr<ULevel>* L = GArena_OldLevel.Find(ArenaId))
	{
		return !L->IsValid();
	}
	return true;
}

static void MaybeFlushStreaming(UWorld* World, const FGuid& ArenaId, float dt)
{
	if (!IsValid(World)) return;
	float& Acc = GArena_FlushAccum.FindOrAdd(ArenaId);
	Acc += dt;
	if (Acc >= 0.5f) // 每 ~0.5s 刷一次
	{
		UGameplayStatics::FlushLevelStreaming(World);
		Acc = 0.f;
	}
}

static void MaybeDoOneGC(const FGuid& ArenaId)
{
	if (GArena_DidGC.Contains(ArenaId)) return;
	CollectGarbage(RF_NoFlags);
	GArena_DidGC.Add(ArenaId);
}

static void ClearArenaUnloadState(const FGuid& ArenaId)
{
	GArena_OldStreaming.Remove(ArenaId);
	GArena_OldLevel.Remove(ArenaId);
	GArena_FlushAccum.Remove(ArenaId);
	GArena_DidGC.Remove(ArenaId);
}

static UWorld* GetArenaWorld()
{
	if (!GEngine) return nullptr;
	for (const FWorldContext& Ctx : GEngine->GetWorldContexts())
		if (Ctx.WorldType == EWorldType::Game || Ctx.WorldType == EWorldType::PIE)
			return Ctx.World();
	return nullptr;
}

void UArenaGrpcSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	FWorldDelegates::OnPostWorldInitialization.AddUObject(this, &ThisClass::HandlePostWorldInit);
	Instance = this;
}

void UArenaGrpcSubsystem::Deinitialize()
{
	Instance = nullptr;
	FWorldDelegates::OnPostWorldInitialization.RemoveAll(this);
	Super::Deinitialize();
}

void UArenaGrpcSubsystem::Tick(float DeltaTime)
{
	// 为避免遍历时自删导致迭代器失效，先拷贝 Key 再逐个 Tick

	{
		// Load
		TArray<FGuid> Keys;
		LoadReactors.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = LoadReactors.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}
	{
		// Reset
		TArray<FGuid> Keys;
		ResetReactors.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = ResetReactors.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}
	{
		// Destroy
		TArray<FGuid> Keys;
		DestroyReactors.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = DestroyReactors.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}
	{
		// Move
		TArray<FGuid> Keys;
		MoveReactors.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = MoveReactors.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}
}


void UArenaGrpcSubsystem::HandlePostWorldInit(UWorld*, const UWorld::InitializationValues)
{
	if (UTSGrpcSubsystem* Grpc = UTSGrpcSubsystem::GetInstance())
	{
		Grpc->RegisterReactor<ThisClass::FLoadArenaReactor>("/tongsim_lite.arena.ArenaService/LoadArena");
		Grpc->RegisterReactor<ThisClass::FResetArenaReactor>("/tongsim_lite.arena.ArenaService/ResetArena");
		Grpc->RegisterReactor<ThisClass::FDestroyArenaReactor>("/tongsim_lite.arena.ArenaService/DestroyArena");

		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/SetArenaVisible", &ThisClass::SetArenaVisible);
		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/ListArenas", &ThisClass::ListArenas);

		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/SpawnActorInArena", &ThisClass::SpawnActorInArena);
		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/SetActorPoseLocal", &ThisClass::SetActorPoseLocal);
		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/GetActorPoseLocal", &ThisClass::GetActorPoseLocal);

		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/LocalToWorld", &ThisClass::LocalToWorld);
		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/WorldToLocal", &ThisClass::WorldToLocal);

		Grpc->RegisterUnaryHandler("/tongsim_lite.arena.ArenaService/DestroyActorInArena", &ThisClass::DestroyActorInArena);
		Grpc->RegisterReactor<ThisClass::FSimpleMoveTowardsInArenaReactor>("/tongsim_lite.arena.ArenaService/SimpleMoveTowardsInArena");
	}
}

// --------- helpers (与 DemoRLSubsystem 同口径) ----------
bool UArenaGrpcSubsystem::BytesLEToFGuid(const uint8 In[16], FGuid& Out)
{
	auto ReadLE32 = [](const uint8* p)-> uint32
	{
		return (uint32)p[0] | ((uint32)p[1] << 8) | ((uint32)p[2] << 16) | ((uint32)p[3] << 24);
	};
	Out = FGuid((int32)ReadLE32(In + 0), (int32)ReadLE32(In + 4), (int32)ReadLE32(In + 8), (int32)ReadLE32(In + 12));
	return Out.IsValid();
}

void UArenaGrpcSubsystem::FGuidToBytesLE(const FGuid& G, uint8 Out[16])
{
	const uint32 P[4] = {(uint32)G.A, (uint32)G.B, (uint32)G.C, (uint32)G.D};
	for (int i = 0; i < 4; ++i)
	{
		const uint32 v = P[i];
		Out[i * 4 + 0] = v & 0xFF;
		Out[i * 4 + 1] = (v >> 8) & 0xFF;
		Out[i * 4 + 2] = (v >> 16) & 0xFF;
		Out[i * 4 + 3] = (v >> 24) & 0xFF;
	}
}

static FVector FromP(const tongsim_lite::common::Vector3f& v) { return FVector(v.x(), v.y(), v.z()); }

static tongsim_lite::common::Vector3f ToP(const FVector& v)
{
	tongsim_lite::common::Vector3f o;
	o.set_x(v.X);
	o.set_y(v.Y);
	o.set_z(v.Z);
	return o;
}

static FRotator FromR(const tongsim_lite::common::Rotatorf& r) { return FRotator(r.pitch_deg(), r.yaw_deg(), r.roll_deg()); }

static tongsim_lite::common::Rotatorf ToR(const FRotator& r)
{
	tongsim_lite::common::Rotatorf o;
	o.set_roll_deg(r.Roll);
	o.set_pitch_deg(r.Pitch);
	o.set_yaw_deg(r.Yaw);
	return o;
}

FTransform UArenaGrpcSubsystem::FromProtoXf(const tongsim_lite::common::Transform& t) { return FTransform(FromR(t.rotation()), FromP(t.location()), FromP(t.scale())); }

tongsim_lite::common::Transform UArenaGrpcSubsystem::ToProtoXf(const FTransform& t)
{
	tongsim_lite::common::Transform o;
	*o.mutable_location() = ToP(t.GetLocation());
	*o.mutable_rotation() = ToR(t.Rotator());
	*o.mutable_scale() = ToP(t.GetScale3D());
	return o;
}

bool UArenaGrpcSubsystem::ObjectIdToGuid(const tongsim_lite::object::ObjectId& Id, FGuid& Out)
{
	const std::string& B = Id.guid();
	if (B.size() != 16) return false;
	return BytesLEToFGuid(reinterpret_cast<const uint8*>(B.data()), Out);
}

// --------- handlers ----------
static UTSArenaSubsystem* Mgr()
{
	UWorld* W = GetArenaWorld();
	return W ? W->GetSubsystem<UTSArenaSubsystem>() : nullptr;
}

// ---------- Reactor: LoadArena ----------
void UArenaGrpcSubsystem::FLoadArenaReactor::onRequest(tongsim_lite::arena::LoadArenaRequest& Req)
{
	// 同一 Arena 跨类型互斥
	if (Instance->BusyArenas.Contains(ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "Another operation in this arena is in progress."));
		return;
	}

	if (auto* S = Mgr())
	{
		const FSoftObjectPath P(UTF8_TO_TCHAR(Req.level_asset_path().c_str()));
		const TSoftObjectPtr<UWorld> Asset(P);
		Anchor = Instance->FromProtoXf(Req.anchor());
		ArenaId = S->LoadArena(Asset, Anchor, Req.make_visible());
		if (!ArenaId.IsValid())
		{
			this->finish(ResponseStatus(grpc::StatusCode::UNKNOWN, "LoadArena failed"));
			return;
		}
		// 登记并发
		Instance->BusyArenas.Add(ArenaId);
		Instance->LoadReactors.Add(ArenaId, this->sharedSelf<FLoadArenaReactor>());

		Deadline = Instance->AsyncGrpcDeadline;
		Elapsed = 0.f;
		return;
	}
	this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem"));
}

void UArenaGrpcSubsystem::FLoadArenaReactor::onCancel()
{
	Instance->LoadReactors.Remove(ArenaId);
	Instance->BusyArenas.Remove(ArenaId);
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "LoadArena cancelled."));
}

void UArenaGrpcSubsystem::FLoadArenaReactor::Tick(float dt)
{
	Elapsed += dt;

	if (auto* S = Mgr())
	{
		if (S->IsArenaReady(ArenaId, /*bRequireVisible=*/true))
		{
			uint8 B[16];
			Instance->FGuidToBytesLE(ArenaId, B);
			tongsim_lite::arena::LoadArenaResponse R;
			R.mutable_arena_id()->set_guid(reinterpret_cast<const char*>(B), 16);

			UTSGrpcSubsystem::GetInstance()->RefreshActorMappings();

			this->writeAndFinish(R);
			Instance->LoadReactors.Remove(ArenaId);
			Instance->BusyArenas.Remove(ArenaId);
			return;
		}
	}

	if (Elapsed >= Deadline)
	{
		Instance->LoadReactors.Remove(ArenaId);
		Instance->BusyArenas.Remove(ArenaId);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "LoadArena timeout."));
	}
}

// ---------- Reactor: ResetArena ----------
void UArenaGrpcSubsystem::FResetArenaReactor::onRequest(tongsim_lite::arena::ResetArenaRequest& Req)
{
	if (!Instance->ObjectIdToGuid(Req.arena_id(), ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id"));
		return;
	}

	// 同一 Arena 跨类型互斥
	if (Instance->BusyArenas.Contains(ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "Another operation in this arena is in progress."));
		return;
	}

	if (auto* S = Mgr())
	{
		if (ULevelStreamingDynamic* Old = S->GetStreaming(ArenaId).Get())
		{
			GArena_OldStreaming.Add(ArenaId, Old);
		}
		if (ULevel* OldLevel = S->GetArenaULevel(ArenaId))
		{
			GArena_OldLevel.Add(ArenaId, OldLevel);
		}
		GArena_FlushAccum.Add(ArenaId, 0.f);
		GArena_DidGC.Remove(ArenaId);


		if (!S->ResetArena(ArenaId))
		{
			this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "Arena not found or reset failed"));
			return;
		}

		Instance->BusyArenas.Add(ArenaId);
		Instance->ResetReactors.Add(ArenaId, this->sharedSelf<FResetArenaReactor>());

		Deadline = Instance->AsyncGrpcDeadline;
		Elapsed = 0.f;
		return;
	}
	this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem"));
}

void UArenaGrpcSubsystem::FResetArenaReactor::onCancel()
{
	Instance->ResetReactors.Remove(ArenaId);
	Instance->BusyArenas.Remove(ArenaId);
	ClearArenaUnloadState(ArenaId); // [ADD]
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "ResetArena cancelled."));
}

void UArenaGrpcSubsystem::FResetArenaReactor::Tick(float dt)
{
	Elapsed += dt;

	if (auto* S = Mgr())
	{
		const bool bNewReady = S->IsArenaReady(ArenaId, /*bRequireVisible=*/true);
		const bool bOldGone = IsOldArenaFullyUnloaded(ArenaId);

		// [MOD] —— “双门闸”：新 Ready 且 旧彻底卸载
		if (bNewReady && bOldGone)
		{
			// 可选：收尾 GC（只做一次）
			MaybeDoOneGC(ArenaId);

			UTSGrpcSubsystem::GetInstance()->RefreshActorMappings();

			tongsim_lite::common::Empty E;
			this->writeAndFinish(E);

			Instance->ResetReactors.Remove(ArenaId);
			Instance->BusyArenas.Remove(ArenaId);
			ClearArenaUnloadState(ArenaId);
			return;
		}

		// [ADD] 等待期间偶尔 Flush，推动卸载/加载
		MaybeFlushStreaming(GetArenaWorld(), ArenaId, dt);
	}

	if (Elapsed >= Deadline)
	{
		Instance->ResetReactors.Remove(ArenaId);
		Instance->BusyArenas.Remove(ArenaId);
		ClearArenaUnloadState(ArenaId);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "ResetArena timeout."));
	}
}

// ---------- Reactor: DestroyArena ----------
void UArenaGrpcSubsystem::FDestroyArenaReactor::onRequest(tongsim_lite::arena::DestroyArenaRequest& Req)
{
	if (!Instance->ObjectIdToGuid(Req.arena_id(), ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id"));
		return;
	}

	if (Instance->BusyArenas.Contains(ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "Another operation in this arena is in progress."));
		return;
	}

	if (auto* S = Mgr())
	{
		if (ULevelStreamingDynamic* Old = S->GetStreaming(ArenaId).Get())
		{
			GArena_OldStreaming.Add(ArenaId, Old);
		}
		if (ULevel* OldLevel = S->GetArenaULevel(ArenaId))
		{
			GArena_OldLevel.Add(ArenaId, OldLevel);
		}
		GArena_FlushAccum.Add(ArenaId, 0.f);
		GArena_DidGC.Remove(ArenaId);

		if (!S->DestroyArena(ArenaId))
		{
			this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "Arena not found"));
			return;
		}

		Instance->BusyArenas.Add(ArenaId);
		Instance->DestroyReactors.Add(ArenaId, this->sharedSelf<FDestroyArenaReactor>());

		Deadline = Instance->AsyncGrpcDeadline;
		Elapsed = 0.f;
		return;
	}
	this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem"));
}

void UArenaGrpcSubsystem::FDestroyArenaReactor::onCancel()
{
	Instance->DestroyReactors.Remove(ArenaId);
	Instance->BusyArenas.Remove(ArenaId);
	ClearArenaUnloadState(ArenaId); // [ADD]
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "DestroyArena cancelled."));
}

void UArenaGrpcSubsystem::FDestroyArenaReactor::Tick(float dt)
{
	Elapsed += dt;

	if (auto* S = Mgr())
	{
		// [MOD] —— 等“旧关卡彻底卸载”，不再以 !IsArenaReady 作为唯一完成条件
		if (IsOldArenaFullyUnloaded(ArenaId))
		{
			// 可选：收尾 GC（只做一次）
			MaybeDoOneGC(ArenaId);

			UTSGrpcSubsystem::GetInstance()->RefreshActorMappings();

			tongsim_lite::common::Empty E;
			this->writeAndFinish(E);

			Instance->DestroyReactors.Remove(ArenaId);
			Instance->BusyArenas.Remove(ArenaId);
			ClearArenaUnloadState(ArenaId);
			return;
		}

		// [ADD] 等待期间偶尔 Flush，推动卸载（避免每帧刷）
		MaybeFlushStreaming(GetArenaWorld(), ArenaId, dt);
	}

	if (Elapsed >= Deadline)
	{
		Instance->DestroyReactors.Remove(ArenaId);
		Instance->BusyArenas.Remove(ArenaId);
		ClearArenaUnloadState(ArenaId);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "DestroyArena timeout."));
	}
}

// ---------- Unary: DestroyActorInArena ----------
tongos::ResponseStatus UArenaGrpcSubsystem::DestroyActorInArena(
	tongsim_lite::arena::DestroyActorInArenaRequest& Req, tongsim_lite::common::Empty&)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");

	AActor* Actor = nullptr;
	if (UTSGrpcSubsystem* G = UTSGrpcSubsystem::GetInstance())
	{
		FGuid Aid;
		if (ObjectIdToGuid(Req.actor_id(), Aid))
			if (const TWeakObjectPtr<AActor>* Found = G->GetIdToActorMap().Find(Aid))
				Actor = Found->Get();
	}
	if (!IsValid(Actor)) return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found");

	if (auto* S = Mgr())
	{
		if (!S->IsActorInArena(ArenaId, Actor))
			return ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "Actor not in the arena");

		Actor->Destroy(true);
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

// ---------- Reactor: SimpleMoveTowardsInArena ----------

void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::onRequest(
	tongsim_lite::arena::SimpleMoveTowardsInArenaRequest& Req)
{
	if (!Instance->ObjectIdToGuid(Req.arena_id(), ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id"));
		return;
	}

	// 同一 Arena 互斥（沿用你并发改造后的 BusyArenas）
	if (Instance->BusyArenas.Contains(ArenaId))
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "Another operation in this arena is in progress."));
		return;
	}

	// 选择控制 Pawn：Arena 内唯一 RL_Agent（与原逻辑一致）
	UWorld* World = GetArenaWorld();
	if (!World)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UWorld"));
		return;
	}
	TArray<AActor*> Actors;
	UGameplayStatics::GetAllActorsWithTag(World, FName(TEXT("RL_Agent")), Actors);
	if (auto* S = Mgr())
	{
		for (AActor* A : Actors)
		{
			if (APawn* P = Cast<APawn>(A))
			{
				if (S->IsActorInArena(ArenaId, P))
				{
					ControlledPawn = P;
					break;
				}
			}
		}
	}
	if (!ControlledPawn.IsValid())
	{
		this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "No RL_Agent pawn in arena."));
		return;
	}

	// 计算 World 目标（Arena-Local -> World）
	if (auto* S = Mgr())
	{
		FTransform Local(FRotator::ZeroRotator,
		                 FVector(Req.target_local_location().x(), Req.target_local_location().y(), Req.target_local_location().z()),
		                 FVector(1, 1, 1));
		FTransform WorldXf;
		if (!S->LocalToWorld(ArenaId, Local, WorldXf))
		{
			this->finish(ResponseStatus(grpc::StatusCode::UNKNOWN, "LocalToWorld failed"));
			return;
		}
		Target = WorldXf.GetLocation();
	}

	// 读取朝向控制（GIVEN 仅应用一次；FACE_MOVEMENT 持续朝向移动方向）
	OrientationMode = (int32)Req.orientation_mode();
	bGivenOrientationValid = false;
	bGivenApplied = false;
	if (Req.orientation_mode() == tongsim_lite::arena::SimpleMoveTowardsInArenaRequest::ORIENTATION_GIVEN)
	{
		GivenForwardXY = FVector2D(Req.given_forward().x(), Req.given_forward().y()).GetSafeNormal();
		bGivenOrientationValid = GivenForwardXY.IsNearlyZero() == false;
	}

	Elapsed = 0.f;
	bHitSomething = false;

	// —— 起点即到达（XY 平面判定，与 DemoRL 一致） ——
	if (APawn* Pawn = ControlledPawn.Get())
	{
		const FVector Curr = Pawn->GetActorLocation();
		const double Dist2 = FVector::DistSquaredXY(Target, Curr);
		if (Dist2 <= (ToleranceUU * ToleranceUU))
		{
			if (OrientationMode == 2 /*GIVEN*/ && bGivenOrientationValid && !bGivenApplied)
			{
				ApplyGivenOrientationOnce();
			}
			WriteAndFinishResponse();
			return; // 不登记 Busy/Map，直接完成
		}
	}

	// —— 登记并发与互斥 ——
	Instance->BusyArenas.Add(ArenaId);
	Instance->MoveReactors.Add(ArenaId, this->sharedSelf<FSimpleMoveTowardsInArenaReactor>());
}

void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::onCancel()
{
	Instance->MoveReactors.Remove(ArenaId);
	Instance->BusyArenas.Remove(ArenaId);
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "SimpleMoveTowardsInArena cancelled."));
}

void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::Tick(float dt)
{
	Elapsed += dt;

	APawn* Pawn = ControlledPawn.Get();
	if (!IsValid(Pawn))
	{
		Instance->MoveReactors.Remove(ArenaId);
		Instance->BusyArenas.Remove(ArenaId);
		this->finish(ResponseStatus(grpc::StatusCode::ABORTED, "Pawn lost."));
		return;
	}

	const FVector Curr = Pawn->GetActorLocation();
	const FVector Delta = Target - Curr;
	const double Dist2 = FVector::DistSquaredXY(Target, Curr);

	// —— 朝向处理（与 DemoRL 一致） ——
	if (OrientationMode == 2 /*GIVEN*/ && bGivenOrientationValid && !bGivenApplied)
	{
		ApplyGivenOrientationOnce();
	}
	else if (OrientationMode == 1 /*FACE_MOVEMENT*/ && !Delta.IsNearlyZero())
	{
		const FVector StepDirXY = FVector(Delta.X, Delta.Y, 0.0).GetSafeNormal();
		ApplyFaceMovementYaw(StepDirXY);
	}

	// —— 已到达 ——
	if (Dist2 <= (ToleranceUU * ToleranceUU))
	{
		WriteAndFinishResponse();
		return;
	}

	// —— 直线限速 ——
	const float StepLen = SpeedUUPerSec * FMath::Max(dt, 0.f);
	const FVector StepDirXY = FVector(Delta.X, Delta.Y, 0.0).GetSafeNormal();

	// —— 过冲钳制：若本帧步长将越过目标点，则直接到目标 XY（保持 Z） ——
	if ((double)StepLen * (double)StepLen >= Dist2)
	{
		FHitResult Hit;
		const FVector TargetXY(Target.X, Target.Y, Curr.Z);
		Pawn->SetActorLocation(TargetXY, /*bSweep=*/true, &Hit, ETeleportType::None);

		if (Hit.bBlockingHit)
		{
			AActor* HitActor = Hit.GetActor();
			if (!HitActor || !HitActor->ActorHasTag(FName(TEXT("RL_Floor"))))
			{
				bHitSomething = true;
				LastHit = Hit;
			}
		}
		WriteAndFinishResponse();
		return;
	}

	// —— 常规步进（Sweep 碰撞），命中时忽略 RL_Floor ——
	{
		FHitResult Hit;
		const FVector StepPoint = Curr + StepDirXY * StepLen;
		Pawn->SetActorLocation(StepPoint, /*bSweep=*/true, &Hit, ETeleportType::None);

		if (Hit.bBlockingHit)
		{
			AActor* HitActor = Hit.GetActor();
			if (!HitActor || !HitActor->ActorHasTag(FName(TEXT("RL_Floor"))))
			{
				bHitSomething = true;
				LastHit = Hit;
				WriteAndFinishResponse();
				return;
			}
		}
	}

	// —— 超时 ——
	if (Elapsed >= Instance->AsyncGrpcDeadline)
	{
		Instance->MoveReactors.Remove(ArenaId);
		Instance->BusyArenas.Remove(ArenaId);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "Move timeout."));
	}
}


void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::ApplyFaceMovementYaw(const FVector& StepDir)
{
	APawn* Pawn = ControlledPawn.Get();
	if (!IsValid(Pawn)) return;

	const double YawDeg = FMath::RadiansToDegrees(FMath::Atan2(StepDir.Y, StepDir.X));
	FRotator R = Pawn->GetActorRotation();
	R.Yaw = (float)YawDeg; // 仅改 Yaw
	Pawn->SetActorRotation(R);
}

void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::ApplyGivenOrientationOnce()
{
	APawn* Pawn = ControlledPawn.Get();
	if (!IsValid(Pawn)) return;

	if (!GivenForwardXY.IsNearlyZero())
	{
		const double YawDeg = FMath::RadiansToDegrees(FMath::Atan2(GivenForwardXY.Y, GivenForwardXY.X));
		FRotator R = Pawn->GetActorRotation();
		R.Yaw = (float)YawDeg; // 仅改 Yaw
		Pawn->SetActorRotation(R);
	}
	bGivenApplied = true;
}

void UArenaGrpcSubsystem::FSimpleMoveTowardsInArenaReactor::WriteAndFinishResponse()
{
	tongsim_lite::arena::SimpleMoveTowardsInArenaResponse R;

	APawn* Pawn = ControlledPawn.Get();
	const FVector Loc = IsValid(Pawn) ? Pawn->GetActorLocation() : FVector::ZeroVector;
	*R.mutable_current_location() = ToP(Loc);

	if (bHitSomething)
	{
		if (AActor* HitActor = LastHit.GetActor())
		{
			// —— 保持原 proto 字段：返回名称（若你后续扩展 proto，可在此追加 GUID/ObjectInfo）
			R.mutable_hit_result()->set_hit_actor(TCHAR_TO_UTF8(*HitActor->GetName()));
		}
	}

	this->writeAndFinish(R);
	Instance->MoveReactors.Remove(ArenaId);
	Instance->BusyArenas.Remove(ArenaId);
}


tongos::ResponseStatus UArenaGrpcSubsystem::SetArenaVisible(
	tongsim_lite::arena::SetArenaVisibleRequest& Req, tongsim_lite::common::Empty&)
{
	FGuid Id;
	if (!ObjectIdToGuid(Req.arena_id(), Id)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");
	if (auto* S = Mgr()) return S->SetArenaVisible(Id, Req.visible()) ? ResponseStatus::OK : ResponseStatus(grpc::StatusCode::NOT_FOUND, "Arena not found");
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::ListArenas(
	tongsim_lite::arena::ListArenasRequest&, tongsim_lite::arena::ListArenasResponse& Resp)
{
	if (auto* S = Mgr())
	{
		TArray<FArenaDescriptor> Arr;
		S->GetArenas(Arr); // C++ 侧已提供汇总描述
		for (const FArenaDescriptor& D : Arr)
		{
			auto* O = Resp.add_arenas();
			uint8 Buf[16];
			FGuidToBytesLE(D.Id, Buf);
			O->mutable_arena_id()->set_guid(reinterpret_cast<const char*>(Buf), 16);
			O->set_asset_path(TCHAR_TO_UTF8(*D.AssetPath));
			*O->mutable_anchor() = ToProtoXf(D.Anchor);
			O->set_is_loaded(D.bIsLoaded);
			O->set_is_visible(D.bIsVisible);
			O->set_num_actors(D.NumActors);
		}
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::SpawnActorInArena(
	tongsim_lite::arena::SpawnActorInArenaRequest& Req, tongsim_lite::arena::SpawnActorInArenaResponse& Resp)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");

	if (auto* S = Mgr())
	{
		const FSoftClassPath ClsPath(UTF8_TO_TCHAR(Req.class_path().c_str()));
		const FTransform Local = FromProtoXf(Req.local_transform());
		AActor* A = S->SpawnActorInArenaByPath(ArenaId, ClsPath, Local);
		if (!IsValid(A)) return ResponseStatus(grpc::StatusCode::UNKNOWN, "SpawnActorInArena failed");

		// 用 DemoRLSubsystem 的做法从 UTSGrpcSubsystem 取 GUID 并回填 ObjectInfo
		if (UTSGrpcSubsystem* G = UTSGrpcSubsystem::GetInstance())
		{
			const FGuid Gid = G->FindGuidByActor(A);
			uint8 B[16];
			FGuidToBytesLE(Gid, B);
			auto* Info = Resp.mutable_actor();
			Info->mutable_id()->set_guid(reinterpret_cast<const char*>(B), 16);
			Info->set_name(TCHAR_TO_UTF8(*A->GetName()));
			Info->set_class_path(TCHAR_TO_UTF8(*A->GetClass()->GetPathName()));
		}
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::SetActorPoseLocal(
	tongsim_lite::arena::SetActorPoseLocalRequest& Req, tongsim_lite::common::Empty&)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");
	// 复用 DemoRL 的 ObjectId->Actor 映射思路
	AActor* Actor = nullptr;
	if (UTSGrpcSubsystem* G = UTSGrpcSubsystem::GetInstance())
	{
		FGuid Aid;
		if (ObjectIdToGuid(Req.actor_id(), Aid))
			if (const TWeakObjectPtr<AActor>* Found = G->GetIdToActorMap().Find(Aid))
				Actor = Found->Get();
	}
	if (!IsValid(Actor))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found");
	}

	if (auto* S = Mgr())
	{
		const FTransform Local = FromProtoXf(Req.local_transform());
		return S->SetActorPoseLocal(ArenaId, Actor, Local, Req.reset_physics())
			       ? ResponseStatus::OK
			       : ResponseStatus(grpc::StatusCode::UNKNOWN, "SetActorPoseLocal failed");
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::GetActorPoseLocal(
	tongsim_lite::arena::GetActorPoseLocalRequest& Req, tongsim_lite::arena::GetActorPoseLocalResponse& Resp)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");

	AActor* Actor = nullptr;
	if (UTSGrpcSubsystem* G = UTSGrpcSubsystem::GetInstance())
	{
		FGuid Aid;
		if (ObjectIdToGuid(Req.actor_id(), Aid))
			if (const TWeakObjectPtr<AActor>* Found = G->GetIdToActorMap().Find(Aid))
				Actor = Found->Get();
	}
	if (!IsValid(Actor)) return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found");

	if (auto* S = Mgr())
	{
		FTransform Local;
		if (!S->GetActorPoseLocal(ArenaId, Actor, Local))
			return ResponseStatus(grpc::StatusCode::UNKNOWN, "GetActorPoseLocal failed");
		*Resp.mutable_local_transform() = ToProtoXf(Local);
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::LocalToWorld(
	tongsim_lite::arena::LocalToWorldRequest& Req, tongsim_lite::arena::LocalToWorldResponse& Resp)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");
	if (auto* S = Mgr())
	{
		FTransform World;
		if (!S->LocalToWorld(ArenaId, FromProtoXf(Req.local()), World))
			return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Arena not found");
		*Resp.mutable_world() = ToProtoXf(World);
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}

tongos::ResponseStatus UArenaGrpcSubsystem::WorldToLocal(
	tongsim_lite::arena::WorldToLocalRequest& Req, tongsim_lite::arena::WorldToLocalResponse& Resp)
{
	FGuid ArenaId;
	if (!ObjectIdToGuid(Req.arena_id(), ArenaId)) return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Bad arena_id");
	if (auto* S = Mgr())
	{
		FTransform Local;
		if (!S->WorldToLocal(ArenaId, FromProtoXf(Req.world()), Local))
			return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Arena not found");
		*Resp.mutable_local() = ToProtoXf(Local);
		return ResponseStatus::OK;
	}
	return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No UTSArenaSubsystem");
}
