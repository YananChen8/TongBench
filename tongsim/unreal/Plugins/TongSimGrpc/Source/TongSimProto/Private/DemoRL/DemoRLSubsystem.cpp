// DemoRLSubsystem.cpp

#include "DemoRL/DemoRLSubsystem.h"

#include "TSGrpcSubsystem.h"
#include "TSVoxelGridFuncLib.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/LevelStreaming.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#if ENABLE_DRAW_DEBUG
#include "DrawDebugHelpers.h"
#endif
#include "GameFramework/Pawn.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/SoftObjectPath.h"
#include "NavigationSystem.h"
#include "NavigationPath.h"
#include "AI/Navigation/NavigationTypes.h"
#include "NavFilters/NavigationQueryFilter.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "AIController.h"
#include "AITypes.h"
#include "Navigation/PathFollowingComponent.h"
#include "Character/TSItemInteractComponent.h"

using namespace tongos;

UDemoRLSubsystem* UDemoRLSubsystem::Instance = nullptr;

namespace DemoRLServiceHelpers
{
	/* ---------- GUID <-> bytes (LE) ---------- */

	void FGuidToBytesLE(const FGuid& G, uint8 Out[16])
	{
		const uint32 Parts[4] = {
			static_cast<uint32>(G.A), static_cast<uint32>(G.B),
			static_cast<uint32>(G.C), static_cast<uint32>(G.D)
		};
		for (int i = 0; i < 4; ++i)
		{
			const uint32 v = Parts[i];
			Out[i * 4 + 0] = static_cast<uint8>(v & 0xFF);
			Out[i * 4 + 1] = static_cast<uint8>((v >> 8) & 0xFF);
			Out[i * 4 + 2] = static_cast<uint8>((v >> 16) & 0xFF);
			Out[i * 4 + 3] = static_cast<uint8>((v >> 24) & 0xFF);
		}
	}

	bool BytesLEToFGuid(const uint8 In[16], FGuid& Out)
	{
		auto ReadLE32 = [](const uint8* p) -> uint32
		{
			return (uint32)p[0] | ((uint32)p[1] << 8) | ((uint32)p[2] << 16) | ((uint32)p[3] << 24);
		};
		const uint32 A = ReadLE32(In + 0);
		const uint32 B = ReadLE32(In + 4);
		const uint32 C = ReadLE32(In + 8);
		const uint32 D = ReadLE32(In + 12);
		Out = FGuid((int32)A, (int32)B, (int32)C, (int32)D);
		return Out.IsValid();
	}

	/* ---------- World & Level ---------- */

	UWorld* GetGameWorld()
	{
		if (!GEngine) return nullptr;
		for (const FWorldContext& Ctx : GEngine->GetWorldContexts())
		{
			if (Ctx.WorldType == EWorldType::Game || Ctx.WorldType == EWorldType::PIE)
			{
				return Ctx.World();
			}
		}
		return nullptr;
	}

	bool IsWorldFullyLoaded(UWorld* World)
	{
		if (!World) return false;

		if (!World->PersistentLevel || !World->PersistentLevel->bIsVisible)
			return false;

		const TArray<ULevelStreaming*>& StreamingLevels = World->GetStreamingLevels();
		for (const ULevelStreaming* LS : StreamingLevels)
		{
			if (!LS) continue;
			if (!LS->HasLoadedLevel() || !LS->IsLevelVisible())
				return false;
		}

		if (!World->AreActorsInitialized())
			return false;

		return true;
	}

	/* ---------- Proto <-> UE Types ---------- */

	FVector FromProtoVector3f(const tongsim_lite::common::Vector3f& V)
	{
		return FVector(static_cast<double>(V.x()),
		               static_cast<double>(V.y()),
		               static_cast<double>(V.z()));
	}

	tongsim_lite::common::Vector3f ToProtoVector3f(const FVector& V)
	{
		tongsim_lite::common::Vector3f P;
		P.set_x(static_cast<float>(V.X));
		P.set_y(static_cast<float>(V.Y));
		P.set_z(static_cast<float>(V.Z));
		return P;
	}

	FRotator FromProtoRotatorf(const tongsim_lite::common::Rotatorf& R)
	{
		// UE: Pitch(Y), Yaw(Z), Roll(X) (单位：度)
		return FRotator(R.pitch_deg(), R.yaw_deg(), R.roll_deg());
	}

	tongsim_lite::common::Rotatorf ToProtoRotatorf(const FRotator& R)
	{
		tongsim_lite::common::Rotatorf Out;
		Out.set_roll_deg(R.Roll);
		Out.set_pitch_deg(R.Pitch);
		Out.set_yaw_deg(R.Yaw);
		return Out;
	}

	FTransform FromProtoTransform(const tongsim_lite::common::Transform& T)
	{
		const FVector Loc = FromProtoVector3f(T.location());
		const FRotator Rot = FromProtoRotatorf(T.rotation());
		const FVector Scl = FromProtoVector3f(T.scale());
		return FTransform(Rot, Loc, Scl);
	}

	tongsim_lite::common::Transform ToProtoTransform(const FTransform& T)
	{
		tongsim_lite::common::Transform Out;
		*Out.mutable_location() = ToProtoVector3f(T.GetLocation());
		*Out.mutable_rotation() = ToProtoRotatorf(T.Rotator());
		*Out.mutable_scale() = ToProtoVector3f(T.GetScale3D());
		return Out;
	}

	/* ---------- Object & State ---------- */

	void FillObjectInfo(const FGuid& Guid, const AActor* Actor, tongsim_lite::object::ObjectInfo& OutInfo)
	{
		uint8 GuidBytes[16] = {0};
		FGuidToBytesLE(Guid, GuidBytes);
		OutInfo.mutable_id()->set_guid(reinterpret_cast<const char*>(GuidBytes), 16);

		const FString NameStr = Actor ? Actor->GetName() : TEXT("None");
		OutInfo.set_name(TCHAR_TO_UTF8(*NameStr));

		const UClass* Cls = Actor ? Actor->GetClass() : nullptr;
		const FString ClassPath = Cls ? Cls->GetPathName() : TEXT("None");
		OutInfo.set_class_path(TCHAR_TO_UTF8(*ClassPath));
	}

	void FillActorState(const FGuid& Guid, const AActor* Actor, tongsim_lite::demo_rl::ActorState& OutState)
	{
		if (!Actor) return;

		FillObjectInfo(Guid, Actor, *OutState.mutable_object_info());

		if (Actor->IsA(AInfo::StaticClass()))
		{
			return;
		}

		const FVector Loc = Actor->GetActorLocation();
		const FVector Fwd = Actor->GetActorForwardVector().GetSafeNormal();
		const FVector Rgt = Actor->GetActorRightVector().GetSafeNormal();

		*OutState.mutable_location() = ToProtoVector3f(Loc);
		*OutState.mutable_unit_forward_vector() = ToProtoVector3f(Fwd);
		*OutState.mutable_unit_right_vector() = ToProtoVector3f(Rgt);

		const FBox B = Actor->GetComponentsBoundingBox(/*bNonColliding=*/true);
		auto* Box = OutState.mutable_bounding_box();
		*Box->mutable_min_vertex() = ToProtoVector3f(B.Min);
		*Box->mutable_max_vertex() = ToProtoVector3f(B.Max);

		FString TagStr;
		if (Actor->Tags.Num() > 0)
		{
			TagStr = Actor->Tags[0].ToString();
		}
		OutState.set_tag(TCHAR_TO_UTF8(*TagStr));

		float Speed = Actor->GetVelocity().Size();
		OutState.set_current_speed(Speed);
	}

	bool ObjectIdToGuid(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid)
	{
		const std::string& Buf = Id.guid();
		if (Buf.size() != 16) return false;
		return BytesLEToFGuid(reinterpret_cast<const uint8*>(Buf.data()), OutGuid);
	}

	AActor* FindActorByObjectId(const tongsim_lite::object::ObjectId& Id)
	{
		UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance();
		if (!GrpcSubsystem) return nullptr;

		FGuid G;
		if (!ObjectIdToGuid(Id, G)) return nullptr;

		const TMap<FGuid, TWeakObjectPtr<AActor>>& Map = GrpcSubsystem->GetIdToActorMap();
		if (const TWeakObjectPtr<AActor>* Found = Map.Find(G))
		{
			return Found->Get();
		}
		return nullptr;
	}

	/* ---------- Tag Conveniences ---------- */
	FName RLAgentName = FName(TEXT("RL_Agent"));
	FName RLFloorName = FName(TEXT("RL_Floor"));
}

void UDemoRLSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	FWorldDelegates::OnPostWorldInitialization.AddUObject(this, &ThisClass::HandlePostWorldInit);
	Instance = this;
}

void UDemoRLSubsystem::Deinitialize()
{
	Instance = nullptr;
	FWorldDelegates::OnPostWorldInitialization.RemoveAll(this);
	Super::Deinitialize();
}

void UDemoRLSubsystem::Tick(float DeltaTime)
{
	if (ResetLevelReactorPtr)
	{
		ResetLevelReactorPtr->Tick(DeltaTime);
	}

	{
		// SimpleMove
		TArray<FGuid> Keys;
		SimpleMoveReactorMap.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = SimpleMoveReactorMap.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}

	{
		// NavMove
		TArray<FGuid> Keys;
		NavMoveReactorMap.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = NavMoveReactorMap.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}

	{
		// PickUp
		TArray<FGuid> Keys;
		PickUpReactorMap.GetKeys(Keys);
		for (const FGuid& K : Keys)
			if (auto* SP = PickUpReactorMap.Find(K)) if (*SP) (*SP)->Tick(DeltaTime);
	}
}

void UDemoRLSubsystem::HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues IVS)
{
	UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance();
	if (!IsValid(GrpcSubsystem))
	{
		return;
	}

	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/QueryState", &ThisClass::QueryState);
	GrpcSubsystem->RegisterReactor<ThisClass::FResetLevelReactor>("/tongsim_lite.demo_rl.DemoRLService/ResetLevel");
	GrpcSubsystem->RegisterReactor<ThisClass::FSimpleMoveTowardsReactor>("/tongsim_lite.demo_rl.DemoRLService/SimpleMoveTowards");

	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/GetActorState", &ThisClass::GetActorState);
	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/GetActorTransform", &ThisClass::GetActorTransform);
	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/SetActorTransform", &ThisClass::SetActorTransform);
	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/SpawnActor", &ThisClass::SpawnActor);

	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.voxel.VoxelService/QueryVoxel", &ThisClass::QueryVoxel);

	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/ExecConsoleCommand", &ThisClass::ExecConsoleCommand);
	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/QueryNavigationPath", &ThisClass::QueryNavigationPath);
	GrpcSubsystem->RegisterReactor<ThisClass::FNavigateToLocationReactor>("/tongsim_lite.demo_rl.DemoRLService/NavigateToLocation");
	GrpcSubsystem->RegisterReactor<ThisClass::FPickUpObjectReactor>("/tongsim_lite.demo_rl.DemoRLService/PickUpObject");
	GrpcSubsystem->RegisterReactor<ThisClass::FDropObjectReactor>("/tongsim_lite.demo_rl.DemoRLService/DropObject");

	GrpcSubsystem->RegisterUnaryHandler("/tongsim_lite.demo_rl.DemoRLService/DestroyActor", &ThisClass::DestroyActor);

	GrpcSubsystem->RegisterUnaryHandler(
		"/tongsim_lite.demo_rl.DemoRLService/BatchSingleLineTraceByObject",
		&ThisClass::BatchSingleLineTraceByObject);

	GrpcSubsystem->RegisterUnaryHandler(
	"/tongsim_lite.demo_rl.DemoRLService/BatchMultiLineTraceByObject",
	&ThisClass::BatchMultiLineTraceByObject);
}

/* ---------- Unary Handlers ---------- */

tongos::ResponseStatus UDemoRLSubsystem::QueryState(
	tongsim_lite::common::Empty& /*Request*/,
	tongsim_lite::demo_rl::DemoRLState& Response)
{
	UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance();
	if (!GrpcSubsystem)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid TongSim gRPC Subsystem.");
	}

	const TMap<FGuid, TWeakObjectPtr<AActor>>& IdToActorMap = GrpcSubsystem->GetIdToActorMap();
	const TSet<FGuid>& DestroyedIds = GrpcSubsystem->GetDestroyedIds();

	for (const auto& Pair : IdToActorMap)
	{
		const FGuid Guid = Pair.Key;
		if (Guid.IsValid())
		{
			if (AActor* Actor = Pair.Value.Get())
			{
				auto* Out = Response.add_actor_states();
				DemoRLServiceHelpers::FillActorState(Guid, Actor, *Out);
			}
			else if (DestroyedIds.Contains(Guid))
			{
				auto* Out = Response.add_actor_states();
				// 仅填 ObjectInfo，名称/类路径会落到 "None"
				DemoRLServiceHelpers::FillObjectInfo(Guid, /*Actor=*/nullptr, *Out->mutable_object_info());
				Out->set_destroyed(true); // 新增字段
			}
		}
	}
	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::GetActorState(
	tongsim_lite::demo_rl::GetActorStateRequest& Request,
	tongsim_lite::demo_rl::GetActorStateResponse& Response)
{
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(Request.actor_id());
	if (!IsValid(Actor))
	{
		return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found.");
	}

	UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance();
	if (!GrpcSubsystem)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid TongSim gRPC Subsystem.");
	}

	const FGuid Guid = GrpcSubsystem->FindGuidByActor(Actor);
	if (!Guid.IsValid())
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNKNOWN, "Actor GUID not registered.");
	}

	DemoRLServiceHelpers::FillActorState(Guid, Actor, *Response.mutable_actor_state());
	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::GetActorTransform(
	tongsim_lite::demo_rl::GetActorTransformRequest& Request,
	tongsim_lite::demo_rl::GetActorTransformResponse& Response)
{
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(Request.actor_id());
	if (!IsValid(Actor))
	{
		return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found.");
	}

	const FTransform T = Actor->GetActorTransform();
	*Response.mutable_transform() = DemoRLServiceHelpers::ToProtoTransform(T);
	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::SetActorTransform(
	tongsim_lite::demo_rl::SetActorTransformRequest& Request,
	tongsim_lite::common::Empty& /*Response*/)
{
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(Request.actor_id());
	if (!IsValid(Actor))
	{
		return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found.");
	}

	const FTransform T = DemoRLServiceHelpers::FromProtoTransform(Request.transform());
	Actor->SetActorTransform(T, /*bSweep=*/false, /*OutHitResult=*/nullptr, ETeleportType::TeleportPhysics);
	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::SpawnActor(
	tongsim_lite::demo_rl::SpawnActorRequest& Request,
	tongsim_lite::demo_rl::SpawnActorResponse& Response)
{
	UWorld* World = Instance ? Instance->GetWorld() : DemoRLServiceHelpers::GetGameWorld();
	if (!World)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld.");
	}

	const FString BlueprintPath = UTF8_TO_TCHAR(Request.blueprint().c_str());

	UClass* ActorClass = LoadClass<AActor>(nullptr, *BlueprintPath);
	if (!ActorClass)
	{
		return tongos::ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Failed to load class from blueprint path.");
	}

	FTransform SpawnTransform = DemoRLServiceHelpers::FromProtoTransform(Request.transform());

	FActorSpawnParameters Params;
	Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
	if (Request.has_name())
	{
		const FString DesiredName = UTF8_TO_TCHAR(Request.name().c_str());
		Params.Name = FName(DesiredName);
	}

	AActor* NewActor = World->SpawnActor<AActor>(ActorClass, SpawnTransform, Params);
	if (!IsValid(NewActor))
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNKNOWN, "SpawnActor failed.");
	}

	// 追加标签
	for (const std::string& TagStrUtf8 : Request.tags())
	{
		const FString TagStr = UTF8_TO_TCHAR(TagStrUtf8.c_str());
		const FName TagName(*TagStr);
		if (!TagName.IsNone())
		{
			NewActor->Tags.AddUnique(TagName);
		}
	}

	// 返回 ObjectInfo
	UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance();
	FGuid Guid;
	if (GrpcSubsystem)
	{
		Guid = GrpcSubsystem->FindGuidByActor(NewActor);
	}
	// 若无可用 GUID，也生成一个零 GUID 占位（不会崩，但建议确保有注册）
	DemoRLServiceHelpers::FillObjectInfo(Guid, NewActor, *Response.mutable_actor());

	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::QueryVoxel(
	tongsim_lite::voxel::QueryVoxelRequest& Request, tongsim_lite::voxel::Voxel& Response)
{
	UWorld* World = Instance ? Instance->GetWorld() : DemoRLServiceHelpers::GetGameWorld();
	if (!World)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld.");
	}

	FTransform QueryTransform = DemoRLServiceHelpers::FromProtoTransform(Request.transform());

	if (Request.voxel_num_x() % 2 != 0 || Request.voxel_num_y() % 2 != 0 || Request.voxel_num_z() % 2 != 0)
	{
		return tongos::ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Voxel num must be even.");
	}

	const uint16 VoxelHalfNumX = Request.voxel_num_x() / 2;
	const uint16 VoxelHalfNumY = Request.voxel_num_y() / 2;
	const uint16 VoxelHalfNumZ = Request.voxel_num_z() / 2;

	const FVector Extent = DemoRLServiceHelpers::FromProtoVector3f(Request.extent());

	FVoxelGridQueryParam QueryParam{World};
	UGameplayStatics::GetAllActorsOfClass(World, AActor::StaticClass(), QueryParam.Actors);

	for (const tongsim_lite::object::ObjectId& ActorId_Proto : Request.actorstoignore())
	{
		if (AActor* ActorToIgnore = DemoRLServiceHelpers::FindActorByObjectId(ActorId_Proto))
		{
			QueryParam.Actors.Remove(ActorToIgnore);
		}
	}

	QueryParam.GridBox = FVoxelBox{
		QueryTransform, VoxelHalfNumX, VoxelHalfNumY, VoxelHalfNumZ, Extent * 2.f
	};
	TArray<uint8> VoxelGrids;
	TSVoxelGridFuncLib::QueryVoxelGrids(QueryParam, VoxelGrids, World);

	Response.set_voxel_buffer(static_cast<uint8*>(VoxelGrids.GetData()), VoxelGrids.Num());

	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::ExecConsoleCommand(
	tongsim_lite::demo_rl::ExecConsoleCommandRequest& Request,
	tongsim_lite::demo_rl::ExecConsoleCommandResponse& Response)
{
	UWorld* World = Instance ? Instance->GetWorld() : DemoRLServiceHelpers::GetGameWorld();
	if (!World)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld.");
	}

	const FString Cmd = UTF8_TO_TCHAR(Request.command().c_str());
	const bool bWriteToLog = Request.write_to_log();

	bool bSuccess = false;

	if (APlayerController* PC = UGameplayStatics::GetPlayerController(World, 0))
	{
		PC->ConsoleCommand(Cmd, bWriteToLog);
		bSuccess = true;
	}
	else if (GEngine)
	{
		bSuccess = GEngine->Exec(World, *Cmd);
	}

	Response.set_success(bSuccess);
	Response.set_message(TCHAR_TO_UTF8(*FString::Printf(TEXT("Executed: %s"), *Cmd)));

	return bSuccess
		       ? tongos::ResponseStatus::OK
		       : tongos::ResponseStatus(grpc::StatusCode::UNKNOWN, "Console command execution failed.");
}


tongos::ResponseStatus UDemoRLSubsystem::QueryNavigationPath(
	tongsim_lite::demo_rl::QueryNavigationPathRequest& Request,
	tongsim_lite::demo_rl::QueryNavigationPathResponse& Response)
{
	UWorld* World = Instance ? Instance->GetWorld() : DemoRLServiceHelpers::GetGameWorld();
	if (!World)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld.");
	}

	UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
	if (!NavSys)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No NavigationSystem.");
	}

	const bool bAllowPartial = Request.allow_partial();
	const bool bRequireNavigableEnd = Request.require_navigable_end_location();
	const float CostLimit = Request.cost_limit();

	FVector Start = DemoRLServiceHelpers::FromProtoVector3f(Request.start());
	FVector End = DemoRLServiceHelpers::FromProtoVector3f(Request.end());

	// 如果要求终点必须是可导航位置，则进行投影（投影失败直接返回 NOT_FOUND）
	if (bRequireNavigableEnd)
	{
		FNavLocation ProjectedEnd;
		if (!NavSys->ProjectPointToNavigation(End, ProjectedEnd, FVector(100.f, 100.f, 300.f)))
		{
			return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "End location is not navigable.");
		}
		End = ProjectedEnd.Location;
	}

	// 默认导航数据
	ANavigationData* NavData = NavSys->GetDefaultNavDataInstance(FNavigationSystem::DontCreate);
	if (!NavData)
	{
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No NavData.");
	}

	FSharedConstNavQueryFilter QueryFilter = UNavigationQueryFilter::GetQueryFilter(*NavData, nullptr, nullptr);
	FPathFindingQuery Query(nullptr, *NavData, Start, End, QueryFilter);
	Query.SetAllowPartialPaths(bAllowPartial);
	if (CostLimit > 0.f) { Query.CostLimit = CostLimit; }
	const FPathFindingResult Result = NavSys->FindPathSync(Query, EPathFindingMode::Regular);
	if (!Result.IsSuccessful() || !Result.Path.IsValid())
	{
		return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Path not found.");
	}

	const FNavPathSharedPtr& Path = Result.Path;
	Response.set_is_partial(Path->IsPartial());

	// Path points
	const TArray<FNavPathPoint>& Points = Path->GetPathPoints();
	for (const FNavPathPoint& P : Points)
	{
		*Response.add_path_points() = DemoRLServiceHelpers::ToProtoVector3f(P.Location);
	}

	// Cost（若可得）
	if (Result.IsSuccessful())
	{
		Response.set_path_cost(static_cast<float>(Path->GetCost()));
	}

	// Length（逐段累加）
	double Length = 0.0;
	for (int32 i = 1; i < Points.Num(); ++i)
	{
		Length += FVector::Distance(Points[i - 1].Location, Points[i].Location);
	}
	Response.set_path_length(static_cast<float>(Length));

	return tongos::ResponseStatus::OK;
}

/* ---------- ResetLevel Reactor ---------- */

void UDemoRLSubsystem::FResetLevelReactor::onRequest(tongsim_lite::common::Empty&)
{
	if (Instance->ResetLevelReactorPtr)
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "ResetLevel is in progress."));
		return;
	}

	UWorld* World = Instance->GetWorld();
	if (!World)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld."));
		return;
	}

	const bool bRemovePIEPrefix = true;
	TargetLevel = FName(*UGameplayStatics::GetCurrentLevelName(World, bRemovePIEPrefix));
	if (TargetLevel.IsNone())
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Current level name is invalid."));
		return;
	}

	Instance->ResetLevelReactorPtr = this->sharedSelf<FResetLevelReactor>();
	TotalTime = 0.f;

	PreLoadHandle = FCoreUObjectDelegates::PreLoadMap.AddRaw(this, &UDemoRLSubsystem::FResetLevelReactor::OnPreLoadMap);
	PostLoadHandle = FCoreUObjectDelegates::PostLoadMapWithWorld.AddRaw(this, &UDemoRLSubsystem::FResetLevelReactor::OnPostLoadMapWithWorld);
	if (GEngine)
	{
		TravelFailHandle = GEngine->OnTravelFailure().AddRaw(
			this, &UDemoRLSubsystem::FResetLevelReactor::OnTravelFailure);
	}

	UGameplayStatics::OpenLevel(World, TargetLevel);
}

void UDemoRLSubsystem::FResetLevelReactor::onCancel()
{
	CleanupDelegates();
	Instance->ResetLevelReactorPtr.reset();
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "ResetLevel cancelled by client."));
}

void UDemoRLSubsystem::FResetLevelReactor::Tick(float DeltaTime)
{
	TotalTime += DeltaTime;

	if (bLoadStarted)
	{
		UWorld* WorldToCheck = NewWorld.IsValid() ? NewWorld.Get() : Instance->GetWorld();
		const bool bFullyLoaded = DemoRLServiceHelpers::IsWorldFullyLoaded(WorldToCheck);

		if (bFullyLoaded)
		{
			CleanupDelegates();
			const tongsim_lite::common::Empty Resp;
			this->writeAndFinish(Resp);
			Instance->ResetLevelReactorPtr.reset();
			return;
		}
	}

	if (TotalTime >= Instance->AsyncGrpcDeadline)
	{
		CleanupDelegates();
		const FString Msg = FString::Printf(TEXT("Reset level time out. Deadline %.1fs"), Instance->AsyncGrpcDeadline);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, TCHAR_TO_UTF8(*Msg)));
		Instance->ResetLevelReactorPtr.reset();
	}
}

void UDemoRLSubsystem::FResetLevelReactor::OnPreLoadMap(const FString& /*MapName*/)
{
	bLoadStarted = true;
}

void UDemoRLSubsystem::FResetLevelReactor::OnPostLoadMapWithWorld(UWorld* LoadedWorld)
{
	NewWorld = LoadedWorld;
}

void UDemoRLSubsystem::FResetLevelReactor::OnTravelFailure(UWorld* /*InWorld*/, ETravelFailure::Type /*FailureType*/, const FString& Error)
{
	CleanupDelegates();
	this->finish(ResponseStatus(grpc::StatusCode::UNKNOWN, TCHAR_TO_UTF8(*Error)));
	if (Instance) Instance->ResetLevelReactorPtr.reset();
}

void UDemoRLSubsystem::FResetLevelReactor::CleanupDelegates()
{
	if (PreLoadHandle.IsValid())
	{
		FCoreUObjectDelegates::PreLoadMap.Remove(PreLoadHandle);
		PreLoadHandle.Reset();
	}
	if (PostLoadHandle.IsValid())
	{
		FCoreUObjectDelegates::PostLoadMapWithWorld.Remove(PostLoadHandle);
		PostLoadHandle.Reset();
	}
	if (TravelFailHandle.IsValid() && GEngine)
	{
		GEngine->OnTravelFailure().Remove(TravelFailHandle);
		TravelFailHandle.Reset();
	}
}

/* ---------- SimpleMoveTowards Reactor ---------- */

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::onRequest(tongsim_lite::demo_rl::SimpleMoveTowardsRequest& request)
{
	UWorld* World = Instance->GetWorld();
	if (!World)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld."));
		return;
	}

	// 解析 actor_id
	FGuid Guid;
	if (!DemoRLServiceHelpers::ObjectIdToGuid(request.actor_id(), Guid))
	{
		this->finish(tongos::ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "actor_id missing/invalid."));
		return;
	}
	ActorGuid = Guid;

	if (std::shared_ptr<FSimpleMoveTowardsReactor>* Found = Instance->SimpleMoveReactorMap.Find(Guid))
	{
		Instance->SimpleMoveReactorMap.Remove(Guid);
	}
	// 自己注册到 Map
	Instance->SimpleMoveReactorMap.Add(Guid, this->sharedSelf<FSimpleMoveTowardsReactor>());

	// 定位 Actor
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(request.actor_id());
	if (!IsValid(Actor))
	{
		this->finish(tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found."));
		return;
	}
	ControlledActor = Actor;

	Target = DemoRLServiceHelpers::FromProtoVector3f(request.target_location());
	TotalTime = 0.f;

	if (request.has_speed_uu_per_sec())  SpeedUUPerSec = request.speed_uu_per_sec();
	if (request.has_tolerance_uu())      ToleranceUU   = request.tolerance_uu();

	// 读取朝向控制
	OrientationMode = request.orientation_mode();
	bGivenOrientationValid = false;
	bGivenApplied = false;
	if (OrientationMode == tongsim_lite::demo_rl::ORIENTATION_GIVEN && request.has_given_orientation())
	{
		const auto& fwd = request.given_orientation(); // Vector3f
		FVector2D v(fwd.x(), fwd.y()); // 只考虑 XY
		if (!v.IsNearlyZero())
		{
			GivenForwardXY = v.GetSafeNormal(); // 单位化的前向方向
			bGivenOrientationValid = true;
		}
	}

	// 如果起点已到达
	if (FVector::DistSquared(Actor->GetActorLocation(), Target) <= (ToleranceUU * ToleranceUU))
	{
		// 若给定朝向，仍然应用一次
		if (OrientationMode == tongsim_lite::demo_rl::ORIENTATION_GIVEN && bGivenOrientationValid)
		{
			ApplyGivenOrientationOnce();
		}
		WriteAndFinishResponse();
		return;
	}
}

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::onCancel()
{
	Instance->SimpleMoveReactorMap.Remove(ActorGuid);
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "SimpleMoveTowards cancelled by client."));
}

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::Tick(float DeltaTime)
{
	TotalTime += DeltaTime;

	AActor* Pawn = ControlledActor.Get();
	if (!IsValid(Pawn))
	{
		Instance->SimpleMoveReactorMap.Remove(ActorGuid);
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Controlled pawn invalidated."));
		return;
	}

	const FVector Curr = Pawn->GetActorLocation();
	const FVector Delta = Target - Curr;
	const double Dist2 = FVector::DistSquaredXY(Target, Curr);

	// 朝向处理（GIVEN 模式仅在首次应用一次；FACE_MOVEMENT 模式逐帧更新）
	if (OrientationMode == tongsim_lite::demo_rl::ORIENTATION_GIVEN && bGivenOrientationValid && !bGivenApplied)
	{
		ApplyGivenOrientationOnce();
	}
	else if (OrientationMode == tongsim_lite::demo_rl::ORIENTATION_FACE_MOVEMENT && !Delta.IsNearlyZero())
	{
		const FVector StepDir = FVector(Delta.X, Delta.Y, 0.0).GetSafeNormal();
		ApplyFaceMovementYaw(StepDir);
	}

	// 已到达
	if (Dist2 <= (ToleranceUU * ToleranceUU))
	{
		WriteAndFinishResponse();
		return;
	}

	// 计算本帧位移（限幅直线）
	const FVector StepDir = FVector(Delta.X, Delta.Y, 0.0).GetSafeNormal();
	const float StepLen = SpeedUUPerSec * FMath::Max(DeltaTime, 0.f);
	const FVector Step = StepDir * StepLen;

	// 若本帧步长将越过目标点，则直接到达目标点（仅改 XY，保留 Z）并返回
	if (StepLen * StepLen >= Dist2)
	{
		FHitResult Hit;
		const FVector TargetXY(Target.X, Target.Y, Curr.Z);
		const bool bMoved = Pawn->SetActorLocation(TargetXY, /*bSweep=*/true, &Hit, ETeleportType::None);
		UE_LOG(LogTemp, Log, TEXT("[ClampToTarget] Move to %s, bMoved: %d"), *TargetXY.ToString(), bMoved);

		if (Hit.bBlockingHit)
		{
			if (!Hit.GetActor() || !Hit.GetActor()->ActorHasTag(DemoRLServiceHelpers::RLFloorName))
			{
				bHitSomething = true;
				LastHit = Hit;
			}
		}

		WriteAndFinishResponse();
		return;
	}

	// Sweep 碰撞移动
	FHitResult Hit;
	FVector StepPoint = Curr + Step;
	const bool bMoved = Pawn->SetActorLocation(Curr + Step, /*bSweep=*/true, &Hit, ETeleportType::None);

	UE_LOG(LogTemp, Log, TEXT("Move to %s, bMoved: %d"), *StepPoint.ToString(), bMoved);
	if (Hit.bBlockingHit)
	{
		if (!Hit.GetActor() || !Hit.GetActor()->ActorHasTag(DemoRLServiceHelpers::RLFloorName))
		{
			bHitSomething = true;
			LastHit = Hit;
			WriteAndFinishResponse();
			return;
		}
	}

	// 超时
	if (TotalTime >= Instance->AsyncGrpcDeadline)
	{
		Instance->SimpleMoveReactorMap.Remove(ActorGuid);
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "Move towards time out."));
	}
}

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::ApplyFaceMovementYaw(const FVector& StepDir)
{
	AActor* Pawn = ControlledActor.Get();
	if (!IsValid(Pawn)) return;

	// UE: Yaw 从 +X 指向 +Y，atan2(Y, X)
	const double YawDeg = FMath::RadiansToDegrees(FMath::Atan2(StepDir.Y, StepDir.X));
	FRotator R = Pawn->GetActorRotation();
	R.Yaw = (float)YawDeg;
	Pawn->SetActorRotation(R);
}

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::ApplyGivenOrientationOnce()
{
	AActor* Pawn = ControlledActor.Get();
	if (!IsValid(Pawn)) return;

	if (!GivenForwardXY.IsNearlyZero())
	{
		const double YawDeg = FMath::RadiansToDegrees(FMath::Atan2(GivenForwardXY.Y, GivenForwardXY.X));
		FRotator R = Pawn->GetActorRotation();
		R.Yaw = (float)YawDeg; // 只改 Yaw，保持 Pitch/Roll 不变
		Pawn->SetActorRotation(R);
	}
	bGivenApplied = true;
}

void UDemoRLSubsystem::FSimpleMoveTowardsReactor::WriteAndFinishResponse()
{
	tongsim_lite::demo_rl::SimpleMoveTowardsResponse Resp;

	AActor* Pawn = ControlledActor.Get();
	const FVector Loc = IsValid(Pawn) ? Pawn->GetActorLocation() : FVector::ZeroVector;
	*Resp.mutable_current_location() = DemoRLServiceHelpers::ToProtoVector3f(Loc);

	if (bHitSomething)
	{
		if (AActor* HitActor = LastHit.GetActor())
		{
			if (UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance())
			{
				const FGuid HitActorGuid = GrpcSubsystem->FindGuidByActor(HitActor);
				if (HitActorGuid.IsValid())
				{
					auto* HR = Resp.mutable_hit_result();
					// 按新 proto：HitResult 内是 ActorState
					DemoRLServiceHelpers::FillActorState(HitActorGuid, HitActor, *HR->mutable_hit_actor());
				}
			}
		}
	}

	this->writeAndFinish(Resp);
	Instance->SimpleMoveReactorMap.Remove(ActorGuid);
}

/* ---------- NavigateToLocation Reactor ---------- */

void UDemoRLSubsystem::FNavigateToLocationReactor::onRequest(tongsim_lite::demo_rl::NavigateToLocationRequest& request)
{
	UWorld* World = Instance->GetWorld();
	if (!World)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld."));
		return;
	}

	// 解析 actor_id
	FGuid Guid;
	if (!DemoRLServiceHelpers::ObjectIdToGuid(request.actor_id(), Guid))
	{
		this->finish(ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "actor_id missing/invalid."));
		return;
	}
	ActorGuid = Guid;

	// 同一 Actor 互斥：避免并发请求导致挂起
	if (Instance->NavMoveReactorMap.Contains(Guid))
	{
		this->finish(ResponseStatus(grpc::StatusCode::ALREADY_EXISTS, "NavigateToLocation is already in progress for this actor."));
		return;
	}

	// 定位 Actor
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(request.actor_id());
	if (!IsValid(Actor))
	{
		this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found."));
		return;
	}

	ACharacter* Character = Cast<ACharacter>(Actor);
	if (!IsValid(Character))
	{
		this->finish(ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "Actor is not a Character."));
		return;
	}
	ControlledCharacter = Character;

	AAIController* AIController = Cast<AAIController>(Character->GetController());
	if (!IsValid(AIController))
	{
		// 尝试补一个默认 Controller（若角色配置了默认 AIControllerClass）
		Character->SpawnDefaultController();
		AIController = Cast<AAIController>(Character->GetController());
	}
	if (!IsValid(AIController))
	{
		this->finish(ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "AIController not found for Character."));
		return;
	}
	CachedAIController = AIController;

	UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(World);
	if (!NavSys)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No NavigationSystem."));
		return;
	}

	ANavigationData* NavData = NavSys->GetDefaultNavDataInstance(FNavigationSystem::DontCreate);
	if (!NavData)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No NavData."));
		return;
	}

	const bool bAllowPartial = request.allow_partial();
	const FVector Start = Character->GetActorLocation();
	FVector End = DemoRLServiceHelpers::FromProtoVector3f(request.target_location());

	// 目标点尽量投影到导航区域（投影失败则仍然尝试寻路，允许 partial path 的情况下可能仍能得到可达路径）
	FNavLocation ProjectedEnd;
	if (NavSys->ProjectPointToNavigation(End, ProjectedEnd, FVector(100.f, 100.f, 300.f)))
	{
		End = ProjectedEnd.Location;
	}

	FSharedConstNavQueryFilter QueryFilter = UNavigationQueryFilter::GetQueryFilter(*NavData, AIController, nullptr);
	FPathFindingQuery Query(AIController, *NavData, Start, End, QueryFilter);
	Query.SetAllowPartialPaths(bAllowPartial);
	const FPathFindingResult Result = NavSys->FindPathSync(Query, EPathFindingMode::Regular);
	if (!Result.IsSuccessful() || !Result.Path.IsValid())
	{
		this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "Path not found."));
		return;
	}

	bIsPartialPath = Result.Path->IsPartial();
	if (bIsPartialPath && !bAllowPartial)
	{
		this->finish(ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "Only partial path found but allow_partial is false."));
		return;
	}

	const TArray<FNavPathPoint>& Points = Result.Path->GetPathPoints();
	if (Points.Num() <= 0)
	{
		this->finish(ResponseStatus(grpc::StatusCode::NOT_FOUND, "Navigation path is empty."));
		return;
	}
	GoalLocation = Points.Last().Location;

	AcceptRadiusUU = FMath::Max(request.accept_radius(), 0.0f);
	TotalTime = 0.f;
	bStopRequested = false;
	BestDistUU = 0.f;
	TimeSinceBest = 0.f;

	// 注册到 Tick Map（必须在返回前）
	Instance->NavMoveReactorMap.Add(Guid, this->sharedSelf<FNavigateToLocationReactor>());

	// 可选：覆盖角色行走速度（MaxWalkSpeed）
	if (request.has_speed_uu_per_sec())
	{
		if (UCharacterMovementComponent* MoveComp = Character->GetCharacterMovement())
		{
			OriginalMaxWalkSpeed = MoveComp->MaxWalkSpeed;
			MoveComp->MaxWalkSpeed = FMath::Max(request.speed_uu_per_sec(), 0.0f);
			bOverrideMaxWalkSpeed = true;
		}
	}

	// 起点已在 accept radius 内：不再发 Move 请求，按“低速阈值”规则等停稳即可
	const double Dist2 = FVector::DistSquaredXY(Character->GetActorLocation(), GoalLocation);
	BestDistUU = FMath::Sqrt((float)Dist2);
	TimeSinceBest = 0.f;
	if (Dist2 <= (double)AcceptRadiusUU * (double)AcceptRadiusUU)
	{
		AIController->StopMovement();
		bStopRequested = true;
		return;
	}

	// 使用预计算 Path 发起移动；不使用 UE 默认 accept radius（AcceptanceRadius=-1）
	FAIMoveRequest MoveReq;
	MoveReq.SetGoalLocation(GoalLocation);
	MoveReq.SetAcceptanceRadius(0.0f);
	MoveReq.SetAllowPartialPath(bAllowPartial);
	MoveReq.SetUsePathfinding(false);        // Path 已给定
	MoveReq.SetProjectGoalLocation(false);   // 目标已投影
	MoveReq.SetReachTestIncludesGoalRadius(false);
	MoveReq.SetReachTestIncludesAgentRadius(false);

	const FAIRequestID MoveRequestId = AIController->RequestMove(MoveReq, Result.Path);
	if (!MoveRequestId.IsValid())
	{
		Instance->NavMoveReactorMap.Remove(Guid);
		RestoreMaxWalkSpeed();
		this->finish(ResponseStatus(grpc::StatusCode::ABORTED, "Failed to start navigation request."));
		return;
	}
}

void UDemoRLSubsystem::FNavigateToLocationReactor::RestoreMaxWalkSpeed()
{
	if (!bOverrideMaxWalkSpeed) return;

	ACharacter* Character = ControlledCharacter.Get();
	if (IsValid(Character))
	{
		if (UCharacterMovementComponent* MoveComp = Character->GetCharacterMovement())
		{
			MoveComp->MaxWalkSpeed = OriginalMaxWalkSpeed;
		}
	}
	bOverrideMaxWalkSpeed = false;
}

void UDemoRLSubsystem::FNavigateToLocationReactor::WriteAndFinishResponse(bool bSuccess, const FString& Message)
{
	tongsim_lite::demo_rl::NavigateToLocationResponse Resp;
	Resp.set_success(bSuccess);
	Resp.set_message(TCHAR_TO_UTF8(*Message));

	ACharacter* Character = ControlledCharacter.Get();
	const FVector FinalLoc = IsValid(Character) ? Character->GetActorLocation() : FVector::ZeroVector;
	*Resp.mutable_final_location() = DemoRLServiceHelpers::ToProtoVector3f(FinalLoc);
	Resp.set_is_partial(bIsPartialPath);

	RestoreMaxWalkSpeed();
	this->writeAndFinish(Resp);
	Instance->NavMoveReactorMap.Remove(ActorGuid);
}

void UDemoRLSubsystem::FNavigateToLocationReactor::onCancel()
{
	if (AAIController* AIController = CachedAIController.Get())
	{
		AIController->StopMovement();
	}
	Instance->NavMoveReactorMap.Remove(ActorGuid);
	RestoreMaxWalkSpeed();
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "NavigateToLocation cancelled by client."));
}

void UDemoRLSubsystem::FNavigateToLocationReactor::Tick(float DeltaTime)
{
	TotalTime += DeltaTime;

	ACharacter* Character = ControlledCharacter.Get();
	AAIController* AIController = CachedAIController.Get();
	if (!IsValid(Character) || !IsValid(AIController))
	{
		Instance->NavMoveReactorMap.Remove(ActorGuid);
		RestoreMaxWalkSpeed();
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Character/AIController invalidated."));
		return;
	}

	const FVector Curr = Character->GetActorLocation();
	const double Dist2 = FVector::DistSquaredXY(Curr, GoalLocation);
	const double Accept2 = (double)AcceptRadiusUU * (double)AcceptRadiusUU;
	const float DistUU = FMath::Sqrt((float)Dist2);

	// 进入 accept radius：先 StopMovement，让速度自然衰减；待低于阈值后再 StopMovementImmediately
	constexpr float kStopSpeedThresholdUUPerSec = 5.0f;
	constexpr float kProgressEpsilonUU = 5.0f;
	constexpr float kStuckSeconds = 0.6f;
	if (Dist2 <= Accept2)
	{
		if (!bStopRequested)
		{
			AIController->StopMovement();
			bStopRequested = true;
		}

		const float Speed2D = Character->GetVelocity().Size2D();
		if (Speed2D <= kStopSpeedThresholdUUPerSec)
		{
			if (UCharacterMovementComponent* MoveComp = Character->GetCharacterMovement())
			{
				MoveComp->StopMovementImmediately();
			}

			WriteAndFinishResponse(/*bSuccess=*/true, TEXT("OK"));
			return;
		}
	}
	else
	{
		// 记录最接近距离（用于判断“走不动了但速度很低”）
		if (TimeSinceBest == 0.f && BestDistUU == 0.f)
		{
			BestDistUU = DistUU;
		}
		if (DistUU + kProgressEpsilonUU < BestDistUU)
		{
			BestDistUU = DistUU;
			TimeSinceBest = 0.f;
		}
		else
		{
			TimeSinceBest += DeltaTime;
		}

		const float Speed2D = Character->GetVelocity().Size2D();
		if (!bStopRequested && Speed2D <= kStopSpeedThresholdUUPerSec && TimeSinceBest >= kStuckSeconds)
		{
			AIController->StopMovement();
			if (UCharacterMovementComponent* MoveComp = Character->GetCharacterMovement())
			{
				MoveComp->StopMovementImmediately();
			}

			const FString Msg = FString::Printf(
				TEXT("Stopped (stuck). dist=%.1fUU accept=%.1fUU"),
				DistUU, AcceptRadiusUU);
			WriteAndFinishResponse(/*bSuccess=*/false, Msg);
			return;
		}
	}

	// 超时
	if (TotalTime >= Instance->AsyncGrpcDeadline)
	{
		AIController->StopMovement();
		Instance->NavMoveReactorMap.Remove(ActorGuid);
		RestoreMaxWalkSpeed();
		this->finish(ResponseStatus(grpc::StatusCode::DEADLINE_EXCEEDED, "NavigateToLocation time out."));
	}
}

/* ---------- PickUp/Drop Object Reactors ---------- */

void UDemoRLSubsystem::FPickUpObjectReactor::onRequest(tongsim_lite::demo_rl::PickUpObjectRequest& request)
{
	UWorld* World = Instance ? Instance->GetWorld() : DemoRLServiceHelpers::GetGameWorld();
	if (!World)
	{
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "No valid UWorld."));
		return;
	}

	// 解析 actor_id
	FGuid Guid;
	if (!DemoRLServiceHelpers::ObjectIdToGuid(request.actor_id(), Guid) || !Guid.IsValid())
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("actor_id missing/invalid.");
		this->writeAndFinish(Resp);
		return;
	}
	ActorGuid = Guid;

	// 同一 Actor 互斥：避免并发抓取请求
	if (Instance->PickUpReactorMap.Contains(Guid))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("PickUpObject is already in progress for this actor.");
		this->writeAndFinish(Resp);
		return;
	}

	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(request.actor_id());
	ACharacter* Character = Cast<ACharacter>(Actor);
	if (!IsValid(Character))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("Actor not found or not a Character.");
		this->writeAndFinish(Resp);
		return;
	}

	UTSItemInteractComponent* InteractComp = Character->FindComponentByClass<UTSItemInteractComponent>();
	if (!IsValid(InteractComp))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("TSItemInteractComponent not found on Character.");
		this->writeAndFinish(Resp);
		return;
	}

	const ETSHand PreferredHand =
		request.hand() == tongsim_lite::demo_rl::HAND_LEFT ? ETSHand::Left : ETSHand::Right;

	AActor* TargetActor = DemoRLServiceHelpers::FindActorByObjectId(request.target_object_id());
	if (!IsValid(TargetActor))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("target_object_id missing/invalid or actor not found.");
		this->writeAndFinish(Resp);
		return;
	}

	const FVector TargetLocHint = request.has_target_object_location()
		                              ? DemoRLServiceHelpers::FromProtoVector3f(request.target_object_location())
		                              : TargetActor->GetActorLocation();

	FString Error;
	if (!InteractComp->StartPickUpTargetActor(TargetActor, TargetLocHint, PreferredHand, Error))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message(TCHAR_TO_UTF8(*Error));
		this->writeAndFinish(Resp);
		return;
	}

	InteractComponent = InteractComp;
	TotalTime = 0.f;
	Instance->PickUpReactorMap.Add(Guid, this->sharedSelf<FPickUpObjectReactor>());
}

void UDemoRLSubsystem::FPickUpObjectReactor::onCancel()
{
	if (UTSItemInteractComponent* InteractComp = InteractComponent.Get())
	{
		InteractComp->CancelCurrentAction(TEXT("gRPC cancelled"));
	}
	Instance->PickUpReactorMap.Remove(ActorGuid);
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "PickUpObject cancelled by client."));
}

void UDemoRLSubsystem::FPickUpObjectReactor::Tick(float DeltaTime)
{
	TotalTime += DeltaTime;

	UTSItemInteractComponent* InteractComp = InteractComponent.Get();
	if (!IsValid(InteractComp))
	{
		Instance->PickUpReactorMap.Remove(ActorGuid);
		this->finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "InteractComponent invalidated."));
		return;
	}

	FTSItemInteractResult Result;
	if (InteractComp->ConsumeLastResult(Result))
	{
		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(Result.bSuccess);
		Resp.set_message(TCHAR_TO_UTF8(*Result.Message));
		this->writeAndFinish(Resp);
		Instance->PickUpReactorMap.Remove(ActorGuid);
		return;
	}

	if (TotalTime >= Instance->AsyncGrpcDeadline)
	{
		InteractComp->CancelCurrentAction(TEXT("gRPC deadline exceeded"));

		tongsim_lite::demo_rl::PickUpObjectResponse Resp;
		Resp.set_success(false);
		Resp.set_message("PickUpObject time out.");
		this->writeAndFinish(Resp);
		Instance->PickUpReactorMap.Remove(ActorGuid);
	}
}

void UDemoRLSubsystem::FDropObjectReactor::onRequest(tongsim_lite::demo_rl::DropObjectRequest& /*request*/)
{
	tongsim_lite::demo_rl::DropObjectResponse Resp;
	Resp.set_success(false);
	Resp.set_message("Not implemented yet.");
	this->writeAndFinish(Resp);
}

void UDemoRLSubsystem::FDropObjectReactor::onCancel()
{
	this->finish(ResponseStatus(grpc::StatusCode::CANCELLED, "DropObject cancelled by client."));
}

tongos::ResponseStatus UDemoRLSubsystem::DestroyActor(
	tongsim_lite::demo_rl::DestroyActorRequest& Request,
	tongsim_lite::common::Empty&)
{
	AActor* Actor = DemoRLServiceHelpers::FindActorByObjectId(Request.actor_id());
	if (!IsValid(Actor))
		return tongos::ResponseStatus(grpc::StatusCode::NOT_FOUND, "Actor not found.");

	Actor->Destroy(/*bNetForce=*/true);
	return tongos::ResponseStatus::OK;
}

static void BuildObjectQueryParams(
	const google::protobuf::RepeatedField<int>& Types,
	FCollisionObjectQueryParams& OutObjParams)
{
	OutObjParams = FCollisionObjectQueryParams::DefaultObjectQueryParam;
	OutObjParams.ObjectTypesToQuery = 0; // 清零再置位

	for (int v : Types)
	{
		switch (static_cast<tongsim_lite::demo_rl::CollisionObjectType>(v))
		{
		case tongsim_lite::demo_rl::OBJECT_WORLD_STATIC:   OutObjParams.AddObjectTypesToQuery(ECC_WorldStatic); break;
		case tongsim_lite::demo_rl::OBJECT_WORLD_DYNAMIC:  OutObjParams.AddObjectTypesToQuery(ECC_WorldDynamic); break;
		case tongsim_lite::demo_rl::OBJECT_PAWN:           OutObjParams.AddObjectTypesToQuery(ECC_Pawn); break;
		case tongsim_lite::demo_rl::OBJECT_PHYSICS_BODY:   OutObjParams.AddObjectTypesToQuery(ECC_PhysicsBody); break;
		case tongsim_lite::demo_rl::OBJECT_VEHICLE:        OutObjParams.AddObjectTypesToQuery(ECC_Vehicle); break;
		case tongsim_lite::demo_rl::OBJECT_DESTRUCTIBLE:   OutObjParams.AddObjectTypesToQuery(ECC_Destructible); break;
		default: break;
		}
	}
}

static bool GuidOfActor(const AActor* Actor, FGuid& OutGuid)
{
	if (!Actor) return false;
	if (UTSGrpcSubsystem* GrpcSubsystem = UTSGrpcSubsystem::GetInstance())
	{
		const TMap<FGuid, TWeakObjectPtr<AActor>>& Map = GrpcSubsystem->GetIdToActorMap();
		for (const auto& KVP : Map)
		{
			if (KVP.Value.Get() == Actor)
			{
				OutGuid = KVP.Key;
				return true;
			}
		}
	}
	return false;
}

tongos::ResponseStatus UDemoRLSubsystem::BatchSingleLineTraceByObject(tongsim_lite::demo_rl::BatchSingleLineTraceByObjectRequest& Request, tongsim_lite::demo_rl::BatchSingleLineTraceByObjectResponse& Response)
{
	UWorld* World = Instance->GetWorld();
	if (!IsValid(World))
		return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "World invalid");

	// [HARD LIMIT] 一次最多处理 kMaxLineTraceJobsPerCall 条线
	// TODO:
	constexpr int32 kMaxLineTraceJobsPerCall = 20000;
	const int32 NumJobs = FMath::Min<int32>(Request.jobs_size(), kMaxLineTraceJobsPerCall);

	// 全局忽略
	for (int32 JobIndex = 0; JobIndex < NumJobs; ++JobIndex)
	{
		const auto& Job = Request.jobs(JobIndex);

		// 构造 Query Params
		FCollisionObjectQueryParams ObjParams;
		BuildObjectQueryParams(Job.object_types(), ObjParams);

		FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(BatchSingleLineTraceByObject), Job.trace_complex());
		QueryParams.bReturnPhysicalMaterial = false;

		for (const auto& Oid : Job.actors_to_ignore())
		{
			if (AActor* A = DemoRLServiceHelpers::FindActorByObjectId(Oid))
				QueryParams.AddIgnoredActor(A);
		}

		// 执行 LineTraceSingleByObjectType
		FHitResult Hit;
		const FVector Start(Job.start().x(), Job.start().y(), Job.start().z());
		const FVector End(Job.end().x(), Job.end().y(), Job.end().z());
		const bool bHit = World->LineTraceSingleByObjectType(Hit, Start, End, ObjParams, QueryParams);

		auto* Out = Response.add_results();
		Out->set_job_index(JobIndex);
		Out->set_blocking_hit(bHit);

		if (bHit)
		{
			Out->set_distance((Hit.ImpactPoint - Start).Size());
			*Out->mutable_impact_point() = DemoRLServiceHelpers::ToProtoVector3f(Hit.ImpactPoint);

			if (AActor* HitActor = Hit.GetActor())
			{
				tongsim_lite::demo_rl::ActorState* S = Out->mutable_actor_state();
				FGuid G;
				if (GuidOfActor(HitActor, G))
				{
					DemoRLServiceHelpers::FillActorState(G, HitActor, *S);
				}
				else
				{
					// 无 Guid（比如不是注册过的 Actor），也返回最基本信息
					DemoRLServiceHelpers::FillActorState(FGuid(), HitActor, *S);
				}
			}
		}
		else
		{
			Out->set_distance(0.f);
			*Out->mutable_impact_point() = DemoRLServiceHelpers::ToProtoVector3f(FVector::ZeroVector);
		}
	}

	return tongos::ResponseStatus::OK;
}

tongos::ResponseStatus UDemoRLSubsystem::BatchMultiLineTraceByObject(
    tongsim_lite::demo_rl::BatchMultiLineTraceByObjectRequest& Request,
    tongsim_lite::demo_rl::BatchMultiLineTraceByObjectResponse& Response)
{
    UWorld* World = Instance->GetWorld();
    if (!IsValid(World))
        return tongos::ResponseStatus(grpc::StatusCode::UNAVAILABLE, "World invalid");

    constexpr int32 kMaxLineTraceJobsPerCall = 20000; // 与 single 保持一致
    const int32 NumJobs = FMath::Min<int32>(Request.jobs_size(), kMaxLineTraceJobsPerCall);
    const bool bEnableDebugDraw = Request.enable_debug_draw();

    for (int32 JobIndex = 0; JobIndex < NumJobs; ++JobIndex)
    {
        const auto& Job = Request.jobs(JobIndex);

        // 1) 构造 Object/Query 参数（与 single 相同）
        FCollisionObjectQueryParams ObjParams;
        BuildObjectQueryParams(Job.object_types(), ObjParams); // 你已有的工具函数
        FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(BatchMultiLineTraceByObject), Job.trace_complex());
        QueryParams.bReturnPhysicalMaterial = false;

        for (const auto& Oid : Job.actors_to_ignore())
        {
            if (AActor* A = DemoRLServiceHelpers::FindActorByObjectId(Oid))
                QueryParams.AddIgnoredActor(A);
        }

        // 2) 执行 Multi Trace
        const FVector Start(Job.start().x(), Job.start().y(), Job.start().z());
        const FVector End(Job.end().x(), Job.end().y(), Job.end().z());
        TArray<FHitResult> Hits;
        const bool bAnyHit = World->LineTraceMultiByObjectType(Hits, Start, End, ObjParams, QueryParams);

        // 3) 仅保留 blocking，按距离升序
        bool bHasBlockingHits = false;
        if (bAnyHit)
        {
            // 过滤 blocking
            Hits = Hits.FilterByPredicate([](const FHitResult& H){ return H.bBlockingHit; });
            bHasBlockingHits = Hits.Num() > 0;

            // 按从起点的距离排序（有些平台 H.Distance 未必可靠，这里用 ImpactPoint 与 Start 计算）
            Hits.Sort([&Start](const FHitResult& A, const FHitResult& B)
            {
                const double DA = FVector::Dist(A.ImpactPoint, Start);
                const double DB = FVector::Dist(B.ImpactPoint, Start);
                return DA < DB;
            });
        }
        else
        {
            Hits.Reset();
        }

#if ENABLE_DRAW_DEBUG
        if (bEnableDebugDraw)
        {
            const float DebugLifeTime = 0.1f;
            const FColor LineColor = bHasBlockingHits ? FColor::Red : FColor::Green;
            DrawDebugLine(World, Start, End, LineColor, false, DebugLifeTime, 0, 1.5f);
            DrawDebugPoint(World, Start, 8.0f, FColor::Cyan, false, DebugLifeTime);
            DrawDebugPoint(World, End, 8.0f, FColor::Cyan, false, DebugLifeTime);

            if (bHasBlockingHits)
            {
                for (const FHitResult& Hit : Hits)
                {
                    const FVector ImpactPoint = !Hit.ImpactPoint.IsNearlyZero() ? Hit.ImpactPoint : Hit.Location;
                    const FVector ImpactNormal = !Hit.ImpactNormal.IsNearlyZero() ? Hit.ImpactNormal : Hit.Normal;

                    DrawDebugPoint(World, ImpactPoint, 10.0f, FColor::Yellow, false, DebugLifeTime);
                    DrawDebugLine(World, Start, Hit.Location, FColor::Orange, false, DebugLifeTime, 0, 0.9f);

                    if (!ImpactNormal.IsNearlyZero())
                    {
                        DrawDebugLine(World, ImpactPoint, ImpactPoint + ImpactNormal * 50.0f, FColor::Cyan, false, DebugLifeTime, 0, 1.0f);
                    }
                }
            }
        }
#endif

        // 4) 写回响应
        auto* Out = Response.add_results();
        Out->set_job_index(JobIndex);

        for (const FHitResult& H : Hits)
        {
            auto* HH = Out->add_hits();
            const FVector ImpactPoint = !H.ImpactPoint.IsNearlyZero() ? H.ImpactPoint : H.Location;
            const FVector ImpactNormal = !H.ImpactNormal.IsNearlyZero() ? H.ImpactNormal : H.Normal;
            HH->set_distance((ImpactPoint - Start).Size());
            *HH->mutable_impact_point() = DemoRLServiceHelpers::ToProtoVector3f(ImpactPoint);
            *HH->mutable_impact_normal() = DemoRLServiceHelpers::ToProtoVector3f(ImpactNormal);

            if (AActor* HitActor = H.GetActor())
            {
                tongsim_lite::demo_rl::ActorState* S = HH->mutable_actor_state();
                FGuid G;
                if (GuidOfActor(HitActor, G)) // 你已有的工具函数
                {
                    DemoRLServiceHelpers::FillActorState(G, HitActor, *S);
                }
                else
                {
                    // 没有注册 GUID 也返回基本信息
                    DemoRLServiceHelpers::FillActorState(FGuid(), HitActor, *S);
                }
            }
        }
    }

    return tongos::ResponseStatus::OK;
}
