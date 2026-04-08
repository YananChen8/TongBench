#pragma once
#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "TSArenaSubsystem.h"
#include "rpc_reactor.h" // 你的 gRPC 注册/响应包装
// Protobuf
#include <tongsim_lite_protobuf/common.pb.h>
#include <tongsim_lite_protobuf/object.pb.h>
#include <tongsim_lite_protobuf/arena.pb.h>

#include "ArenaGrpcSubsystem.generated.h"

UCLASS()
class UArenaGrpcSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// Tickable
	virtual void Tick(float DeltaTime) override;
	virtual bool IsTickable() const override { return !IsTemplate(); }
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UArenaGrpcSubsystem, STATGROUP_Tickables); }

	void HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues IVS);

	// ---- Unary handlers ----
	static tongos::ResponseStatus SetArenaVisible(
		tongsim_lite::arena::SetArenaVisibleRequest& Req,
		tongsim_lite::common::Empty& Resp);

	static tongos::ResponseStatus ListArenas(
		tongsim_lite::arena::ListArenasRequest& Req,
		tongsim_lite::arena::ListArenasResponse& Resp);

	static tongos::ResponseStatus SpawnActorInArena(
		tongsim_lite::arena::SpawnActorInArenaRequest& Req,
		tongsim_lite::arena::SpawnActorInArenaResponse& Resp);

	static tongos::ResponseStatus SetActorPoseLocal(
		tongsim_lite::arena::SetActorPoseLocalRequest& Req,
		tongsim_lite::common::Empty& Resp);

	static tongos::ResponseStatus GetActorPoseLocal(
		tongsim_lite::arena::GetActorPoseLocalRequest& Req,
		tongsim_lite::arena::GetActorPoseLocalResponse& Resp);

	static tongos::ResponseStatus LocalToWorld(
		tongsim_lite::arena::LocalToWorldRequest& Req,
		tongsim_lite::arena::LocalToWorldResponse& Resp);

	static tongos::ResponseStatus WorldToLocal(
		tongsim_lite::arena::WorldToLocalRequest& Req,
		tongsim_lite::arena::WorldToLocalResponse& Resp);

	static tongos::ResponseStatus DestroyActorInArena(
		tongsim_lite::arena::DestroyActorInArenaRequest& Req,
		tongsim_lite::common::Empty& Resp);

	// ---- Reactor handlers (延迟返回) ----
	class FLoadArenaReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::arena::LoadArenaRequest, tongsim_lite::arena::LoadArenaResponse>
	{
	public:
		void onRequest(tongsim_lite::arena::LoadArenaRequest& Req) override;
		void onCancel() override;
		void Tick(float dt);

		FGuid ArenaId;
		FTransform Anchor;
		float Deadline = 60.f, Elapsed = 0.f;
	};


	class FResetArenaReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::arena::ResetArenaRequest, tongsim_lite::common::Empty>
	{
	public:
		void onRequest(tongsim_lite::arena::ResetArenaRequest& Req) override;
		void onCancel() override;
		void Tick(float dt);

		FGuid ArenaId;
		float Deadline = 60.f, Elapsed = 0.f;
	};

	class FDestroyArenaReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::arena::DestroyArenaRequest, tongsim_lite::common::Empty>
	{
	public:
		friend class UArenaGrpcSubsystem;

		void onRequest(tongsim_lite::arena::DestroyArenaRequest& Req) override;
		void onCancel() override;
		void Tick(float dt);

		FGuid ArenaId;
		float Deadline = 60.f, Elapsed = 0.f;
	};

	class FSimpleMoveTowardsInArenaReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::arena::SimpleMoveTowardsInArenaRequest, tongsim_lite::arena::SimpleMoveTowardsInArenaResponse>
	{
	public:
		friend class UArenaGrpcSubsystem;

		void onRequest(tongsim_lite::arena::SimpleMoveTowardsInArenaRequest& Req) override;
		void onCancel() override;
		void Tick(float dt);

		TWeakObjectPtr<APawn> ControlledPawn;
		FGuid ArenaId;
		FVector Target = FVector::ZeroVector;
		int32 OrientationMode = 0;
		FVector2D GivenForwardXY = FVector2D::ZeroVector;
		bool bGivenOrientationValid = false;
		bool bGivenApplied = false;
		float Elapsed = 0.f;
		FHitResult LastHit;

		float SpeedUUPerSec = 300.f;
		float ToleranceUU   = 5.f;
		bool  bHitSomething = false;

		void ApplyFaceMovementYaw(const FVector& StepDir);
		void ApplyGivenOrientationOnce();
		void WriteAndFinishResponse();
	};

private:
	TMap<FGuid, std::shared_ptr<FLoadArenaReactor>>        LoadReactors;
	TMap<FGuid, std::shared_ptr<FResetArenaReactor>>       ResetReactors;
	TMap<FGuid, std::shared_ptr<FDestroyArenaReactor>>     DestroyReactors;
	TMap<FGuid, std::shared_ptr<FSimpleMoveTowardsInArenaReactor>> MoveReactors;
	// 说明：Load/Reset/Destroy/Move 在同一 Arena 内互斥；不同 Arena 互不影响。
	TSet<FGuid> BusyArenas;

	float AsyncGrpcDeadline = 60.f;

private:
	static UArenaGrpcSubsystem* Instance;
	static bool BytesLEToFGuid(const uint8 In[16], FGuid& Out);
	static void FGuidToBytesLE(const FGuid& G, uint8 Out[16]);
	static FTransform FromProtoXf(const tongsim_lite::common::Transform& T);
	static tongsim_lite::common::Transform ToProtoXf(const FTransform& T);
	static bool ObjectIdToGuid(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid);
};
