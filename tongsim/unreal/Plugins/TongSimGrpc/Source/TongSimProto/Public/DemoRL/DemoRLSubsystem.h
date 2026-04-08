// DemoRLSubsystem.h

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "rpc_reactor.h"

// Protobuf
#include <tongsim_lite_protobuf/common.pb.h>
#include <tongsim_lite_protobuf/demo_rl.pb.h>
#include <tongsim_lite_protobuf/object.pb.h>
#include <tongsim_lite_protobuf/voxel.pb.h>

#include "DemoRLSubsystem.generated.h"

namespace tongos
{
	class ResponseStatus;
}

class AAIController;
class ACharacter;
class UTSItemInteractComponent;

UCLASS()
class UDemoRLSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
	GENERATED_BODY()

public:
	// USubsystem
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// Tickable
	virtual void Tick(float DeltaTime) override;
	virtual bool IsTickable() const override { return !IsTemplate(); }
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UDemoRLSubsystem, STATGROUP_Tickables); }

	// World hook
	void HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues IVS);

	/* ---------- Unary Handlers ---------- */

	/** QueryState: 返回全局 Actor 状态列表 */
	static tongos::ResponseStatus QueryState(
		tongsim_lite::common::Empty& Request,
		tongsim_lite::demo_rl::DemoRLState& Response);

	/** GetActorState: 通过 ObjectId 获取 ActorState */
	static tongos::ResponseStatus GetActorState(
		tongsim_lite::demo_rl::GetActorStateRequest& Request,
		tongsim_lite::demo_rl::GetActorStateResponse& Response);

	/** GetActorTransform: 通过 ObjectId 获取 Transform */
	static tongos::ResponseStatus GetActorTransform(
		tongsim_lite::demo_rl::GetActorTransformRequest& Request,
		tongsim_lite::demo_rl::GetActorTransformResponse& Response);

	/** SetActorTransform: 通过 ObjectId 设置 Transform */
	static tongos::ResponseStatus SetActorTransform(
		tongsim_lite::demo_rl::SetActorTransformRequest& Request,
		tongsim_lite::common::Empty& Response);

	/** SpawnActor: 按给定蓝图/类与初始 Transform 生成 Actor */
	static tongos::ResponseStatus SpawnActor(
		tongsim_lite::demo_rl::SpawnActorRequest& Request,
		tongsim_lite::demo_rl::SpawnActorResponse& Response);

	/** TODO: */
	static tongos::ResponseStatus QueryVoxel(
		tongsim_lite::voxel::QueryVoxelRequest& Request,
		tongsim_lite::voxel::Voxel& Response);

	/** ExecConsoleCommand: 执行 UE 控制台命令 */
	static tongos::ResponseStatus ExecConsoleCommand(
		tongsim_lite::demo_rl::ExecConsoleCommandRequest& Request,
		tongsim_lite::demo_rl::ExecConsoleCommandResponse& Response);

	/** QueryNavigationPath: 查询导航路径 */
	static tongos::ResponseStatus QueryNavigationPath(
		tongsim_lite::demo_rl::QueryNavigationPathRequest& Request,
		tongsim_lite::demo_rl::QueryNavigationPathResponse& Response);

	static tongos::ResponseStatus DestroyActor(
		tongsim_lite::demo_rl::DestroyActorRequest& Request,
		tongsim_lite::common::Empty& Response);

	static tongos::ResponseStatus BatchSingleLineTraceByObject(
		tongsim_lite::demo_rl::BatchSingleLineTraceByObjectRequest& Request,
		tongsim_lite::demo_rl::BatchSingleLineTraceByObjectResponse& Response);

	static tongos::ResponseStatus BatchMultiLineTraceByObject(
		tongsim_lite::demo_rl::BatchMultiLineTraceByObjectRequest& Request,
		tongsim_lite::demo_rl::BatchMultiLineTraceByObjectResponse& Response);
	/* ---------- Reactor(s) ---------- */

	/** ResetLevel 的 Reactor：Unary + 异步完成 */
	class FResetLevelReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::common::Empty, tongsim_lite::common::Empty>
	{
	public:
		void onRequest(tongsim_lite::common::Empty& request) override;
		void onCancel() override;

		void Tick(float DeltaTime);

		friend class UDemoRLSubsystem;

	private:
		void OnPreLoadMap(const FString& MapName);
		void OnPostLoadMapWithWorld(UWorld* LoadedWorld);
		void OnTravelFailure(UWorld* InWorld, ETravelFailure::Type FailureType, const FString& Error);
		void CleanupDelegates();

		float TotalTime = 0.f;
		FName TargetLevel;

		bool bLoadStarted = false;
		TWeakObjectPtr<UWorld> NewWorld;

		FDelegateHandle PreLoadHandle;
		FDelegateHandle PostLoadHandle;
		FDelegateHandle TravelFailHandle;
	};

	std::shared_ptr<FResetLevelReactor> ResetLevelReactorPtr;

	/** SimpleMoveTowards 的 Reactor */
	class FSimpleMoveTowardsReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::demo_rl::SimpleMoveTowardsRequest, tongsim_lite::demo_rl::SimpleMoveTowardsResponse>
	{
	public:
		void onRequest(tongsim_lite::demo_rl::SimpleMoveTowardsRequest& request) override;
		void onCancel() override;

		void Tick(float DeltaTime);

		friend class UDemoRLSubsystem;

	private:
		// 运行态
		FGuid ActorGuid;
		TWeakObjectPtr<AActor> ControlledActor;
		FVector Target = FVector::ZeroVector;
		float TotalTime = 0.f;

		// 参数
		float SpeedUUPerSec = 300.f; // UU/s
		float ToleranceUU = 5.f; // 到达阈值

		// 朝向控制
		tongsim_lite::demo_rl::OrientationMode OrientationMode =
			tongsim_lite::demo_rl::ORIENTATION_KEEP_CURRENT;
		FVector2D GivenForwardXY = FVector2D::ZeroVector; // ORIENTATION_GIVEN: 前向向量的 XY 分量（单位化）
		bool bGivenOrientationValid = false;

		// 命中记录
		bool bHitSomething = false;
		FHitResult LastHit;

		// helpers
		void WriteAndFinishResponse();
		void ApplyFaceMovementYaw(const FVector& StepDir);
		void ApplyGivenOrientationOnce();
		bool bGivenApplied = false;
	};

	TMap<FGuid, std::shared_ptr<FSimpleMoveTowardsReactor>> SimpleMoveReactorMap;

	/** NavigateToLocation 的 Reactor：NavMesh 自动导航（沿 Path 前进，满足自定义 accept radius + 低速阈值后停止） */
	class FNavigateToLocationReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::demo_rl::NavigateToLocationRequest, tongsim_lite::demo_rl::NavigateToLocationResponse>
	{
	public:
		void onRequest(tongsim_lite::demo_rl::NavigateToLocationRequest& request) override;
		void onCancel() override;

		void Tick(float DeltaTime);

		friend class UDemoRLSubsystem;

	private:
		FGuid ActorGuid;
		TWeakObjectPtr<ACharacter> ControlledCharacter;
		TWeakObjectPtr<AAIController> CachedAIController;

		FVector GoalLocation = FVector::ZeroVector;
		float AcceptRadiusUU = 50.f;
		bool bIsPartialPath = false;

		float TotalTime = 0.f;
		bool bStopRequested = false;

		// movement speed override
		float OriginalMaxWalkSpeed = 0.f;
		bool bOverrideMaxWalkSpeed = false;

		// stuck/progress detection
		float BestDistUU = 0.f;
		float TimeSinceBest = 0.f;

		void RestoreMaxWalkSpeed();
		void WriteAndFinishResponse(bool bSuccess, const FString& Message);
	};

	TMap<FGuid, std::shared_ptr<FNavigateToLocationReactor>> NavMoveReactorMap;

	/** PickUpObject 的 Reactor：驱动 TongSimCore 的抓取组件并延迟返回 */
	class FPickUpObjectReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::demo_rl::PickUpObjectRequest, tongsim_lite::demo_rl::PickUpObjectResponse>
	{
	public:
		void onRequest(tongsim_lite::demo_rl::PickUpObjectRequest& request) override;
		void onCancel() override;

		void Tick(float DeltaTime);

		friend class UDemoRLSubsystem;

	private:
		FGuid ActorGuid;
		TWeakObjectPtr<UTSItemInteractComponent> InteractComponent;
		float TotalTime = 0.f;
	};

	TMap<FGuid, std::shared_ptr<FPickUpObjectReactor>> PickUpReactorMap;

	/** DropObject 的 Reactor：先打通 gRPC，UE 逻辑留空 */
	class FDropObjectReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::demo_rl::DropObjectRequest, tongsim_lite::demo_rl::DropObjectResponse>
	{
	public:
		void onRequest(tongsim_lite::demo_rl::DropObjectRequest& request) override;
		void onCancel() override;
	};

private:
	static UDemoRLSubsystem* Instance;

	/** 超时时间（秒），默认 60 秒 */
	float AsyncGrpcDeadline = 60.f;
};
