#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Tickable.h"
#include "rpc_reactor.h"

#include <memory>

#include <tongsim_lite_protobuf/capture.pb.h>

#include "CaptureGrpcSubsystem.generated.h"

class UTSGrpcSubsystem;
class UTSCaptureSubsystem;
class ATSCaptureCameraActor;
class UTSCaptureBPLibrary;

namespace tongos
{
class ResponseStatus;
}

UCLASS()
class UCaptureGrpcSubsystem : public UGameInstanceSubsystem, public FTickableGameObject
{
 	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	virtual void Tick(float DeltaTime) override;
	virtual bool IsTickable() const override { return !IsTemplate(); }
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UCaptureGrpcSubsystem, STATGROUP_Tickables); }

	void HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues Values);

	// Unary handlers
	static tongos::ResponseStatus ListCaptureCameras(
		tongsim_lite::capture::ListCaptureCamerasRequest& Req,
		tongsim_lite::capture::ListCaptureCamerasResponse& Resp);

	static tongos::ResponseStatus CreateCaptureCamera(
		tongsim_lite::capture::CreateCaptureCameraRequest& Req,
		tongsim_lite::capture::CreateCaptureCameraResponse& Resp);

	static tongos::ResponseStatus DestroyCaptureCamera(
		tongsim_lite::capture::DestroyCaptureCameraRequest& Req,
		tongsim_lite::common::Empty& Resp);

	static tongos::ResponseStatus SetCaptureCameraPose(
		tongsim_lite::capture::SetCaptureCameraPoseRequest& Req,
		tongsim_lite::common::Empty& Resp);

	static tongos::ResponseStatus UpdateCaptureCameraParams(
		tongsim_lite::capture::UpdateCaptureCameraParamsRequest& Req,
		tongsim_lite::capture::UpdateCaptureCameraParamsResponse& Resp);

	static tongos::ResponseStatus AttachCaptureCamera(
		tongsim_lite::capture::AttachCaptureCameraRequest& Req,
		tongsim_lite::common::Empty& Resp);

	static tongos::ResponseStatus GetCaptureStatus(
		tongsim_lite::capture::GetCaptureStatusRequest& Req,
		tongsim_lite::capture::GetCaptureStatusResponse& Resp);

	struct FCaptureCameraState
	{
		TWeakObjectPtr<ATSCaptureCameraActor> CameraActor;
		FName CaptureId;
		tongsim_lite::capture::CaptureCameraParams ProtoParams;
		tongsim_lite::capture::CaptureCameraStatus ProtoStatus;
	};

	class FCaptureSnapshotReactor final
		: public tongos::RpcReactorUnary<tongsim_lite::capture::CaptureSnapshotRequest, tongsim_lite::capture::CaptureFrame>
	{
	public:
		void onRequest(tongsim_lite::capture::CaptureSnapshotRequest& Req) override;
	};

private:
	static UCaptureGrpcSubsystem* Instance;

	TMap<FGuid, FCaptureCameraState> CameraStates;

	UTSCaptureSubsystem* ResolveCaptureSubsystem() const;
	UTSGrpcSubsystem* ResolveGrpcSubsystem() const;

	ATSCaptureCameraActor* FindCameraActorById(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid) const;
	FCaptureCameraState* EnsureCameraState(FGuid CameraGuid, ATSCaptureCameraActor* Actor);

	static bool ObjectIdToGuid(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid);
	static void GuidToObjectId(const FGuid& Guid, tongsim_lite::object::ObjectId& OutId);
	static tongsim_lite::capture::CaptureCameraParams ToProtoParams(const struct FTSCaptureCameraParams& Params);
	static void FromProtoParams(const tongsim_lite::capture::CaptureCameraParams& Proto, struct FTSCaptureCameraParams& Out);
	static tongsim_lite::capture::CaptureFrame ToProtoFrame(const FGuid& CameraGuid, const TSharedPtr<struct FTSCaptureFrame>& Frame, const FCaptureCameraState* State, bool bIncludeColor, bool bIncludeDepth);
	static tongsim_lite::capture::CaptureCameraStatus ToProtoStatus(const struct FTSCaptureStatus& Status);

	void UpdateStatusFromSubsystem(FGuid CameraGuid, FCaptureCameraState& State);
	static tongos::ResponseStatus CaptureSnapshotInternal(const tongsim_lite::capture::CaptureSnapshotRequest& Req, tongsim_lite::capture::CaptureFrame& OutFrame);
};
