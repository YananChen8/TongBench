#include "Capture/CaptureGrpcSubsystem.h"

#include "Async/Async.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "grpcpp/support/status.h"
#include "TongosGrpc/Public/TSGrpcSubsystem.h"
#include "TongSimCapture/Public/TSCaptureBPLibrary.h"
#include "TongSimCapture/Public/TSCaptureCameraActor.h"
#include "TongSimCapture/Public/TSCaptureSubsystem.h"
#include "TongSimCapture/Public/TSCaptureTypes.h"

using namespace tongos;

namespace
{
constexpr const char* kServicePrefix = "/tongsim_lite.capture.CaptureService/";

FTransform FromProtoTransform(const tongsim_lite::common::Transform& Proto)
{
	const FVector Location(Proto.location().x(), Proto.location().y(), Proto.location().z());
	const FRotator Rotation(Proto.rotation().pitch_deg(), Proto.rotation().yaw_deg(), Proto.rotation().roll_deg());
	const FVector Scale(Proto.scale().x(), Proto.scale().y(), Proto.scale().z());
	return FTransform(Rotation, Location, Scale);
}

tongsim_lite::common::Transform ToProtoTransform(const FTransform& Transform)
{
	tongsim_lite::common::Transform Out;
	Out.mutable_location()->set_x(Transform.GetLocation().X);
	Out.mutable_location()->set_y(Transform.GetLocation().Y);
	Out.mutable_location()->set_z(Transform.GetLocation().Z);
	const FRotator Rot = Transform.Rotator();
	Out.mutable_rotation()->set_roll_deg(Rot.Roll);
	Out.mutable_rotation()->set_pitch_deg(Rot.Pitch);
	Out.mutable_rotation()->set_yaw_deg(Rot.Yaw);
	Out.mutable_scale()->set_x(Transform.GetScale3D().X);
	Out.mutable_scale()->set_y(Transform.GetScale3D().Y);
	Out.mutable_scale()->set_z(Transform.GetScale3D().Z);
	return Out;
}

tongsim_lite::capture::CaptureColorSource FromUECaptureSource(TEnumAsByte<ESceneCaptureSource> Source)
{
	return static_cast<tongsim_lite::capture::CaptureColorSource>(static_cast<int32>(Source.GetValue()));
}

TEnumAsByte<ESceneCaptureSource> ToUECaptureSource(tongsim_lite::capture::CaptureColorSource Source)
{
	return TEnumAsByte<ESceneCaptureSource>(static_cast<uint8>(Source));
}

tongsim_lite::capture::CaptureRenderTargetFormat FromUERenderTargetFormat(TEnumAsByte<ETextureRenderTargetFormat> Format)
{
	return static_cast<tongsim_lite::capture::CaptureRenderTargetFormat>(static_cast<int32>(Format.GetValue()));
}

TEnumAsByte<ETextureRenderTargetFormat> ToUERenderTargetFormat(tongsim_lite::capture::CaptureRenderTargetFormat Format)
{
	return TEnumAsByte<ETextureRenderTargetFormat>(static_cast<uint8>(Format));
}

tongsim_lite::capture::CaptureDepthMode FromUEDepthMode(ETSCaptureDepthMode Mode)
{
	return static_cast<tongsim_lite::capture::CaptureDepthMode>(static_cast<uint8>(Mode));
}

ETSCaptureDepthMode ToUEDepthMode(tongsim_lite::capture::CaptureDepthMode Mode)
{
	return static_cast<ETSCaptureDepthMode>(static_cast<uint8>(Mode));
}

tongsim_lite::capture::CaptureRgbCodec FromUERgbCodec(ETSRgbCodec Codec)
{
	return static_cast<tongsim_lite::capture::CaptureRgbCodec>(static_cast<uint8>(Codec));
}

ETSRgbCodec ToUERgbCodec(tongsim_lite::capture::CaptureRgbCodec Codec)
{
	return static_cast<ETSRgbCodec>(static_cast<uint8>(Codec));
}

tongsim_lite::capture::CaptureDepthCodec FromUEDepthCodec(ETSDepthCodec Codec)
{
	return static_cast<tongsim_lite::capture::CaptureDepthCodec>(static_cast<uint8>(Codec));
}

ETSDepthCodec ToUEDepthCodec(tongsim_lite::capture::CaptureDepthCodec Codec)
{
	return static_cast<ETSDepthCodec>(static_cast<uint8>(Codec));
}

bool BytesLEToGuid(const uint8 In[16], FGuid& OutGuid)
{
	uint32 Parts[4];
	for (int32 Index = 0; Index < 4; ++Index)
	{
		const int32 Offset = Index * 4;
		Parts[Index] = static_cast<uint32>(In[Offset]) |
			(static_cast<uint32>(In[Offset + 1]) << 8) |
			(static_cast<uint32>(In[Offset + 2]) << 16) |
			(static_cast<uint32>(In[Offset + 3]) << 24);
	}
	OutGuid = FGuid(Parts[0], Parts[1], Parts[2], Parts[3]);
	return true;
}

bool BytesLEToGuid(const std::string& Bytes, FGuid& OutGuid)
{
	if (Bytes.size() != 16)
	{
		return false;
	}
	return BytesLEToGuid(reinterpret_cast<const uint8*>(Bytes.data()), OutGuid);
}

void GuidToBytesLE(const FGuid& Guid, uint8 Out[16])
{
	const uint32 Parts[4] = {static_cast<uint32>(Guid.A), static_cast<uint32>(Guid.B), static_cast<uint32>(Guid.C), static_cast<uint32>(Guid.D)};
	for (int32 Index = 0; Index < 4; ++Index)
	{
		const uint32 Value = Parts[Index];
		const int32 Offset = Index * 4;
		Out[Offset + 0] = static_cast<uint8>(Value & 0xFF);
		Out[Offset + 1] = static_cast<uint8>((Value >> 8) & 0xFF);
		Out[Offset + 2] = static_cast<uint8>((Value >> 16) & 0xFF);
		Out[Offset + 3] = static_cast<uint8>((Value >> 24) & 0xFF);
	}
}

UWorld* GetGameWorld()
{
	if (!GEngine)
	{
		return nullptr;
	}
	for (const FWorldContext& Context : GEngine->GetWorldContexts())
	{
		if (Context.WorldType == EWorldType::Game || Context.WorldType == EWorldType::PIE)
		{
			return Context.World();
		}
	}
	return nullptr;
}
} // namespace

UCaptureGrpcSubsystem* UCaptureGrpcSubsystem::Instance = nullptr;

void UCaptureGrpcSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	Instance = this;
	FWorldDelegates::OnPostWorldInitialization.AddUObject(this, &ThisClass::HandlePostWorldInit);
}

void UCaptureGrpcSubsystem::Deinitialize()
{
	FWorldDelegates::OnPostWorldInitialization.RemoveAll(this);
	Instance = nullptr;
	CameraStates.Empty();
	Super::Deinitialize();
}

void UCaptureGrpcSubsystem::Tick(float DeltaTime)
{
	for (auto It = CameraStates.CreateIterator(); It; ++It)
	{
		if (!It.Value().CameraActor.IsValid())
		{
			It.RemoveCurrent();
		}
	}
}

void UCaptureGrpcSubsystem::HandlePostWorldInit(UWorld* World, const UWorld::InitializationValues)
{
	if (UTSGrpcSubsystem* Grpc = ResolveGrpcSubsystem())
	{
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "ListCaptureCameras", &ThisClass::ListCaptureCameras);
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "CreateCaptureCamera", &ThisClass::CreateCaptureCamera);
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "DestroyCaptureCamera", &ThisClass::DestroyCaptureCamera);
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "SetCaptureCameraPose", &ThisClass::SetCaptureCameraPose);
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "UpdateCaptureCameraParams", &ThisClass::UpdateCaptureCameraParams);
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "AttachCaptureCamera", &ThisClass::AttachCaptureCamera);
		Grpc->RegisterReactor<UCaptureGrpcSubsystem::FCaptureSnapshotReactor>(std::string(kServicePrefix) + "CaptureSnapshot");
		Grpc->RegisterUnaryHandler(std::string(kServicePrefix) + "GetCaptureStatus", &ThisClass::GetCaptureStatus);
	}

	if (World)
	{
		for (TActorIterator<ATSCaptureCameraActor> It(World); It; ++It)
		{
			ATSCaptureCameraActor* Camera = *It;
			if (!IsValid(Camera))
			{
				continue;
			}
			FGuid CameraGuid;
			if (UTSGrpcSubsystem* GrpcSubsystem = ResolveGrpcSubsystem())
			{
				CameraGuid = GrpcSubsystem->FindGuidByActor(Camera);
			}
			if (CameraGuid.IsValid())
			{
				EnsureCameraState(CameraGuid, Camera);
			}
		}
	}
}

UTSCaptureSubsystem* UCaptureGrpcSubsystem::ResolveCaptureSubsystem() const
{
	return GetGameInstance() ? GetGameInstance()->GetSubsystem<UTSCaptureSubsystem>() : nullptr;
}

UTSGrpcSubsystem* UCaptureGrpcSubsystem::ResolveGrpcSubsystem() const
{
	return UTSGrpcSubsystem::GetInstance();
}

ATSCaptureCameraActor* UCaptureGrpcSubsystem::FindCameraActorById(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid) const
{
	OutGuid.Invalidate();
	if (!ResolveGrpcSubsystem())
	{
		return nullptr;
	}
	if (!ObjectIdToGuid(Id, OutGuid))
	{
		return nullptr;
	}
	if (AActor* Actor = ResolveGrpcSubsystem()->FindActorByGuid(OutGuid))
	{
		return Cast<ATSCaptureCameraActor>(Actor);
	}
	return nullptr;
}

UCaptureGrpcSubsystem::FCaptureCameraState* UCaptureGrpcSubsystem::EnsureCameraState(FGuid CameraGuid, ATSCaptureCameraActor* Actor)
{
	if (!CameraGuid.IsValid() || !IsValid(Actor))
	{
		return nullptr;
	}
	if (FCaptureCameraState* Existing = CameraStates.Find(CameraGuid))
	{
		Existing->CameraActor = Actor;
		Existing->CaptureId = Actor->CaptureId;
		Existing->ProtoParams = ToProtoParams(Actor->Params);
		UpdateStatusFromSubsystem(CameraGuid, *Existing);
		return Existing;
	}
	FCaptureCameraState& NewState = CameraStates.Add(CameraGuid);
	NewState.CameraActor = Actor;
	NewState.CaptureId = Actor->CaptureId;
	NewState.ProtoParams = ToProtoParams(Actor->Params);
	UpdateStatusFromSubsystem(CameraGuid, NewState);
	return &NewState;
}

bool UCaptureGrpcSubsystem::ObjectIdToGuid(const tongsim_lite::object::ObjectId& Id, FGuid& OutGuid)
{
	return BytesLEToGuid(Id.guid(), OutGuid);
}

void UCaptureGrpcSubsystem::GuidToObjectId(const FGuid& Guid, tongsim_lite::object::ObjectId& OutId)
{
	uint8 Bytes[16];
	GuidToBytesLE(Guid, Bytes);
	OutId.set_guid(reinterpret_cast<const char*>(Bytes), 16);
}

tongsim_lite::capture::CaptureCameraParams UCaptureGrpcSubsystem::ToProtoParams(const FTSCaptureCameraParams& Params)
{
	tongsim_lite::capture::CaptureCameraParams Out;
	Out.set_width(Params.Width);
	Out.set_height(Params.Height);
	Out.set_fov_degrees(Params.FovDegrees);
	Out.set_qps(Params.Qps);
	Out.set_enable_depth(Params.bEnableDepth);
	Out.set_color_source(FromUECaptureSource(Params.ColorCaptureSource));
	Out.set_color_format(FromUERenderTargetFormat(Params.ColorRenderTargetFormat));
	Out.set_enable_post_process(Params.bEnablePostProcess);
	Out.set_enable_temporal_aa(Params.bEnableTemporalAA);
	Out.set_depth_near(Params.DepthNearPlane);
	Out.set_depth_far(Params.DepthFarPlane);
	Out.set_depth_mode(FromUEDepthMode(Params.DepthMode));
	Out.set_rgb_codec(FromUERgbCodec(Params.RgbCodec));
	Out.set_depth_codec(FromUEDepthCodec(Params.DepthCodec));
	Out.set_jpeg_quality(Params.JpegQuality);
	return Out;
}

void UCaptureGrpcSubsystem::FromProtoParams(const tongsim_lite::capture::CaptureCameraParams& Proto, FTSCaptureCameraParams& Out)
{
	Out.Width = Proto.width();
	Out.Height = Proto.height();
	Out.FovDegrees = Proto.fov_degrees();
	Out.Qps = Proto.qps();
	Out.bEnableDepth = Proto.enable_depth();
	Out.ColorCaptureSource = ToUECaptureSource(Proto.color_source());
	Out.ColorRenderTargetFormat = ToUERenderTargetFormat(Proto.color_format());
	Out.bEnablePostProcess = Proto.enable_post_process();
	Out.bEnableTemporalAA = Proto.enable_temporal_aa();
	Out.DepthNearPlane = Proto.depth_near();
	Out.DepthFarPlane = Proto.depth_far();
	Out.DepthMode = ToUEDepthMode(Proto.depth_mode());
	Out.RgbCodec = ToUERgbCodec(Proto.rgb_codec());
	Out.DepthCodec = ToUEDepthCodec(Proto.depth_codec());
	Out.JpegQuality = Proto.jpeg_quality();
}

tongsim_lite::capture::CaptureCameraStatus UCaptureGrpcSubsystem::ToProtoStatus(const FTSCaptureStatus& Status)
{
	tongsim_lite::capture::CaptureCameraStatus Out;
	Out.set_capturing(Status.bCapturing);
	Out.set_queue_count(Status.QueueCount);
	Out.set_compressed_queue_count(Status.CompressedQueueCount);
	Out.set_width(Status.Width);
	Out.set_height(Status.Height);
	Out.set_fov_degrees(Status.FovDegrees);
	Out.set_depth_mode(FromUEDepthMode(Status.DepthMode));
	return Out;
}

void UCaptureGrpcSubsystem::UpdateStatusFromSubsystem(FGuid CameraGuid, FCaptureCameraState& State)
{
	if (UTSCaptureSubsystem* CaptureSubsystem = ResolveCaptureSubsystem())
	{
		FTSCaptureStatus LocalStatus;
		if (CaptureSubsystem->GetStatus(State.CaptureId, LocalStatus))
		{
			State.ProtoStatus = ToProtoStatus(LocalStatus);
		}
	}
}

tongos::ResponseStatus UCaptureGrpcSubsystem::CaptureSnapshotInternal(
	const tongsim_lite::capture::CaptureSnapshotRequest& Req,
	tongsim_lite::capture::CaptureFrame& OutFrame)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}

	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera) || !CameraGuid.IsValid())
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}

	UTSCaptureSubsystem* CaptureSubsystem = Instance->ResolveCaptureSubsystem();
	if (!CaptureSubsystem)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}

	FTSCaptureFrame Frame;
	const bool bSuccess = CaptureSubsystem->CaptureSnapshotOnActor(
		Camera->CaptureId,
		Camera,
		Camera->Params.Width,
		Camera->Params.Height,
		Camera->Params.FovDegrees,
		Camera->Params.bEnableDepth,
		Frame,
		Req.timeout_seconds());

	if (!bSuccess)
	{
		return ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "Snapshot failed");
	}

	const FCaptureCameraState* State = Instance->EnsureCameraState(CameraGuid, Camera);
	OutFrame = Instance->ToProtoFrame(
		CameraGuid,
		MakeShared<FTSCaptureFrame>(MoveTemp(Frame)),
		State,
		Req.include_color(),
		Req.include_depth());
	return ResponseStatus::OK;
}

tongsim_lite::capture::CaptureFrame UCaptureGrpcSubsystem::ToProtoFrame(const FGuid& CameraGuid, const TSharedPtr<FTSCaptureFrame>& Frame, const FCaptureCameraState* State, bool bIncludeColor, bool bIncludeDepth)
{
	tongsim_lite::capture::CaptureFrame Out;
	if (CameraGuid.IsValid())
	{
		GuidToObjectId(CameraGuid, *Out.mutable_camera_id());
	}
	Out.set_frame_id(Frame->FrameId);
	Out.set_game_time_seconds(Frame->GameTimeSeconds);
	Out.set_gpu_ready_timestamp(Frame->GpuReadyTimestamp);
	Out.set_width(Frame->Width);
	Out.set_height(Frame->Height);
	*Out.mutable_world_pose() = ToProtoTransform(Frame->Pose);
	Out.mutable_intrinsics()->set_fx(Frame->Intrinsics.Fx);
	Out.mutable_intrinsics()->set_fy(Frame->Intrinsics.Fy);
	Out.mutable_intrinsics()->set_cx(Frame->Intrinsics.Cx);
	Out.mutable_intrinsics()->set_cy(Frame->Intrinsics.Cy);
	if (State)
	{
		Out.set_depth_near(State->ProtoParams.depth_near());
		Out.set_depth_far(State->ProtoParams.depth_far());
		Out.set_depth_mode(State->ProtoParams.depth_mode());
	}
	else
	{
		Out.set_depth_near(0.f);
		Out.set_depth_far(0.f);
		Out.set_depth_mode(tongsim_lite::capture::CaptureDepthMode::CAPTURE_DEPTH_NONE);
	}
	if (bIncludeColor && Frame->Rgba8.Num() > 0)
	{
		Out.set_rgba8(Frame->Rgba8.GetData(), Frame->Rgba8.Num());
		Out.set_has_color(true);
	}
	else
	{
		Out.set_has_color(false);
	}
	if (bIncludeDepth && Frame->DepthR32.Num() > 0)
	{
		const uint8* DepthBytes = reinterpret_cast<const uint8*>(Frame->DepthR32.GetData());
		Out.set_depth_r32(DepthBytes, Frame->DepthR32.Num() * sizeof(float));
		Out.set_has_depth(true);
	}
	else
	{
		Out.set_has_depth(false);
	}
	return Out;
}

void UCaptureGrpcSubsystem::FCaptureSnapshotReactor::onRequest(tongsim_lite::capture::CaptureSnapshotRequest& Req)
{
	if (!Instance)
	{
		finish(ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable"));
		return;
	}

	auto Self = this->template sharedSelf<FCaptureSnapshotReactor>();
	tongsim_lite::capture::CaptureSnapshotRequest RequestCopy = Req;

	AsyncTask(ENamedThreads::GameThread, [Self, RequestCopy]() mutable
	{
		tongsim_lite::capture::CaptureFrame ResponseFrame;
		ResponseStatus Status = UCaptureGrpcSubsystem::CaptureSnapshotInternal(RequestCopy, ResponseFrame);
		if (Status.ok())
		{
			Self->writeAndFinish(ResponseFrame);
		}
		else
		{
			Self->finish(Status);
		}
	});
}

// -------------------------- Unary Handlers --------------------------

ResponseStatus UCaptureGrpcSubsystem::ListCaptureCameras(tongsim_lite::capture::ListCaptureCamerasRequest&, tongsim_lite::capture::ListCaptureCamerasResponse& Resp)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	for (auto& Pair : Instance->CameraStates)
	{
		FCaptureCameraState& State = Pair.Value;
		if (!State.CameraActor.IsValid())
		{
			continue;
		}
		tongsim_lite::capture::CaptureCameraDescriptor* Desc = Resp.add_cameras();
		GuidToObjectId(Pair.Key, *Desc->mutable_camera()->mutable_id());
		Desc->mutable_camera()->set_name(TCHAR_TO_UTF8(*GetNameSafe(State.CameraActor.Get())));
		Desc->mutable_camera()->set_class_path(TCHAR_TO_UTF8(*State.CameraActor->GetClass()->GetPathName()));
		*Desc->mutable_params() = State.ProtoParams;
		*Desc->mutable_status() = State.ProtoStatus;
	}
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::CreateCaptureCamera(tongsim_lite::capture::CreateCaptureCameraRequest& Req, tongsim_lite::capture::CreateCaptureCameraResponse& Resp)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	UWorld* World = GetGameWorld();
	if (!IsValid(World))
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Game world unavailable");
	}

	FName CaptureName = Req.capture_name().empty() ? FName(*FString::Printf(TEXT("GrpcCapture_%d"), Instance->CameraStates.Num() + 1)) : FName(UTF8_TO_TCHAR(Req.capture_name().c_str()));
	FTransform WorldTransform = FromProtoTransform(Req.world_transform());
	FTSCaptureCameraParams Params;
	FromProtoParams(Req.params(), Params);

	FActorSpawnParameters SpawnParams;
	SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	ATSCaptureCameraActor* Camera = World->SpawnActor<ATSCaptureCameraActor>(ATSCaptureCameraActor::StaticClass(), WorldTransform, SpawnParams);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::UNKNOWN, "Failed to spawn camera");
	}
	Camera->CaptureId = CaptureName;
	Camera->Params = Params;
	if (!UTSCaptureBPLibrary::UpdateCameraParams(Camera, Params))
	{
		Camera->Destroy();
		return ResponseStatus(grpc::StatusCode::UNKNOWN, "Failed to apply params");
	}

	if (!Req.attach_parent().guid().empty())
	{
		FGuid ParentGuid;
		if (!ObjectIdToGuid(Req.attach_parent(), ParentGuid))
		{
			Camera->Destroy();
			return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Invalid parent id");
		}
		if (UTSGrpcSubsystem* Grpc = Instance->ResolveGrpcSubsystem())
		{
			AActor* Parent = Grpc->FindActorByGuid(ParentGuid);
			if (!IsValid(Parent))
			{
				Camera->Destroy();
				return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Parent actor not found");
			}
			UTSCaptureBPLibrary::AttachCaptureCamera(Camera, Parent, UTF8_TO_TCHAR(Req.attach_socket().c_str()), Req.keep_world());
		}
	}

	FGuid CameraGuid;
	if (UTSGrpcSubsystem* Grpc = Instance->ResolveGrpcSubsystem())
	{
		CameraGuid = Grpc->FindGuidByActor(Camera);
	}
	if (!CameraGuid.IsValid())
	{
		Camera->Destroy();
		return ResponseStatus(grpc::StatusCode::UNKNOWN, "Camera GUID unavailable");
	}

	Instance->EnsureCameraState(CameraGuid, Camera);
	GuidToObjectId(CameraGuid, *Resp.mutable_camera()->mutable_id());
	Resp.mutable_camera()->set_name(TCHAR_TO_UTF8(*Camera->GetName()));
	Resp.mutable_camera()->set_class_path(TCHAR_TO_UTF8(*Camera->GetClass()->GetPathName()));
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::DestroyCaptureCamera(tongsim_lite::capture::DestroyCaptureCameraRequest& Req, tongsim_lite::common::Empty&)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}
	if (Req.force_stop_capture())
	{
		UTSCaptureBPLibrary::StopCapture(Camera);
	}
	UTSCaptureBPLibrary::DestroyCaptureCamera(Camera);
	Instance->CameraStates.Remove(CameraGuid);
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::SetCaptureCameraPose(tongsim_lite::capture::SetCaptureCameraPoseRequest& Req, tongsim_lite::common::Empty&)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}
	UTSCaptureBPLibrary::SetCaptureCameraPose(Camera, FromProtoTransform(Req.world_transform()));
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::UpdateCaptureCameraParams(tongsim_lite::capture::UpdateCaptureCameraParamsRequest& Req, tongsim_lite::capture::UpdateCaptureCameraParamsResponse& Resp)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}
	if (UTSCaptureSubsystem* CaptureSubsystem = Instance->ResolveCaptureSubsystem())
	{
		if (CaptureSubsystem->IsCapturing(Camera->CaptureId))
		{
			return ResponseStatus(grpc::StatusCode::FAILED_PRECONDITION, "Camera is capturing");
		}
	}
	FTSCaptureCameraParams Params;
	FromProtoParams(Req.params(), Params);
	if (!UTSCaptureBPLibrary::UpdateCameraParams(Camera, Params))
	{
		return ResponseStatus(grpc::StatusCode::UNKNOWN, "Failed to apply params");
	}
	Camera->Params = Params;
	if (FCaptureCameraState* State = Instance->EnsureCameraState(CameraGuid, Camera))
	{
		State->ProtoParams = ToProtoParams(Params);
	}
	*Resp.mutable_applied_params() = ToProtoParams(Params);
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::AttachCaptureCamera(tongsim_lite::capture::AttachCaptureCameraRequest& Req, tongsim_lite::common::Empty&)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}
	FGuid ParentGuid;
	if (!ObjectIdToGuid(Req.parent_actor_id(), ParentGuid))
	{
		return ResponseStatus(grpc::StatusCode::INVALID_ARGUMENT, "Invalid parent id");
	}
	if (UTSGrpcSubsystem* Grpc = Instance->ResolveGrpcSubsystem())
	{
		AActor* Parent = Grpc->FindActorByGuid(ParentGuid);
		if (!IsValid(Parent))
		{
			return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Parent actor not found");
		}
		UTSCaptureBPLibrary::AttachCaptureCamera(Camera, Parent, UTF8_TO_TCHAR(Req.socket_name().c_str()), Req.keep_world());
	}
	return ResponseStatus::OK;
}

ResponseStatus UCaptureGrpcSubsystem::GetCaptureStatus(tongsim_lite::capture::GetCaptureStatusRequest& Req, tongsim_lite::capture::GetCaptureStatusResponse& Resp)
{
	if (!Instance)
	{
		return ResponseStatus(grpc::StatusCode::UNAVAILABLE, "Capture subsystem unavailable");
	}
	FGuid CameraGuid;
	ATSCaptureCameraActor* Camera = Instance->FindCameraActorById(Req.camera_id(), CameraGuid);
	if (!IsValid(Camera))
	{
		return ResponseStatus(grpc::StatusCode::NOT_FOUND, "Camera not found");
	}
	FTSCaptureStatus Status;
	if (UTSCaptureSubsystem* CaptureSubsystem = Instance->ResolveCaptureSubsystem())
	{
		if (!CaptureSubsystem->GetStatus(Camera->CaptureId, Status))
		{
			return ResponseStatus(grpc::StatusCode::UNKNOWN, "Failed to query status");
		}
	}
	Resp.mutable_status()->CopyFrom(ToProtoStatus(Status));
	if (FCaptureCameraState* State = Instance->EnsureCameraState(CameraGuid, Camera))
	{
		State->ProtoStatus = Resp.status();
	}
	return ResponseStatus::OK;
}
