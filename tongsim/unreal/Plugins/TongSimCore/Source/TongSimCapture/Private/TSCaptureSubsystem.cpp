#include "TSCaptureSubsystem.h"
#include "TSCaptureViewExtension.h"
#include "TSCaptureDepthCompute.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/TextureRenderTarget2D.h"

#include "RHI.h"
#include "RHIResources.h"
#include "RenderResource.h"
#include "Containers/Ticker.h"
#include "RHIGPUReadback.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "Async/Async.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Modules/ModuleManager.h"
#include "ShowFlags.h"
#include "SceneView.h"
#include "SceneRendering.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "ScreenPass.h"
#include "RenderGraphUtils.h"
#include "PixelFormat.h"
#include "RHIGlobals.h"
#include "Math/Color.h"
#include "RenderingThread.h"
#include "Engine/World.h"
#include "HAL/PlatformProcess.h"
#include "Engine/Engine.h"
#include <cfloat>

DEFINE_LOG_CATEGORY(LogTongSimCapture);

namespace TSCapture_Internal
{
	static constexpr int32 kDefaultRingCapacity = 3;
	static constexpr int32 kMaxQueuedRequests = 2;

	struct FCaptureRequest
	{
		FTSCaptureNode::FPendingMeta Meta;
		FSceneViewStateInterface* ViewState = nullptr;
	};

	struct FRenderState
	{
		TQueue<FCaptureRequest> PendingRequests;
		int32 PendingRequestCount = 0;
		FCaptureRequest InFlight;
		TUniquePtr<FRHIGPUTextureReadback> ColorReadback;
		TUniquePtr<FRHIGPUTextureReadback> DepthReadback;
		TUniquePtr<FTSCaptureDepthComputeDevice> DepthComputeDevice;
		bool bColorInFlight = false;
		bool bDepthInFlight = false;
		TWeakPtr<FTSCaptureNode> NodeWeak;
		FSceneViewStateInterface* ViewState = nullptr;
		FName CaptureId;
	};

	static TMap<FName, TSharedPtr<FRenderState>> GRenderStates;
}

using namespace TSCapture_Internal;

static UWorld* GetSubsystemWorld(UGameInstance* GI)
{
	return GI ? GI->GetWorld() : nullptr;
}

// Global per-node readback storage (render thread only)

void UTSCaptureSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_Initialize);
	Super::Initialize(Collection);

	// Register view extension for future RDG hooks and to keep alignment with design.
	ViewExtension = FSceneViewExtensions::NewExtension<FTSCaptureViewExtension>(this);

	// Ensure ImageWrapper is loaded on the game thread before any async workers use it.
	// This avoids any module-load work happening on thread pool threads later.
	{
		TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_PreloadImageWrapper);
		FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
	}

	// Use ticker for driving capture cadence and readback pump
	TickerHandle = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateUObject(this, &UTSCaptureSubsystem::Tick));

	// Bind to world cleanup so we auto-stop on map changes for this world
	WorldCleanupHandle = FWorldDelegates::OnWorldCleanup.AddUObject(this, &UTSCaptureSubsystem::HandleWorldCleanup);
}

void UTSCaptureSubsystem::Deinitialize()
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_Deinitialize);
	// Stop ticking first to avoid scheduling any more work
	FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
	TickerHandle.Reset();

	// Unbind world cleanup delegate
	if (WorldCleanupHandle.IsValid())
	{
		FWorldDelegates::OnWorldCleanup.Remove(WorldCleanupHandle);
		WorldCleanupHandle.Reset();
	}

	// Clear render-thread state and flush to guarantee all queued GPU readbacks are finalized
	ENQUEUE_RENDER_COMMAND(TSCapture_ClearAll)([](FRHICommandListImmediate& RHICmdList)
	{
		GRenderStates.Empty();
	});
	FlushRenderingCommands();

	// Destroy any spawned actors we own explicitly (world teardown would normally handle this, but be explicit)
	for (auto& Pair : Registry)
	{
		if (TSharedPtr<FTSCaptureNode> Node = Pair.Value)
		{
			if (Node->bOwnsActor)
			{
				if (AActor* Actor = Node->OwnerActor.Get())
				{
					Actor->Destroy();
				}
			}
			else
			{
				if (USceneCaptureComponent2D* SceneCap = Node->ColorCapture.Get())
				{
					SceneCap->DestroyComponent();
				}
			}
		}
	}
	Registry.Empty();
	ViewExtension.Reset();

	Super::Deinitialize();
}

bool UTSCaptureSubsystem::StartCapture(const FName CaptureId, int32 Width, int32 Height, float FovDegrees, const FTransform& WorldTransform, float Qps, bool bEnableDepth)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_StartCapture);
	if (CaptureId.IsNone())
	{
		UE_LOG(LogTongSimCapture, Error, TEXT("StartCapture requires a non-empty CaptureId"));
		return false;
	}

	if (Registry.Contains(CaptureId))
	{
		UE_LOG(LogTongSimCapture, Warning, TEXT("CaptureId %s already exists; reconfiguring."), *CaptureId.ToString());
	}

	UWorld* World = GetSubsystemWorld(GetGameInstance());
	if (!World)
	{
		return false;
	}

	FActorSpawnParameters SpawnParams;
	SpawnParams.Name = CaptureId;
	SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	AActor* Actor = World->SpawnActor<AActor>(AActor::StaticClass(), WorldTransform, SpawnParams);
	if (!Actor)
	{
		return false;
	}

	if (!Actor->GetRootComponent())
	{
		USceneComponent* Root = NewObject<USceneComponent>(Actor, TEXT("TSCaptureRoot"));
		Actor->SetRootComponent(Root);
		Root->RegisterComponent();
	}

	// Make node
	TSharedPtr<FTSCaptureNode> Node = MakeShared<FTSCaptureNode>();
	Node->CaptureId = CaptureId;
	Node->OwnerActor = Actor;
	Node->bOwnsActor = true;
	Node->Config.Width = Width;
	Node->Config.Height = Height;
	Node->Config.Fov = FovDegrees;
	Node->Config.Qps = Qps;
	Node->Config.bEnableDepth = bEnableDepth;

	// Initialize components/targets on game thread
	EnsureTargetsAndComponents_GameThread(Node);

	ENQUEUE_RENDER_COMMAND(TSCapture_RegisterNode)([CaptureId, NodeWeak = TWeakPtr<FTSCaptureNode>(Node)](FRHICommandListImmediate& RHICmdList)
	{
		TSharedPtr<FRenderState>& StatePtr = GRenderStates.FindOrAdd(CaptureId);
		if (!StatePtr.IsValid())
		{
			StatePtr = MakeShared<FRenderState>();
		}
		FRenderState& State = *StatePtr;
		State.NodeWeak = NodeWeak;
		State.PendingRequests.Empty();
		State.PendingRequestCount = 0;
		State.InFlight = FCaptureRequest();
		State.bColorInFlight = false;
		State.bDepthInFlight = false;

		const ERHIFeatureLevel::Type FeatureLevel = GMaxRHIFeatureLevel;
		const EShaderPlatform ShaderPlatform = GShaderPlatformForFeatureLevel[FeatureLevel];
		State.DepthComputeDevice = MakeUnique<FTSCaptureDepthComputeDevice>(ShaderPlatform, FeatureLevel);

		if (TSharedPtr<FTSCaptureNode> NodeSP = NodeWeak.Pin())
		{
			State.ViewState = NodeSP->ViewState;
		}
		else
		{
			State.ViewState = nullptr;
		}
		State.CaptureId = CaptureId;
	});

	Registry.Add(CaptureId, Node);
	UE_LOG(LogTongSimCapture, Log, TEXT("[%s] Started capture (%dx%d, FOV=%.2f, QPS=%.2f, Depth=%s)"), *CaptureId.ToString(), Width, Height, FovDegrees, Qps, bEnableDepth?TEXT("On"):TEXT("Off"));
	return true;
}

bool UTSCaptureSubsystem::StopCapture(const FName CaptureId)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_StopCapture);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		ENQUEUE_RENDER_COMMAND(TSCapture_RemoveNode)([CaptureId](FRHICommandListImmediate& RHICmdList)
		{
			GRenderStates.Remove(CaptureId);
		});
		if (Node->bOwnsActor)
		{
			if (AActor* Actor = Node->OwnerActor.Get())
			{
				Actor->Destroy();
			}
		}
		else
		{
			if (USceneCaptureComponent2D* SceneCap = Node->ColorCapture.Get())
			{
				SceneCap->DestroyComponent();
			}
		}
		Registry.Remove(CaptureId);
		UE_LOG(LogTongSimCapture, Log, TEXT("[%s] Stopped capture"), *CaptureId.ToString());
		return true;
	}
	return false;
}

void UTSCaptureSubsystem::StopAllCaptures()
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_StopAllCaptures);
	if (Registry.Num() == 0)
	{
		return;
	}

	// Remove corresponding render-thread states for our registry entries
	{
		TArray<FName> Keys;
		Keys.Reserve(Registry.Num());
		for (auto& Pair : Registry)
		{
			Keys.Add(Pair.Key);
		}
		ENQUEUE_RENDER_COMMAND(TSCapture_RemoveStatesForStopAll)([Keys = MoveTemp(Keys)](FRHICommandListImmediate& RHICmdList) mutable
		{
			for (const FName& Key : Keys)
			{
				GRenderStates.Remove(Key);
			}
		});
	}

	// Destroy spawned actors (owned by subsystem) and clear registry
	for (auto& Pair : Registry)
	{
		if (TSharedPtr<FTSCaptureNode> Node = Pair.Value)
		{
			if (Node->bOwnsActor)
			{
				if (AActor* Actor = Node->OwnerActor.Get())
				{
					Actor->Destroy();
				}
			}
			else
			{
				if (USceneCaptureComponent2D* SceneCap = Node->ColorCapture.Get())
				{
					SceneCap->DestroyComponent();
				}
			}
		}
	}
	Registry.Empty();
}

bool UTSCaptureSubsystem::StartCaptureOnActor(const FName CaptureId, AActor* OwnerActor, int32 Width, int32 Height, float FovDegrees, float Qps, bool bEnableDepth)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_StartCaptureOnActor);
	if (CaptureId.IsNone() || OwnerActor == nullptr)
	{
		UE_LOG(LogTongSimCapture, Error, TEXT("StartCaptureOnActor requires valid CaptureId and OwnerActor"));
		return false;
	}
	if (Registry.Contains(CaptureId))
	{
		UE_LOG(LogTongSimCapture, Warning, TEXT("CaptureId %s already exists"), *CaptureId.ToString());
		return false;
	}

	if (!OwnerActor->GetRootComponent())
	{
		USceneComponent* Root = NewObject<USceneComponent>(OwnerActor, TEXT("TSCaptureRoot"));
		OwnerActor->SetRootComponent(Root);
		Root->RegisterComponent();
	}

	TSharedPtr<FTSCaptureNode> Node = MakeShared<FTSCaptureNode>();
	Node->CaptureId = CaptureId;
	Node->OwnerActor = OwnerActor;
	Node->bOwnsActor = false;
	Node->Config.Width = Width;
	Node->Config.Height = Height;
	Node->Config.Fov = FovDegrees;
	Node->Config.Qps = Qps;
	Node->Config.bEnableDepth = bEnableDepth;

	EnsureTargetsAndComponents_GameThread(Node);

	ENQUEUE_RENDER_COMMAND(TSCapture_RegisterNodeOnActor)([CaptureId, NodeWeak = TWeakPtr<FTSCaptureNode>(Node)](FRHICommandListImmediate& RHICmdList)
	{
		TSharedPtr<FRenderState>& StatePtr = GRenderStates.FindOrAdd(CaptureId);
		if (!StatePtr.IsValid())
		{
			StatePtr = MakeShared<FRenderState>();
		}
		FRenderState& State = *StatePtr;
		State.NodeWeak = NodeWeak;
		State.PendingRequests.Empty();
		State.PendingRequestCount = 0;
		State.InFlight = FCaptureRequest();
		State.bColorInFlight = false;
		State.bDepthInFlight = false;

		const ERHIFeatureLevel::Type FeatureLevel = GMaxRHIFeatureLevel;
		const EShaderPlatform ShaderPlatform = GShaderPlatformForFeatureLevel[FeatureLevel];
		State.DepthComputeDevice = MakeUnique<FTSCaptureDepthComputeDevice>(ShaderPlatform, FeatureLevel);

		if (TSharedPtr<FTSCaptureNode> NodeSP = NodeWeak.Pin())
		{
			State.ViewState = NodeSP->ViewState;
		}
		else
		{
			State.ViewState = nullptr;
		}
		State.CaptureId = CaptureId;
	});

	Registry.Add(CaptureId, Node);
	UE_LOG(LogTongSimCapture, Log, TEXT("[%s] Started capture on actor (%dx%d, FOV=%.2f, QPS=%.2f, Depth=%s)"), *CaptureId.ToString(), Width, Height, FovDegrees, Qps, bEnableDepth?TEXT("On"):TEXT("Off"));
	return true;
}

bool UTSCaptureSubsystem::CaptureSnapshotOnActor(const FName CaptureId, AActor* OwnerActor, int32 Width, int32 Height, float FovDegrees, bool bEnableDepth, FTSCaptureFrame& OutFrame, float TimeoutSeconds)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_CaptureSnapshot);
	if (IsCapturing(CaptureId) || OwnerActor == nullptr)
	{
		return false;
	}

	TSharedPtr<FTSCaptureNode> Node = MakeShared<FTSCaptureNode>();
	Node->CaptureId = CaptureId;
	Node->OwnerActor = OwnerActor;
	Node->bOwnsActor = false;
	Node->Config.Width = Width;
	Node->Config.Height = Height;
	Node->Config.Fov = FovDegrees;
	Node->Config.Qps = 0.f;
	Node->Config.bEnableDepth = bEnableDepth;

	EnsureTargetsAndComponents_GameThread(Node);

	ENQUEUE_RENDER_COMMAND(TSCapture_RegisterSnapshotNode)([CaptureId, NodeWeak = TWeakPtr<FTSCaptureNode>(Node)](FRHICommandListImmediate& RHICmdList)
	{
		TSharedPtr<FRenderState>& StatePtr = GRenderStates.FindOrAdd(CaptureId);
		if (!StatePtr.IsValid())
		{
			StatePtr = MakeShared<FRenderState>();
		}
		FRenderState& State = *StatePtr;
		State.NodeWeak = NodeWeak;
		State.PendingRequests.Empty();
		State.PendingRequestCount = 0;
		State.InFlight = FCaptureRequest();
		State.bColorInFlight = false;
		State.bDepthInFlight = false;
		State.ViewState = nullptr;
		State.CaptureId = CaptureId;
	});

	Registry.Add(CaptureId, Node);
	EnqueueCaptureAndReadback_GameThread(Node);

	const double EndTime = FPlatformTime::Seconds() + FMath::Max(0.01, TimeoutSeconds);
	bool bGotFrame = false;
	while (FPlatformTime::Seconds() < EndTime)
	{
		PumpReadbacks_RenderThread();
		FlushRenderingCommands();
		TSharedPtr<FTSCaptureFrame> Latest;
		if (Node->FrameQueue.Dequeue(Latest))
		{
			Node->QueueCount.DecrementExchange();
			if (Latest.IsValid())
			{
				OutFrame = *Latest.Get();
				bGotFrame = true;
				break;
			}
		}
		FPlatformProcess::Sleep(0.001f);
	}

	ENQUEUE_RENDER_COMMAND(TSCapture_RemoveSnapshotNode)([CaptureId](FRHICommandListImmediate& RHICmdList)
	{
		GRenderStates.Remove(CaptureId);
	});
	if (USceneCaptureComponent2D* SceneCap = Node->ColorCapture.Get())
	{
		SceneCap->DestroyComponent();
	}
	Registry.Remove(CaptureId);

	return bGotFrame;
}

bool UTSCaptureSubsystem::GetLatestFrame(const FName CaptureId, FTSCaptureFrame& OutFrame)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_GetLatestFrame);
	TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId);
	if (!NodePtr)
	{
		return false;
	}
	TSharedPtr<FTSCaptureNode> Node = *NodePtr;

	// Drain queue to latest
	TSharedPtr<FTSCaptureFrame> Latest;
	while (true)
	{
		TSharedPtr<FTSCaptureFrame> Tmp;
		if (!Node->FrameQueue.Dequeue(Tmp))
		{
			break;
		}
		Latest = MoveTemp(Tmp);
		Node->QueueCount.DecrementExchange();
	}

	if (Latest.IsValid())
	{
		OutFrame = *Latest.Get();
		UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] GetLatestFrame -> FrameId=%llu RgbaBytes=%d DepthCount=%d QueueAfter=%d"),
		       *CaptureId.ToString(),
		       (unsigned long long)OutFrame.FrameId,
		       OutFrame.Rgba8.Num(),
		       OutFrame.DepthR32.Num(),
		       Node->QueueCount.Load());
		return true;
	}

	if (Node->QueueCount.Load() > 0)
	{
		UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] GetLatestFrame found no dequeued frame but QueueCount=%d"), *CaptureId.ToString(), Node->QueueCount.Load());
	}
	return false;
}

bool UTSCaptureSubsystem::GetLatestCompressedFrame(const FName CaptureId, FTSCaptureCompressedFrame& OutFrame)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_GetLatestCompressedFrame);
	TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId);
	if (!NodePtr)
	{
		return false;
	}
	TSharedPtr<FTSCaptureNode> Node = *NodePtr;

	TSharedPtr<FTSCaptureCompressedFrame> Latest;
	while (true)
	{
		TSharedPtr<FTSCaptureCompressedFrame> Tmp;
		if (!Node->CompressedQueue.Dequeue(Tmp))
		{
			break;
		}
		Latest = MoveTemp(Tmp);
		Node->CompressedQueueCount.DecrementExchange();
	}

	if (Latest.IsValid())
	{
		OutFrame = *Latest.Get();
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::SetDepthEnabled(const FName CaptureId, bool bEnableDepth)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetDepthEnabled);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		(*NodePtr)->Config.bEnableDepth = bEnableDepth;
		EnsureTargetsAndComponents_GameThread(*NodePtr);
		UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] Depth %s"), *CaptureId.ToString(), bEnableDepth ? TEXT("Enabled") : TEXT("Disabled"));
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::SetDepthMode(const FName CaptureId, ETSCaptureDepthMode Mode)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetDepthMode);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		Node->Config.DepthMode = Mode;
		EnsureTargetsAndComponents_GameThread(Node);
		UE_LOG(LogTongSimCapture, Log, TEXT("[%s] DepthMode set to %d"), *CaptureId.ToString(), (int32)Mode);
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::Reconfigure(const FName CaptureId, int32 Width, int32 Height, float FovDegrees, float Qps)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_Reconfigure);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		bool bNeedsResize = (Node->Config.Width != Width) || (Node->Config.Height != Height);
		Node->Config.Width = Width;
		Node->Config.Height = Height;
		Node->Config.Fov = FovDegrees;
		Node->Config.Qps = Qps;
		if (bNeedsResize)
		{
			EnsureTargetsAndComponents_GameThread(Node);
			UE_LOG(LogTongSimCapture, Log, TEXT("[%s] Reconfigured to %dx%d, FOV=%.2f, QPS=%.2f"), *CaptureId.ToString(), Width, Height, FovDegrees, Qps);
		}
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::SetCompression(const FName CaptureId, ETSRgbCodec RgbCodec, ETSDepthCodec DepthCodec, int32 JpegQuality)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetCompression);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		Node->RgbCodec = RgbCodec;
		Node->DepthCodec = DepthCodec;
		Node->JpegQuality = FMath::Clamp(JpegQuality, 1, 100);
		UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] Compression set: RGB=%d, Depth=%d, Q=%d"), *CaptureId.ToString(), (int32)RgbCodec, (int32)DepthCodec, Node->JpegQuality);
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::SetCaptureTransform(const FName CaptureId, const FTransform& WorldTransform)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetTransform);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		if (AActor* Actor = Node->OwnerActor.Get())
		{
			Actor->SetActorTransform(WorldTransform);
			return true;
		}
	}
	return false;
}

bool UTSCaptureSubsystem::SetColorCaptureSettings(const FName CaptureId, ESceneCaptureSource CaptureSource, ETextureRenderTargetFormat RenderTargetFormat, bool bEnablePostProcess, bool bEnableTemporalAA)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetColorSettings);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		Node->Config.ColorSource = CaptureSource;
		Node->Config.ColorRenderTargetFormat = RenderTargetFormat;
		Node->Config.bEnablePostProcess = bEnablePostProcess;
		Node->Config.bEnableTemporalAA = bEnableTemporalAA;
		EnsureTargetsAndComponents_GameThread(Node);
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::SetDepthRange(const FName CaptureId, float NearPlane, float FarPlane)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_SetDepthRange);
	NearPlane = FMath::Max(NearPlane, KINDA_SMALL_NUMBER);
	FarPlane = FMath::Max(FarPlane, NearPlane + KINDA_SMALL_NUMBER);
	if (TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		TSharedPtr<FTSCaptureNode> Node = *NodePtr;
		Node->Config.DepthNearPlane = NearPlane;
		Node->Config.DepthFarPlane = FarPlane;
		EnsureTargetsAndComponents_GameThread(Node);
		return true;
	}
	return false;
}

bool UTSCaptureSubsystem::Tick(float DeltaSeconds)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_Tick);
	UWorld* World = GetSubsystemWorld(GetGameInstance());
	if (!World)
	{
		return true; // keep ticking attempt
	}

	const double Now = World->GetTimeSeconds();

	// Schedule captures based on QPS
	for (auto& Kvp : Registry)
	{
		TSharedPtr<FTSCaptureNode> Node = Kvp.Value;
		if (!Node.IsValid()) continue;

		const double Interval = (Node->Config.Qps > 0.0f) ? (1.0 / Node->Config.Qps) : 0.0;
		if (Interval <= 0.0)
		{
			continue;
		}

		if (Node->LastCaptureGameTime < 0.0 || (Now - Node->LastCaptureGameTime) >= Interval)
		{
			Node->LastCaptureGameTime = Now;
			EnsureTargetsAndComponents_GameThread(Node);
			EnqueueCaptureAndReadback_GameThread(Node);
		}
	}

	// Pump readback completion on render thread
	PumpReadbacks_RenderThread();

	return true;
}

void UTSCaptureSubsystem::EnsureTargetsAndComponents_GameThread(const TSharedPtr<FTSCaptureNode>& Node)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_EnsureTargets);
	if (!Node.IsValid()) return;
	AActor* Actor = Node->OwnerActor.Get();
	if (!Actor) return;

	USceneCaptureComponent2D* SceneCap = Node->ColorCapture.Get();
	if (!SceneCap)
	{
		SceneCap = NewObject<USceneCaptureComponent2D>(Actor, USceneCaptureComponent2D::StaticClass(), NAME_None, RF_Transient);
		SceneCap->SetupAttachment(Actor->GetRootComponent());
		SceneCap->RegisterComponent();
		Node->ColorCapture = SceneCap;
	}
	SceneCap->bCaptureEveryFrame = false;
	SceneCap->bAlwaysPersistRenderingState = true;
	SceneCap->CaptureSource = static_cast<ESceneCaptureSource>(Node->Config.ColorSource.GetValue());
	SceneCap->FOVAngle = Node->Config.Fov;
	SceneCap->bOverride_CustomNearClippingPlane = true;
	SceneCap->CustomNearClippingPlane = Node->Config.DepthNearPlane;
	SceneCap->MaxViewDistanceOverride = Node->Config.DepthFarPlane;

	FEngineShowFlags ShowFlags(ESFIM_Game);
	ShowFlags.SetPostProcessing(Node->Config.bEnablePostProcess);
	ShowFlags.SetTemporalAA(Node->Config.bEnableTemporalAA);
	ShowFlags.SetAntiAliasing(Node->Config.bEnableTemporalAA);
	SceneCap->ShowFlags = ShowFlags;

	// Color RT
	bool bNewColor = false;
	UTextureRenderTarget2D* ColorRT = Node->ColorRT.Get();
	if (!ColorRT)
	{
		ColorRT = NewObject<UTextureRenderTarget2D>(Actor, NAME_None, RF_Transient);
		bNewColor = true;
	}
	if (bNewColor || ColorRT->SizeX != Node->Config.Width || ColorRT->SizeY != Node->Config.Height || ColorRT->RenderTargetFormat != static_cast<ETextureRenderTargetFormat>(Node->Config.ColorRenderTargetFormat.GetValue()))
	{
		ColorRT->RenderTargetFormat = static_cast<ETextureRenderTargetFormat>(Node->Config.ColorRenderTargetFormat.GetValue());
		ColorRT->ClearColor = FLinearColor::Black;
		ColorRT->bAutoGenerateMips = false;
		const bool bForceLinear = (ColorRT->RenderTargetFormat == ETextureRenderTargetFormat::RTF_RGBA16f || ColorRT->RenderTargetFormat == ETextureRenderTargetFormat::RTF_RGBA32f);
		ColorRT->bForceLinearGamma = bForceLinear;
		ColorRT->InitAutoFormat(Node->Config.Width, Node->Config.Height);
		ColorRT->UpdateResourceImmediate(true);
		UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] Color RT init %dx%d"), *Node->CaptureId.ToString(), Node->Config.Width, Node->Config.Height);
	}
	SceneCap->TextureTarget = ColorRT;

	Node->ColorCapture = SceneCap;
	Node->ColorRT = ColorRT;
	Node->ViewState = SceneCap->GetViewState(0);
}

void UTSCaptureSubsystem::EnqueueCaptureAndReadback_GameThread(const TSharedPtr<FTSCaptureNode>& Node)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_EnqueueReadbacks);
	if (!Node.IsValid()) return;
	AActor* Actor = Node->OwnerActor.Get();
	USceneCaptureComponent2D* ColorCap = Node->ColorCapture.Get();
    if (!Actor || !ColorCap) return;

    // Prepare metadata (submit request before triggering capture so Tonemap sees it this frame)
    FTSCaptureNode::FPendingMeta Meta;
    Meta.FrameId = ++Node->FrameCounter;
    Meta.GameTimeSeconds = Node->LastCaptureGameTime;
    Meta.Width = Node->Config.Width;
    Meta.Height = Node->Config.Height;
	Meta.Pose = Actor->GetActorTransform();
	Meta.Intrinsics = MakeIntrinsics(Node->Config.Width, Node->Config.Height, Node->Config.Fov);
	Meta.DepthNear = Node->Config.DepthNearPlane;
	Meta.DepthFar = Node->Config.DepthFarPlane;
	Meta.DepthMode = Node->Config.DepthMode;
	Meta.bCaptureDepth = Node->Config.bEnableDepth && Node->Config.DepthMode != ETSCaptureDepthMode::None;
	Meta.ColorPixelFormat = PF_Unknown;
	Meta.bValid = true;

	FCaptureRequest Request;
	Request.Meta = Meta;
	Request.ViewState = Node->ViewState;

	const FName CaptureId = Node->CaptureId;
	UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] Queueing request FrameId=%llu ViewState=%p"), *CaptureId.ToString(), (unsigned long long)Meta.FrameId, Node->ViewState);
    ENQUEUE_RENDER_COMMAND(TSCapture_SubmitRequest)([CaptureId, Request](FRHICommandListImmediate& RHICmdList) mutable
    {
        if (TSharedPtr<FRenderState>* StatePtr = GRenderStates.Find(CaptureId))
        {
            if (!StatePtr->IsValid())
            {
                *StatePtr = MakeShared<FRenderState>();
            }
            FRenderState& State = *StatePtr->Get();
            while (State.PendingRequestCount >= kMaxQueuedRequests)
            {
                FCaptureRequest Discard;
                if (State.PendingRequests.Dequeue(Discard))
                {
                    State.PendingRequestCount--;
                }
                else
                {
                    UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] Render queue empty while trimming pending"), *CaptureId.ToString());
                    break;
                }
            }
            State.PendingRequests.Enqueue(Request);
            State.PendingRequestCount++;
            UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] Request enqueued FrameId=%llu Pending=%d"), *CaptureId.ToString(), (unsigned long long)Request.Meta.FrameId, State.PendingRequestCount);
        }
        else
        {
            UE_LOG(LogTongSimCapture, Warning, TEXT("[%s] Missing render state when queueing FrameId=%llu"), *CaptureId.ToString(), (unsigned long long)Request.Meta.FrameId);
        }
    });

    // Issue capture after submitting request to render thread
    ColorCap->CaptureScene();
    UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] Enqueued capture FrameId=%llu"), *Node->CaptureId.ToString(), (unsigned long long)(Node->FrameCounter));
}

void UTSCaptureSubsystem::ProcessViewAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_ProcessViewRT);

	if (!View.bIsSceneCapture)
	{
		return;
	}

	FRenderState* StatePtr = nullptr;
	FSceneViewStateInterface* ViewState = View.State;

	if (ViewState)
	{
		for (auto& Pair : GRenderStates)
		{
			if (Pair.Value.IsValid() && Pair.Value->ViewState == ViewState)
			{
				StatePtr = Pair.Value.Get();
				UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("View match via ViewState=%p"), ViewState);
				break;
			}
		}
	}

	if (!StatePtr)
	{
		const AActor* ViewOwner = View.ViewActor.Get();
		if (ViewOwner)
		{
			for (auto& Pair : GRenderStates)
			{
				if (!Pair.Value.IsValid())
				{
					continue;
				}
				if (TSharedPtr<FTSCaptureNode> Node = Pair.Value->NodeWeak.Pin())
				{
					if (Node->OwnerActor.Get() == ViewOwner)
					{
						StatePtr = Pair.Value.Get();
						UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("View match via OwnerActor=%s"), *ViewOwner->GetFName().ToString());
						break;
					}
				}
			}
		}
	}

	if (!StatePtr && GRenderStates.Num() == 1)
	{
		auto It = GRenderStates.CreateIterator();
		if (It.Value().IsValid())
		{
			StatePtr = It.Value().Get();
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("View matched sole capture state (ViewState=%p Owner=%s)"), ViewState, View.ViewActor.IsSet() ? *View.ViewActor.Get()->GetFName().ToString() : TEXT("None"));
		}
	}

	if (!StatePtr)
	{
		UE_LOG(LogTongSimCapture, Verbose, TEXT("Unable to resolve capture state for view (ViewState=%p Owner=%s)"), ViewState, View.ViewActor.IsSet() ? *View.ViewActor.Get()->GetFName().ToString() : TEXT("None"));
		return;
	}

	FRenderState& State = *StatePtr;

	const FString CaptureIdString = State.CaptureId.ToString();
	const TCHAR* CaptureId = *CaptureIdString;


	if (!State.ViewState && ViewState)
	{
		State.ViewState = ViewState;
		UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("Bound ViewState=%p to capture state"), ViewState);
	}

	if (!State.InFlight.Meta.bValid)
	{
		FCaptureRequest NextRequest;
		if (State.PendingRequests.Dequeue(NextRequest))
		{
			State.PendingRequestCount = FMath::Max<int32>(State.PendingRequestCount - 1, 0);
			State.InFlight = MoveTemp(NextRequest);
			State.bColorInFlight = false;
			State.bDepthInFlight = false;
			{
				const FString Message = FString::Printf(TEXT("Dequeued FrameId=%llu Pending=%d"), static_cast<uint64>(State.InFlight.Meta.FrameId), State.PendingRequestCount);
				UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
			}
			if (State.InFlight.ViewState)
			{
				State.ViewState = State.InFlight.ViewState;
			}
		}
	}

	if (!State.InFlight.Meta.bValid)
	{
		UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] No in-flight request after dequeue"), CaptureId);
		return;
	}

	const FScreenPassTextureSlice SceneColorSlice = Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	if (SceneColorSlice.IsValid())
	{
		RDG_EVENT_SCOPE(GraphBuilder, "TSCapture.ColorReadback");
		const FScreenPassTexture SceneColor = FScreenPassTexture::CopyFromSlice(GraphBuilder, SceneColorSlice);
		if (SceneColor.IsValid())
		{
			State.InFlight.Meta.Width = SceneColor.ViewRect.Width();
			State.InFlight.Meta.Height = SceneColor.ViewRect.Height();
			State.InFlight.Meta.ColorPixelFormat = SceneColor.Texture->Desc.Format;
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] SceneColor texture Format=%s Extent=%dx%d ViewRect=%dx%d FrameId=%llu"),
			       CaptureId,
			       GetPixelFormatString(SceneColor.Texture->Desc.Format),
			       SceneColor.Texture->Desc.Extent.X,
			       SceneColor.Texture->Desc.Extent.Y,
			       State.InFlight.Meta.Width,
			       State.InFlight.Meta.Height,
			       static_cast<uint64>(State.InFlight.Meta.FrameId));

			if (!State.ColorReadback.IsValid())
			{
				State.ColorReadback = MakeUnique<FRHIGPUTextureReadback>(TEXT("TSCapture_Color"));
			}

			if (!State.bColorInFlight)
			{
				State.bColorInFlight = true;
				FRDGTextureRef TextureToRead = SceneColor.Texture;
				AddEnqueueCopyPass(GraphBuilder, State.ColorReadback.Get(), TextureToRead);
				{
					const FString Message = FString::Printf(TEXT("Enqueued color readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
			else if (State.ColorReadback.IsValid() && !State.ColorReadback->IsReady())
			{
				{
					const FString Message = FString::Printf(TEXT("Waiting for previous color readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
			else if (State.bColorInFlight)
			{
				{
					const FString Message = FString::Printf(TEXT("Waiting on color readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
		}
		else
		{
			{
				const FString Message = FString::Printf(TEXT("SceneColor invalid FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
				UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] %s"), CaptureId, *Message);
			}
		}
	}
	else
	{
		{
			const FString Message = FString::Printf(TEXT("SceneColor slice missing FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
			UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] %s"), CaptureId, *Message);
		}
	}

	const bool bDoDepth = State.InFlight.Meta.bCaptureDepth;
	if (bDoDepth)
	{
		if (!State.DepthReadback.IsValid())
		{
			State.DepthReadback = MakeUnique<FRHIGPUTextureReadback>(TEXT("TSCapture_Depth"));
		}

		const FViewInfo& ViewInfo = static_cast<const FViewInfo&>(View);
		if (!ViewInfo.ShaderMap)
		{
			const FString Message = FString::Printf(TEXT("ShaderMap missing; skipping depth readback this frame (FrameId=%llu)"), static_cast<uint64>(State.InFlight.Meta.FrameId));
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
		}
		else if (!State.bDepthInFlight)
		{
			if (!State.DepthComputeDevice || !State.DepthComputeDevice->IsValid())
			{
				const ERHIFeatureLevel::Type FeatureLevel = View.GetFeatureLevel();
				const EShaderPlatform ShaderPlatform = GShaderPlatformForFeatureLevel[FeatureLevel];
				State.DepthComputeDevice = MakeUnique<FTSCaptureDepthComputeDevice>(ShaderPlatform, FeatureLevel);
			}

			FRDGTextureRef SceneDepthTexture = nullptr;
			if (const FSceneTextures* SceneTextures = ViewInfo.GetSceneTexturesChecked())
			{
				// Prefer the shader-readable resolve depth when available
				SceneDepthTexture = SceneTextures->Depth.Resolve ? SceneTextures->Depth.Resolve : SceneTextures->Depth.Target;
			}

			if (SceneDepthTexture)
			{
				FRDGTextureRef LinearDepth = nullptr;
				const bool bDepthComputeSupported = State.DepthComputeDevice && State.DepthComputeDevice->IsValid();
				if (bDepthComputeSupported)
				{
					LinearDepth = State.DepthComputeDevice->AddDepthPass(
						GraphBuilder,
						ViewInfo,
						SceneDepthTexture,
						State.InFlight.Meta.Width,
						State.InFlight.Meta.Height,
						State.InFlight.Meta.DepthMode,
						State.InFlight.Meta.DepthNear,
						State.InFlight.Meta.DepthFar);
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] DepthParams Mode=%d Near=%.3f Far=%.3f"), CaptureId, (int32)State.InFlight.Meta.DepthMode, State.InFlight.Meta.DepthNear, State.InFlight.Meta.DepthFar);
				}
				else
				{
					const FString Message = FString::Printf(TEXT("Depth compute unsupported; skipping depth readback this frame (FrameId=%llu)"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] %s"), CaptureId, *Message);
				}

				if (LinearDepth)
				{
					State.bDepthInFlight = true;
					AddEnqueueCopyPass(GraphBuilder, State.DepthReadback.Get(), LinearDepth);
					const FString Message = FString::Printf(TEXT("Enqueued depth readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
				else if (bDepthComputeSupported)
				{
					const FString Message = FString::Printf(TEXT("Depth pass returned null (Shader unavailable) FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
					UE_LOG(LogTongSimCapture, Warning, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
			else
			{
				const FString Message = FString::Printf(TEXT("SceneDepth missing FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
				UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] %s"), CaptureId, *Message);
			}
		}
		else if (State.DepthReadback.IsValid() && !State.DepthReadback->IsReady())
		{
			const FString Message = FString::Printf(TEXT("Waiting for previous depth readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
		}
		else if (State.bDepthInFlight)
		{
			const FString Message = FString::Printf(TEXT("Waiting on depth readback FrameId=%llu"), static_cast<uint64>(State.InFlight.Meta.FrameId));
			UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
		}
	}
	else
	{
		State.bDepthInFlight = false;
	}
}

bool UTSCaptureSubsystem::FindNodeByOwnerActor(const AActor* Owner, TSharedPtr<FTSCaptureNode>& OutNode) const
{
	for (const auto& Pair : Registry)
	{
		const TSharedPtr<FTSCaptureNode>& Node = Pair.Value;
		if (Node.IsValid() && Node->OwnerActor.Get() == Owner)
		{
			OutNode = Node;
			return true;
		}
	}
	return false;
}

bool UTSCaptureSubsystem::GetStatus(const FName CaptureId, FTSCaptureStatus& OutStatus) const
{
	if (const TSharedPtr<FTSCaptureNode>* NodePtr = Registry.Find(CaptureId))
	{
		const TSharedPtr<FTSCaptureNode>& Node = *NodePtr;
		if (Node.IsValid())
		{
			OutStatus.bCapturing = true;
			OutStatus.QueueCount = Node->QueueCount.Load();
			OutStatus.CompressedQueueCount = Node->CompressedQueueCount.Load();
			OutStatus.Width = Node->Config.Width;
			OutStatus.Height = Node->Config.Height;
			OutStatus.FovDegrees = Node->Config.Fov;
			OutStatus.DepthMode = Node->Config.DepthMode;
			return true;
		}
	}
	OutStatus = FTSCaptureStatus();
	OutStatus.bCapturing = false;
	return false;
}

void UTSCaptureSubsystem::PumpReadbacks_RenderThread()
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_PumpReadbacks);
	// We execute the polling on the render thread to minimize stalls.
	TWeakObjectPtr<UTSCaptureSubsystem> WeakThis = this;
	ENQUEUE_RENDER_COMMAND(TSCapture_PollReadbacks)([WeakThis](FRHICommandListImmediate& RHICmdList)
	{
		for (auto& Kvp : GRenderStates)
		{
			TSharedPtr<FRenderState> StatePtr = Kvp.Value;
			if (!StatePtr.IsValid())
			{
				continue;
			}
			FRenderState& State = *StatePtr;
			const FString CaptureIdString = State.CaptureId.ToString();
			const TCHAR* CaptureId = *CaptureIdString;
			if (!State.InFlight.Meta.bValid)
			{
				if (State.PendingRequestCount > 0)
				{
					const FString Message = FString::Printf(TEXT("Pump sees pending=%d but no in-flight"), State.PendingRequestCount);
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
				continue;
			}

			const bool bColorReady = !State.bColorInFlight || (State.ColorReadback.IsValid() && State.ColorReadback->IsReady());
			const bool bDepthRequested = State.InFlight.Meta.bCaptureDepth;
			const bool bDepthReady = !bDepthRequested || !State.bDepthInFlight || (State.DepthReadback.IsValid() && State.DepthReadback->IsReady());

			if (!bColorReady || !bDepthReady)
			{
				const FString Message = FString::Printf(TEXT("Pump waiting color=%d depth=%d FrameId=%llu"), bColorReady, bDepthReady, static_cast<uint64>(State.InFlight.Meta.FrameId));
				UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				continue;
			}

			// Both ready (or not pending) -> lock/copy
			TSharedPtr<FTSCaptureFrame> Frame = MakeShared<FTSCaptureFrame>();
			Frame->FrameId = State.InFlight.Meta.FrameId;
			Frame->GameTimeSeconds = State.InFlight.Meta.GameTimeSeconds;
			Frame->Width = State.InFlight.Meta.Width;
			Frame->Height = State.InFlight.Meta.Height;
			Frame->Pose = State.InFlight.Meta.Pose;
			Frame->Intrinsics = State.InFlight.Meta.Intrinsics;
			Frame->GpuReadyTimestamp = FPlatformTime::Seconds();

			if (State.bColorInFlight && State.ColorReadback.IsValid())
			{
				TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_CopyColorFromReadback);
				int32 RowPitchPixels = 0;
				uint8* Src = static_cast<uint8*>(State.ColorReadback->Lock(RowPitchPixels));

				const EPixelFormat PixelFormat = State.InFlight.Meta.ColorPixelFormat;
				const int32 SourceBytesPerPixel = (PixelFormat != PF_Unknown) ? GPixelFormats[PixelFormat].BlockBytes : 4;
				const int32 SafeSourceBytesPerPixel = SourceBytesPerPixel > 0 ? SourceBytesPerPixel : 4;
				const int32 SrcRowStride = RowPitchPixels * SafeSourceBytesPerPixel;
				constexpr int32 OutputBytesPerPixel = 4;
				const int32 DstRowStride = Frame->Width * OutputBytesPerPixel;
				Frame->Rgba8.SetNumUninitialized(static_cast<int32>(Frame->Width * Frame->Height * OutputBytesPerPixel));
				uint8* Dst = Frame->Rgba8.GetData();
				FColor ConvertedSample(0, 0, 0, 0);
				FString RawSampleDescription = TEXT("N/A");
				bool bLoggedFallbackFormat = false;
				if (PixelFormat == PF_Unknown)
				{
					UE_LOG(LogTongSimCapture, Warning, TEXT("[%s] Color readback sees unknown pixel format; defaulting to float conversion. FrameId=%llu"),
					       CaptureId,
					       static_cast<uint64>(State.InFlight.Meta.FrameId));
				}
				auto DescribeRawPixel =
					[](EPixelFormat Format, const uint8* Data) -> FString
				{
					if (!Data)
					{
						return TEXT("Null");
					}
					switch (Format)
					{
					case PF_B8G8R8A8:
					case PF_R8G8B8A8:
					case PF_A8R8G8B8:
						return FString::Printf(TEXT("U8(%u,%u,%u,%u)"), Data[0], Data[1], Data[2], Data[3]);
					case PF_FloatRGBA:
						{
							const FFloat16Color* Half = reinterpret_cast<const FFloat16Color*>(Data);
							const FLinearColor Linear = Half->GetFloats();
							return FString::Printf(TEXT("F16(%f,%f,%f,%f)"), Linear.R, Linear.G, Linear.B, Linear.A);
						}
					case PF_A32B32G32R32F:
						{
							const float* Floats = reinterpret_cast<const float*>(Data);
							return FString::Printf(TEXT("F32(%f,%f,%f,%f)"), Floats[0], Floats[1], Floats[2], Floats[3]);
						}
					default:
						return FString::Printf(TEXT("Fmt%d Raw0x%02X%02X%02X%02X"), static_cast<int32>(Format), Data[0], Data[1], Data[2], Data[3]);
					}
				};

				if (Src)
				{
					RawSampleDescription = DescribeRawPixel(PixelFormat, Src);
					for (int32 y = 0; y < Frame->Height; ++y)
					{
						const uint8* SrcRow = Src + y * SrcRowStride;
						uint8* DstRow = Dst + y * DstRowStride;

						if (PixelFormat == PF_B8G8R8A8)
						{
							for (int32 x = 0; x < Frame->Width; ++x)
							{
								const uint8* SrcPixel = SrcRow + x * SafeSourceBytesPerPixel;
								uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
								DstPixel[0] = SrcPixel[0];
								DstPixel[1] = SrcPixel[1];
								DstPixel[2] = SrcPixel[2];
								DstPixel[3] = 255;
								if (y == 0 && x == 0)
								{
									ConvertedSample = FColor(DstPixel[2], DstPixel[1], DstPixel[0], DstPixel[3]);
								}
							}
						}
						else if (PixelFormat == PF_R8G8B8A8)
						{
							for (int32 x = 0; x < Frame->Width; ++x)
							{
								const uint8* SrcPixel = SrcRow + x * SafeSourceBytesPerPixel;
								uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
								const uint8 R = SrcPixel[0];
								const uint8 G = SrcPixel[1];
								const uint8 B = SrcPixel[2];
								const uint8 A = SrcPixel[3];
								DstPixel[0] = B;
								DstPixel[1] = G;
								DstPixel[2] = R;
								DstPixel[3] = 255;
								if (y == 0 && x == 0)
								{
									ConvertedSample = FColor(DstPixel[2], DstPixel[1], DstPixel[0], DstPixel[3]);
								}
							}
						}
						else if (PixelFormat == PF_A8R8G8B8)
						{
							for (int32 x = 0; x < Frame->Width; ++x)
							{
								const uint8* SrcPixel = SrcRow + x * SafeSourceBytesPerPixel;
								uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
								const uint8 A = SrcPixel[0];
								const uint8 R = SrcPixel[1];
								const uint8 G = SrcPixel[2];
								const uint8 B = SrcPixel[3];
								DstPixel[0] = B;
								DstPixel[1] = G;
								DstPixel[2] = R;
								DstPixel[3] = 255;
								if (y == 0 && x == 0)
								{
									ConvertedSample = FColor(DstPixel[2], DstPixel[1], DstPixel[0], DstPixel[3]);
								}
							}
						}
						else if (PixelFormat == PF_FloatRGBA)
						{
							const FFloat16Color* SrcPixels = reinterpret_cast<const FFloat16Color*>(SrcRow);
							for (int32 x = 0; x < Frame->Width; ++x)
							{
								const FLinearColor Linear = SrcPixels[x].GetFloats();
								const FColor Srgb = Linear.ToFColorSRGB();
								uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
								DstPixel[0] = Srgb.B;
								DstPixel[1] = Srgb.G;
								DstPixel[2] = Srgb.R;
								DstPixel[3] = Srgb.A;
								if (y == 0 && x == 0)
								{
									ConvertedSample = Srgb;
								}
							}
						}
						else if (PixelFormat == PF_A32B32G32R32F)
						{
							const float* SrcPixels = reinterpret_cast<const float*>(SrcRow);
							for (int32 x = 0; x < Frame->Width; ++x)
							{
								const float R = SrcPixels[x * 4 + 0];
								const float G = SrcPixels[x * 4 + 1];
								const float B = SrcPixels[x * 4 + 2];
								const float A = SrcPixels[x * 4 + 3];
								const FLinearColor Linear(R, G, B, A);
								const FColor Srgb = Linear.ToFColorSRGB();
								uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
								DstPixel[0] = Srgb.B;
								DstPixel[1] = Srgb.G;
								DstPixel[2] = Srgb.R;
								DstPixel[3] = Srgb.A;
								if (y == 0 && x == 0)
								{
									ConvertedSample = Srgb;
								}
							}
						}
						else
						{
							// Fallback: attempt to treat source as float RGBA if stride allows, else zero.
							if (SafeSourceBytesPerPixel >= sizeof(float) * 4)
							{
								const float* SrcPixels = reinterpret_cast<const float*>(SrcRow);
								for (int32 x = 0; x < Frame->Width; ++x)
								{
									const float R = SrcPixels[x * 4 + 0];
									const float G = SrcPixels[x * 4 + 1];
									const float B = SrcPixels[x * 4 + 2];
									const float A = SrcPixels[x * 4 + 3];
									const FLinearColor Linear(R, G, B, A);
									const FColor Srgb = Linear.ToFColorSRGB();
									uint8* DstPixel = DstRow + x * OutputBytesPerPixel;
									DstPixel[0] = Srgb.B;
									DstPixel[1] = Srgb.G;
									DstPixel[2] = Srgb.R;
									DstPixel[3] = Srgb.A;
									if (y == 0 && x == 0)
									{
										ConvertedSample = Srgb;
									}
								}
							}
							else
							{
								FMemory::Memzero(DstRow, DstRowStride);
								if (!bLoggedFallbackFormat)
								{
									bLoggedFallbackFormat = true;
									UE_LOG(LogTongSimCapture, Warning, TEXT("[%s] Color readback fallback zero fill for unsupported format=%s FrameId=%llu"),
									       CaptureId,
									       PixelFormat != PF_Unknown ? GetPixelFormatString(PixelFormat) : TEXT("Unknown"),
									       static_cast<uint64>(State.InFlight.Meta.FrameId));
								}
							}
						}
					}
				}
				State.ColorReadback->Unlock();
				{
					const FString Message = FString::Printf(
						TEXT("Locked color readback FrameId=%llu Format=%s SrcBPP=%d RowPitch=%d RawSample=%s ConvertedSample=%s"),
						static_cast<uint64>(Frame->FrameId),
						PixelFormat != PF_Unknown ? GetPixelFormatString(PixelFormat) : TEXT("Unknown"),
						SafeSourceBytesPerPixel,
						RowPitchPixels,
						*RawSampleDescription,
						*ConvertedSample.ToString());
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
			State.bColorInFlight = false;

			if (bDepthRequested && State.bDepthInFlight && State.DepthReadback.IsValid())
			{
				TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_CopyDepthFromReadback);
				int32 RowPitchPixels = 0;
				uint8* SrcBytes = static_cast<uint8*>(State.DepthReadback->Lock(RowPitchPixels));
				const int32 BytesPerPixel = sizeof(float); // R32F
				const int32 SrcRowStride = RowPitchPixels * BytesPerPixel;
				const int32 DstRowStride = Frame->Width * BytesPerPixel;
				Frame->DepthR32.SetNumUninitialized(static_cast<int32>(Frame->Width * Frame->Height));
				uint8* DstBytes = reinterpret_cast<uint8*>(Frame->DepthR32.GetData());
				float DepthMin = FLT_MAX;
				float DepthMax = -FLT_MAX;
				float FirstDepth = 0.0f;
				bool bCapturedDepthSample = false;
				if (SrcBytes)
				{
					for (int32 y = 0; y < Frame->Height; ++y)
					{
						FMemory::Memcpy(DstBytes + y * DstRowStride, SrcBytes + y * SrcRowStride, DstRowStride);
						const float* SrcRowFloats = reinterpret_cast<const float*>(SrcBytes + y * SrcRowStride);
						for (int32 x = 0; x < Frame->Width; ++x)
						{
							const float Value = SrcRowFloats[x];
							DepthMin = FMath::Min(DepthMin, Value);
							DepthMax = FMath::Max(DepthMax, Value);
							if (!bCapturedDepthSample)
							{
								FirstDepth = Value;
								bCapturedDepthSample = true;
							}
						}
					}
				}
				State.DepthReadback->Unlock();
				{
					const FString Message = FString::Printf(
						TEXT("Locked depth readback FrameId=%llu RowPitch=%d Sample=%f Min=%f Max=%f"),
						static_cast<uint64>(Frame->FrameId),
						RowPitchPixels,
						FirstDepth,
						DepthMin,
						DepthMax);
					UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] %s"), CaptureId, *Message);
				}
			}
			State.bDepthInFlight = false;

			// Hand off to the node's SPSC queue (render thread producer)
			if (TSharedPtr<FTSCaptureNode> NodeSP = State.NodeWeak.Pin())
			{
				FTSCaptureNode* Node = NodeSP.Get();
				while (Node->QueueCount.Load() >= Node->RingCapacity)
				{
					TSharedPtr<FTSCaptureFrame> Dummy;
					if (Node->FrameQueue.Dequeue(Dummy))
					{
						Node->QueueCount.DecrementExchange();
						UE_LOG(LogTongSimCapture, Verbose, TEXT("[%s] Dropped oldest frame to maintain ring capacity"), *Node->CaptureId.ToString());
					}
					else
					{
						break;
					}
				}
				Node->FrameQueue.Enqueue(Frame);
				Node->QueueCount.IncrementExchange();
				UE_LOG(LogTongSimCapture, VeryVerbose, TEXT("[%s] Produced frame FrameId=%llu QueueCount=%d"), *Node->CaptureId.ToString(), (unsigned long long)Frame->FrameId, Node->QueueCount.Load());
			}

			if (UTSCaptureSubsystem* StrongThis = WeakThis.Get())
			{
				const FName ProducedCaptureId = State.CaptureId;
				const TSharedPtr<FTSCaptureFrame> FrameCopy = Frame;
				AsyncTask(ENamedThreads::GameThread, [StrongThis, ProducedCaptureId, FrameCopy]()
				{
					StrongThis->OnFrameProduced().Broadcast(ProducedCaptureId, FrameCopy);
				});
			}

			// Dispatch async compression if configured
			const TWeakPtr<FTSCaptureNode> NodeWeak = State.NodeWeak;
			if (TSharedPtr<FTSCaptureNode> NodeSP = NodeWeak.Pin())
			{
				const bool bDoRgb = (NodeSP->RgbCodec == ETSRgbCodec::JPEG) && (Frame->Rgba8.Num() == Frame->Width * Frame->Height * 4);
				const bool bDoDepth = (NodeSP->DepthCodec == ETSDepthCodec::EXR) && (Frame->DepthR32.Num() == Frame->Width * Frame->Height);
				if (bDoRgb || bDoDepth)
				{
					TArray<uint8> RgbaCopy;
					if (bDoRgb) { RgbaCopy = Frame->Rgba8; }
					TArray<float> DepthCopy;
					if (bDoDepth) { DepthCopy = Frame->DepthR32; }
					const int32 W = Frame->Width;
					const int32 H = Frame->Height;
					const uint64 Fid = Frame->FrameId;
					const int32 Quality = NodeSP->JpegQuality;

					Async(EAsyncExecution::ThreadPool, [NodeWeak, RgbaCopy = MoveTemp(RgbaCopy), DepthCopy = MoveTemp(DepthCopy), W, H, Fid, bDoRgb, bDoDepth, Quality]() mutable
					{
						TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_CompressAsync);
						TSharedPtr<FTSCaptureCompressedFrame> C = MakeShared<FTSCaptureCompressedFrame>();
						C->FrameId = Fid;
						C->Width = W;
						C->Height = H;

						if (bDoRgb)
						{
							IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
							TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);
							if (Wrapper.IsValid())
							{
								Wrapper->SetRaw(RgbaCopy.GetData(), RgbaCopy.Num(), W, H, ERGBFormat::BGRA, 8);
								const TArray64<uint8>& Comp = Wrapper->GetCompressed(Quality);
								C->RgbJpeg.Append(Comp.GetData(), Comp.Num());
							}
						}

						if (bDoDepth)
						{
							TArray<float> RGBA;
							RGBA.SetNumUninitialized(W * H * 4);
							for (int32 i = 0; i < W * H; ++i)
							{
								RGBA[i * 4 + 0] = DepthCopy[i];
								RGBA[i * 4 + 1] = DepthCopy[i];
								RGBA[i * 4 + 2] = DepthCopy[i];
								RGBA[i * 4 + 3] = 1.f;
							}
							IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
							TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::EXR);
							if (Wrapper.IsValid())
							{
								Wrapper->SetRaw(RGBA.GetData(), RGBA.Num() * sizeof(float), W, H, ERGBFormat::RGBAF, 32);
								const TArray64<uint8>& Comp = Wrapper->GetCompressed(0);
								C->DepthExr.Append(Comp.GetData(), Comp.Num());
							}
						}

						if (TSharedPtr<FTSCaptureNode> NodeSP2 = NodeWeak.Pin())
						{
							while (NodeSP2->CompressedQueueCount.Load() >= NodeSP2->CompressedRingCapacity)
							{
								TSharedPtr<FTSCaptureCompressedFrame> Dummy;
								if (NodeSP2->CompressedQueue.Dequeue(Dummy))
								{
									NodeSP2->CompressedQueueCount.DecrementExchange();
								}
								else
								{
									break;
								}
							}
							NodeSP2->CompressedQueue.Enqueue(C);
							NodeSP2->CompressedQueueCount.IncrementExchange();
						}
					});
				}
			}

			State.InFlight = FCaptureRequest();
		}
	});
}

void UTSCaptureSubsystem::HandleWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources)
{
	TRACE_CPUPROFILER_EVENT_SCOPE(TSCapture_HandleWorldCleanup);
	UWorld* MyWorld = GetSubsystemWorld(GetGameInstance());
	if (World != nullptr && World == MyWorld)
	{
		StopAllCaptures();
	}
}

FTSCameraIntrinsics UTSCaptureSubsystem::MakeIntrinsics(int32 Width, int32 Height, float FovDegrees)
{
	FTSCameraIntrinsics I;
	const float FovRad = FMath::DegreesToRadians(FovDegrees);
	// Assume FOV is horizontal
	const float Fx = 0.5f * static_cast<float>(Width) / FMath::Tan(0.5f * FovRad);
	const float Fy = Fx; // square pixels by default
	I.Fx = Fx;
	I.Fy = Fy;
	I.Cx = 0.5f * static_cast<float>(Width);
	I.Cy = 0.5f * static_cast<float>(Height);
	return I;
}
