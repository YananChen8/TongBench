#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Engine/EngineTypes.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TSCaptureTypes.h"
#include "Templates/Atomic.h"
#include "Logging/LogMacros.h"
#include "PixelFormat.h"
#include "TSCaptureSubsystem.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogTongSimCapture, Log, All);

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class AActor;
class FTSCaptureViewExtension;
class FSceneViewStateInterface;
class FRDGBuilder;
class FSceneView;
struct FPostProcessMaterialInputs;

DECLARE_MULTICAST_DELEGATE_TwoParams(FTSCaptureFrameEvent, const FName& /*CaptureId*/, const TSharedPtr<FTSCaptureFrame>& /*Frame*/);

USTRUCT()
struct FTSCaptureNodeConfig
{
	GENERATED_BODY()

	UPROPERTY()
	int32 Width = 640;

	UPROPERTY()
	int32 Height = 480;

	UPROPERTY()
	float Fov = 90.0f; // horizontal degrees

	UPROPERTY()
	float Qps = 30.0f; // queries per second

	UPROPERTY()
	bool bEnableDepth = true;

	// Depth encoding mode (affects how depth is produced). Default LinearDepth.
	UPROPERTY(EditAnywhere)
	ETSCaptureDepthMode DepthMode = ETSCaptureDepthMode::LinearDepth;

	UPROPERTY()
	TEnumAsByte<ESceneCaptureSource> ColorSource = ESceneCaptureSource::SCS_FinalColorLDR;

	UPROPERTY()
	TEnumAsByte<ETextureRenderTargetFormat> ColorRenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;

	UPROPERTY()
	bool bEnablePostProcess = true;

	UPROPERTY()
	bool bEnableTemporalAA = true;

	UPROPERTY()
	float DepthNearPlane = 10.f;

	UPROPERTY()
	float DepthFarPlane = 5000.f;
};

// Runtime node for a single capture instance
struct FTSCaptureNode
{
    // Game thread state
    FName CaptureId;
    TWeakObjectPtr<AActor> OwnerActor;
    bool bOwnsActor = true;
    TWeakObjectPtr<USceneCaptureComponent2D> ColorCapture;
    TWeakObjectPtr<UTextureRenderTarget2D> ColorRT;
    FSceneViewStateInterface* ViewState = nullptr;

	FTSCaptureNodeConfig Config;

	// Scheduling
	double LastCaptureGameTime = -1.0;
	uint64 FrameCounter = 0;

	// Cached metadata for the pending GPU copy (set at enqueue time)
	struct FPendingMeta
	{
		bool bValid = false;
		uint64 FrameId = 0;
		double GameTimeSeconds = 0.0;
		int32 Width = 0;
		int32 Height = 0;
		FTransform Pose;
		FTSCameraIntrinsics Intrinsics;
		float DepthNear = 10.f;
		float DepthFar = 1000.f;
		ETSCaptureDepthMode DepthMode = ETSCaptureDepthMode::LinearDepth;
		bool bCaptureDepth = false;
		EPixelFormat ColorPixelFormat = PF_Unknown;
	} PendingMeta;

	// Lockless SPSC queue: render thread produces, game thread consumes
	TQueue<TSharedPtr<FTSCaptureFrame>, EQueueMode::Spsc> FrameQueue;

	// Ring capacity and count tracking
	int32 RingCapacity = 3;
	TAtomic<int32> QueueCount{0};
	// Compressed output queue (produced by worker threads, consumed on game thread)
	TQueue<TSharedPtr<FTSCaptureCompressedFrame>, EQueueMode::Mpsc> CompressedQueue;
	int32 CompressedRingCapacity = 2;
	TAtomic<int32> CompressedQueueCount{0};

	// Compression config
	ETSRgbCodec RgbCodec = ETSRgbCodec::None;
	ETSDepthCodec DepthCodec = ETSDepthCodec::None;
	int32 JpegQuality = 90;
};

UCLASS()
class TONGSIMCAPTURE_API UTSCaptureSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
	// UGameInstanceSubsystem overrides
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	// Start a capture by spawning an internal host actor with SceneCapture components
	// Order ensures defaulted params are trailing (Qps, bEnableDepth)
	UFUNCTION(BlueprintCallable, meta=(AutoCreateRefTerm="WorldTransform"), Category = "TongSim|Capture")
	bool StartCapture(const FName CaptureId, int32 Width, int32 Height, float FovDegrees, const FTransform& WorldTransform, float Qps = 30.0f, bool bEnableDepth = true);

    // Stop and remove a capture node
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    bool StopCapture(const FName CaptureId);

    // Start capture using an existing owner actor (no actor spawn); returns false on invalid actor
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    bool StartCaptureOnActor(const FName CaptureId, AActor* OwnerActor, int32 Width, int32 Height, float FovDegrees, float Qps = 30.0f, bool bEnableDepth = true);

    // Synchronously capture a single snapshot on an existing actor. Returns false if already capturing or on timeout
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    bool CaptureSnapshotOnActor(const FName CaptureId, AActor* OwnerActor, int32 Width, int32 Height, float FovDegrees, bool bEnableDepth, FTSCaptureFrame& OutFrame, float TimeoutSeconds = 0.5f);

    // Stop and remove all capture nodes
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    void StopAllCaptures();

	// Returns the latest available frame, non-blocking. Returns false if none available.
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool GetLatestFrame(const FName CaptureId, FTSCaptureFrame& OutFrame);

	// Returns the latest available compressed frame (if any), non-blocking.
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool GetLatestCompressedFrame(const FName CaptureId, FTSCaptureCompressedFrame& OutFrame);

	// Change depth mode (currently toggles depth on/off; future: extend to different encodings)
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetDepthEnabled(const FName CaptureId, bool bEnableDepth);

	// Set depth output mode (None / LinearDepth / ViewSpaceZ / Normalized01). Default LinearDepth.
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetDepthMode(const FName CaptureId, ETSCaptureDepthMode Mode);

	// Update capture resolution/FOV. Will reinit render targets if needed.
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool Reconfigure(const FName CaptureId, int32 Width, int32 Height, float FovDegrees, float Qps);

	// Configure asynchronous compression pipeline
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetCompression(const FName CaptureId, ETSRgbCodec RgbCodec, ETSDepthCodec DepthCodec, int32 JpegQuality = 90);

	// Internal: tick by ticker
	bool Tick(float DeltaSeconds);

	// Update capture transform for given capture id
	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetCaptureTransform(const FName CaptureId, const FTransform& WorldTransform);

	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetColorCaptureSettings(const FName CaptureId, ESceneCaptureSource CaptureSource, ETextureRenderTargetFormat RenderTargetFormat, bool bEnablePostProcess, bool bEnableTemporalAA);

	UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
	bool SetDepthRange(const FName CaptureId, float NearPlane, float FarPlane);

	// Map a view's owner actor back to its capture node (for ViewExtension use)
	bool FindNodeByOwnerActor(const AActor* Owner, TSharedPtr<FTSCaptureNode>& OutNode) const;

	FTSCaptureFrameEvent& OnFrameProduced() { return FrameProducedEvent; }

    // True if a capture node with this id exists
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    bool IsCapturing(const FName CaptureId) const { return Registry.Contains(CaptureId); }

    // Query lightweight status info (game thread only)
    UFUNCTION(BlueprintCallable, Category = "TongSim|Capture")
    bool GetStatus(const FName CaptureId, FTSCaptureStatus& OutStatus) const;

private:
    TMap<FName, TSharedPtr<FTSCaptureNode>> Registry;

    FTSTicker::FDelegateHandle TickerHandle;

    // Handle world/map cleanup to auto-stop all captures on level transitions
    FDelegateHandle WorldCleanupHandle;

	TSharedPtr<FTSCaptureViewExtension, ESPMode::ThreadSafe> ViewExtension;

	// Helpers
	void EnsureTargetsAndComponents_GameThread(const TSharedPtr<FTSCaptureNode>& Node);
	void EnqueueCaptureAndReadback_GameThread(const TSharedPtr<FTSCaptureNode>& Node);
	void PumpReadbacks_RenderThread();

	static FTSCameraIntrinsics MakeIntrinsics(int32 Width, int32 Height, float FovDegrees);

    void ProcessViewAfterTonemap_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs);

	friend class FTSCaptureViewExtension;

	// Delegate handler for world cleanup
	void HandleWorldCleanup(UWorld* World, bool bSessionEnded, bool bCleanupResources);

private:
	FTSCaptureFrameEvent FrameProducedEvent;
};
