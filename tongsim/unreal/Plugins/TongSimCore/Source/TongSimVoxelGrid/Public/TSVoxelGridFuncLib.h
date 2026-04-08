// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "PhysicsEngine/ConvexElem.h"

struct FZAxisSegment
	{
	double Z[2];
	int IntersectionPointNum;
	};

struct FMinAndMaxYLineInPlaneX
	{
	int MinYLineIndex;
	int MaxYLineIndex;
	};

struct FVoxelBox
	{
private:
	FTransform BoxTransform;
	uint16 VoxelHalfNumX = 16;
	uint16 VoxelHalfNumY = 16;
	uint16 VoxelHalfNumZ = 16;
	FVector BoxSize = FVector{100., 100., 100.};
	FVector VoxelSize;

public:
	FVoxelBox(){
		VoxelSize.X = BoxSize.X / (2*VoxelHalfNumX);
		VoxelSize.Y = BoxSize.Y / (2*VoxelHalfNumY);
		VoxelSize.Z = BoxSize.Z / (2*VoxelHalfNumZ);
	};

	FVoxelBox(const FTransform& InTransform, uint32 InVoxelHalfNumX, uint32 InVoxelHalfNumY, uint32 InVoxelHalfNumZ,
	          const FVector& InBoxSize){
		bool bFalse = InVoxelHalfNumX == 0 || InVoxelHalfNumY == 0 || InVoxelHalfNumZ == 0
			|| InBoxSize.X <= 0 || InBoxSize.Y <= 0 || InBoxSize.Z <= 0;
		if (bFalse)
		{
			*this = FVoxelBox();
		}
		else
		{
			VoxelHalfNumX = InVoxelHalfNumX;
			VoxelHalfNumY = InVoxelHalfNumY;
			VoxelHalfNumZ = InVoxelHalfNumZ;
			BoxSize = InBoxSize;
			VoxelSize.X = BoxSize.X / (2*InVoxelHalfNumX);
			VoxelSize.Y = BoxSize.Y / (2*InVoxelHalfNumY);
			VoxelSize.Z = BoxSize.Z / (2*InVoxelHalfNumZ);
			BoxTransform = InTransform;
			BoxTransform.RemoveScaling();
		}
	}

	FORCEINLINE bool IsValid() const{
		bool bFalse = VoxelHalfNumX == 0 || VoxelHalfNumY == 0 || VoxelHalfNumZ == 0
			|| BoxSize.X <= 0 || BoxSize.Y <= 0 || BoxSize.Z <= 0;
		return !bFalse;
	};

	FORCEINLINE FVector GetBoxSize() const{
		return BoxSize;
	};
	FORCEINLINE FVector GetGridSize() const{
		return VoxelSize;
	};

	FORCEINLINE uint16 GetGridHalfNumX() const{
		return VoxelHalfNumX;
	};

	FORCEINLINE uint16 GetGridHalfNumY() const{
		return VoxelHalfNumY;
	};

	FORCEINLINE uint16 GetGridHalfNumZ() const{
		return VoxelHalfNumZ;
	};

	FORCEINLINE FTransform GetBoxTransform() const{
		return BoxTransform;
	};
	};

struct FVoxelGridQueryParam
	{
	FVoxelGridQueryParam() = delete;

	FVoxelGridQueryParam(UWorld* InWorld): World(InWorld){
	};

	FVoxelBox GridBox;

	TArray<AActor*> Actors;
	TSet<TObjectPtr<UPrimitiveComponent>> IgnoredPrimitiveComponents;

	TSet<TObjectPtr<USkeletalMeshComponent>> IgnoredSkeletalMeshComponents;

	FORCEINLINE bool IsValid() const{
		return GridBox.IsValid() && World != nullptr;
	}

public:
	UWorld* GetWorld() const{ return World; };

private:
	UWorld* World = nullptr;
	};

struct TONGSIMVOXELGRID_API TSVoxelGridFuncLib
	{
	TSVoxelGridFuncLib() = delete;
	TSVoxelGridFuncLib(const TSVoxelGridFuncLib&) = delete;
	TSVoxelGridFuncLib& operator=(const TSVoxelGridFuncLib&) = delete;

	static auto QueryVoxelGrids(const FVoxelGridQueryParam& QueryParams, TArray<uint8>& VoxelGrids, UWorld* InWorld=nullptr) -> void;

private:
	static void UpdateBodySetupAABBMap(const TArray<AActor*>& Actors, const FVoxelGridQueryParam& QueryParam,
	                                   TArray<TObjectPtr<UPrimitiveComponent>>& WorldPrimitiveComponents,
	                                   TArray<TObjectPtr<USkeletalMeshComponent>>& WorldSkeletalMeshComponents,
	                                   TMap<TObjectPtr<UBodySetup>, FBox>& BodySetupAABBsMap,
	                                   TMap<FName, FBox>& SkeletalMeshComponentAABBsMap);

	static TArray<FVector> GetBoxEdges(const FBox& Box);
	static bool AABBOverlap(const FBox& A, const FBox& B, const FTransform& BTransform);

	FORCEINLINE static bool IsValidCollision(UPrimitiveComponent* Component){
		return IsValid(Component) &&
		(Component->GetCollisionEnabled() == ECollisionEnabled::Type::PhysicsOnly ||
			Component->GetCollisionEnabled() == ECollisionEnabled::Type::QueryAndPhysics) &&
				(Component->GetBodySetup());
	}


	FORCEINLINE static bool IsValidCollision(const USkeletalMeshComponent* Component){
		bool CollisionTypeValid = IsValid(Component) &&
			(Component->GetCollisionEnabled() == ECollisionEnabled::Type::PhysicsOnly ||
				Component->GetCollisionEnabled() == ECollisionEnabled::Type::QueryAndPhysics);
		if (!CollisionTypeValid)
		{
			return false;
		}
		const USkeletalMesh* SkeletalMesh = Component->GetSkeletalMeshAsset();
		return IsValid(SkeletalMesh) && (SkeletalMesh->GetBodySetup());
	}

	// 计算FKAggregateGeom占用空间网格体素的情况，方法内部会调用FixGridVoxelWithCapsule,Sphere,Box,Convex
	static void FixVoxelGridsWithAggGeom(const FVoxelBox& GridBox, const FKAggregateGeom& AggGeom,
	                                     const FTransform& AggGeomTransformInVoxelBoxSpace,
	                                     TArray<uint8>& VoxelGridsArray,
	                                     UWorld* InWorld=nullptr);

	// 计算胶囊碰撞体占用空间网格体素的情况，方法内部会调用FixGridVoxelWithSegment
	static void FixVoxelGridsWithCapsule(const FVoxelBox& GridBox, const FVector& Center1, const FVector& Center2,
	                                     float Radius, TArray<uint8>& VoxelGridsArray);

	// 计算Box碰撞体占用空间网格体素的情况，方法内部会调用FixGridVoxelWithSegment
	static void FixVoxelGridsWithBox(const FVoxelBox& GridBox, const FVector& BoxExtent,
	                                 const FTransform& TransformBS2VS, TArray<uint8>& VoxelGridsArray, UWorld* InWorld=nullptr, bool bIsDrawDebug=false);

	// 计算Sphere碰撞体占用空间网格体素的情况，方法内部会调用FixGridVoxelWithSegment
	static void FixVoxelGridsWithSphere(const FVoxelBox& GridBox, const FVector& Center, float Radius,
	                                    TArray<uint8>& VoxelGridsArray);

	// 计算凸碰撞体占用空间网格体素的情况，方法内部会调用FixGridVoxelWithSegment
	static void FixVoxelGridsWithConvexMesh(const FVoxelBox& GridBox, const FKConvexElem& Convex,
	                                        const FTransform& ConvexTransformInVoxelBoxSpace,
	                                        TArray<uint8>& VoxelGridsArray);

	// 给定一条线段(XPlaneIndex平面上，LineYIndex直线上，最大Z为MaxZ, 最小Z维MinZ)，计算在VoxelBox的局部坐标系下，线段占用的体素网格
	static bool FixVoxelGridsWithSegment(const FVoxelBox& GridBox, int PlaneXIndex, int LineYIndex, float ZMin,
	                                     float ZMax, TArray<uint8>& VoxelGridsArray);

	static void ClearAllVoxels(TArray<uint8>& GridVoxelArray){
		if (!GridVoxelArray.IsEmpty())
		{
			GridVoxelArray.SetNumZeroed(GridVoxelArray.Num());
		}
	}

	static FORCEINLINE int32 GetXIndexFromX(const FVoxelBox& GridBox, double X){
		int XIndex;

		uint16 GridXNum = GridBox.GetGridHalfNumX() * 2;
		double GridSizeX = GridBox.GetGridSize().X;

		if (X - GridSizeX * floor(X / GridSizeX) > 0.5)
		{
			XIndex = floor(X / GridSizeX) + GridXNum / 2;
		}
		else
		{
			XIndex = floor(X / GridSizeX) + GridXNum / 2 - 1;
		}
		XIndex = FMath::Clamp(XIndex, 0, GridXNum - 1);
		return XIndex;
	};

	static FORCEINLINE int32 GetYIndexFromY(const FVoxelBox& GridBox, double Y){
		int YIndex;
		uint16 GridYNum = GridBox.GetGridHalfNumY() * 2;
		double GridSizeY = GridBox.GetGridSize().Y;
		if (Y - GridSizeY * floor(Y / GridSizeY) > 0.5)
		{
			YIndex = floor(Y / GridSizeY) + GridYNum / 2;
		}
		else
		{
			YIndex = floor(Y / GridSizeY) + GridYNum / 2 - 1;
		}
		YIndex = FMath::Clamp(YIndex, 0, GridYNum - 1);
		return YIndex;
	};

	static FORCEINLINE int32 GetZIndexFromZ(const FVoxelBox& GridBox, double Z){
		int ZIndex;
		uint16 GridZNum = GridBox.GetGridHalfNumZ() * 2;
		double GridSizeZ = GridBox.GetGridSize().Z;
		if (Z - GridSizeZ * floor(Z / GridSizeZ) > 0.5)
		{
			ZIndex = floor(Z / GridSizeZ) + GridZNum / 2;
		}
		else
		{
			ZIndex = floor(Z / GridSizeZ) + GridZNum / 2 - 1;
		}
		ZIndex = FMath::Clamp(ZIndex, 0, GridZNum - 1);
		return ZIndex;
	};

	static FORCEINLINE double GetXFromXIndex(const FVoxelBox& GridBox, int XIndex){
		uint16 GridXNum = GridBox.GetGridHalfNumX() * 2;
		double GridSizeX = GridBox.GetGridSize().X;
		XIndex = FMath::Clamp(XIndex, 0, GridXNum - 1);
		return (XIndex - GridXNum / 2 + 0.5) * GridSizeX;
	};

	static FORCEINLINE double GetYFromYIndex(const FVoxelBox& GridBox, int YIndex){
		uint16 GridYNum = GridBox.GetGridHalfNumY() * 2;
		double GridSizeY = GridBox.GetGridSize().Y;
		YIndex = FMath::Clamp(YIndex, 0, GridYNum - 1);
		return (YIndex - GridYNum / 2 + 0.5) * GridSizeY;
	}

	static FORCEINLINE double GetZFromZIndex(const FVoxelBox& GridBox, int ZIndex){
		uint16 GridZNum = GridBox.GetGridHalfNumZ() * 2;
		double GridSizeZ = GridBox.GetGridSize().Z;
		ZIndex = FMath::Clamp(ZIndex, 0, GridZNum - 1);
		return (ZIndex - GridZNum / 2 + 0.5) * GridSizeZ;
	}

	// ======= 任意相交即置 1：区域到索引的统一“外扩包含”版本 =======
	// X 区间 -> X 索引区间
	static FORCEINLINE void GetXIndexRegionFromXRegion(const FVoxelBox& GridBox, double MinX, double MaxX,
													   int& MinXIndex, int& MaxXIndex)
	{
		const uint16 GridXNum = GridBox.GetGridHalfNumX() * 2;
		const double GridSizeX = GridBox.GetGridSize().X;

		// 任何相交都包含对应体素：两端都 floor
		MinXIndex = FMath::FloorToInt(MinX / GridSizeX) + GridXNum / 2;
		MaxXIndex = FMath::FloorToInt(MaxX / GridSizeX) + GridXNum / 2;

		MinXIndex = FMath::Clamp(MinXIndex, 0, GridXNum - 1);
		MaxXIndex = FMath::Clamp(MaxXIndex, 0, GridXNum - 1);
	}

	// Y 区间 -> Y 索引区间
	static FORCEINLINE void GetYIndexRegionFromYRegion(const FVoxelBox& GridBox, double MinY, double MaxY,
													   int& MinYIndex, int& MaxYIndex)
	{
		const uint16 GridYNum = GridBox.GetGridHalfNumY() * 2;
		const double GridSizeY = GridBox.GetGridSize().Y;

		MinYIndex = FMath::FloorToInt(MinY / GridSizeY) + GridYNum / 2;
		MaxYIndex = FMath::FloorToInt(MaxY / GridSizeY) + GridYNum / 2;

		MinYIndex = FMath::Clamp(MinYIndex, 0, GridYNum - 1);
		MaxYIndex = FMath::Clamp(MaxYIndex, 0, GridYNum - 1);
	}

	// Z 区间 -> Z 索引区间
	static FORCEINLINE void GetZIndexRegionFromZRegion(const FVoxelBox& GridBox, double MinZ, double MaxZ,
													   int& MinZIndex, int& MaxZIndex)
	{
		const uint16 GridZNum = GridBox.GetGridHalfNumZ() * 2;
		const double GridSizeZ = GridBox.GetGridSize().Z;

		MinZIndex = FMath::FloorToInt(MinZ / GridSizeZ) + GridZNum / 2;
		MaxZIndex = FMath::FloorToInt(MaxZ / GridSizeZ) + GridZNum / 2;

		MinZIndex = FMath::Clamp(MinZIndex, 0, GridZNum - 1);
		MaxZIndex = FMath::Clamp(MaxZIndex, 0, GridZNum - 1);
	}

	//
	// static FORCEINLINE void GetXIndexRegionFromXRegion(const FVoxelBox& GridBox, double MinX, double MaxX,
	//                                                    int& MinXIndex,
	//                                                    int& MaxXIndex){
	// 	uint16 GridXNum = GridBox.GetGridHalfNumX() * 2;
	// 	double GridSizeX = GridBox.GetGridSize().X;
	// 	if (MinX - floor(MinX / GridSizeX) * GridSizeX < 0.5 * GridSizeX)
	// 	{
	// 		MinXIndex = floor(MinX / GridSizeX) + GridXNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MinXIndex = floor(MinX / GridSizeX) + GridXNum / 2 + 1;
	// 	}
	// 	if (MaxX - floor(MaxX / GridSizeX) * GridSizeX > 0.5 * GridSizeX)
	// 	{
	// 		MaxXIndex = floor(MaxX / GridSizeX) + GridXNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MaxXIndex = floor(MaxX / GridSizeX) + GridXNum / 2 - 1;
	// 	}
	// 	MinXIndex = FMath::Clamp(MinXIndex, 0, GridXNum - 1);
	// 	MaxXIndex = FMath::Clamp(MaxXIndex, 0, GridXNum - 1);
	// }
	//
	// static FORCEINLINE void GetYIndexRegionFromYRegion(const FVoxelBox& GridBox, double MinY, double MaxY,
	//                                                    int& MinYIndex,
	//                                                    int& MaxYIndex){
	// 	uint16 GridYNum = GridBox.GetGridHalfNumY() * 2;
	// 	double GridSizeY = GridBox.GetGridSize().Y;
	// 	if (MinY - floor(MinY / GridSizeY) * GridSizeY < 0.5 * GridSizeY)
	// 	{
	// 		MinYIndex = floor(MinY / GridSizeY) + GridYNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MinYIndex = floor(MinY / GridSizeY) + GridYNum / 2 + 1;
	// 	}
	// 	if (MaxY - floor(MaxY / GridSizeY) * GridSizeY > 0.5 * GridSizeY)
	// 	{
	// 		MaxYIndex = floor(MaxY / GridSizeY) + GridYNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MaxYIndex = floor(MaxY / GridSizeY) + GridYNum / 2 - 1;
	// 	}
	// 	MinYIndex = FMath::Clamp(MinYIndex, 0, GridYNum - 1);
	// 	MaxYIndex = FMath::Clamp(MaxYIndex, 0, GridYNum - 1);
	// }
	//
	// static FORCEINLINE void GetZIndexRegionFromZRegion(const FVoxelBox& GridBox, double MinZ, double MaxZ,
	//                                                    int& MinZIndex,
	//                                                    int& MaxZIndex){
	// 	uint16 GridZNum = GridBox.GetGridHalfNumZ() * 2;
	// 	double GridSizeZ = GridBox.GetGridSize().Z;
	// 	if (MinZ - floor(MinZ / GridSizeZ) * GridSizeZ < 0.5 * GridSizeZ)
	// 	{
	// 		MinZIndex = floor(MinZ / GridSizeZ) + GridZNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MinZIndex = floor(MinZ / GridSizeZ) + GridZNum / 2 + 1;
	// 	}
	// 	if (MaxZ - floor(MaxZ / GridSizeZ) * GridSizeZ > 0.5 * GridSizeZ)
	// 	{
	// 		MaxZIndex = floor(MaxZ / GridSizeZ) + GridZNum / 2;
	// 	}
	// 	else
	// 	{
	// 		MaxZIndex = floor(MaxZ / GridSizeZ) + GridZNum / 2 - 1;
	// 	}
	// 	MinZIndex = FMath::Clamp(MinZIndex, 0, GridZNum - 1);
	// 	MaxZIndex = FMath::Clamp(MaxZIndex, 0, GridZNum - 1);
	// }

	static void ResetZAxisSegmentArray(const FVoxelBox& GridBox, TArray<FZAxisSegment>& ZAxisSegmentArray){
		ZAxisSegmentArray.Empty();
		// if (ZAxisSegmentArray.IsEmpty())
		// {
		// 	return;
		// }
		uint32 GridNumX = GridBox.GetGridHalfNumX() * 2;
		uint32 GridNumY = GridBox.GetGridHalfNumY() * 2;
		ZAxisSegmentArray.SetNumZeroed(GridNumX * GridNumY);
		// FMemory::Memset(ZAxisSegmentArray.GetData(), 0, sizeof(FZAxisSegment) * GridNumX * GridNumY);
	}

	static void ResetMinAndMaxYLineInPlaneXArray(const FVoxelBox& GridBox,
	                                             TArray<FMinAndMaxYLineInPlaneX>& MinAndMaxYLineInPlaneXArray){
		MinAndMaxYLineInPlaneXArray.Empty();
		MinAndMaxYLineInPlaneXArray.SetNumUninitialized(GridBox.GetGridHalfNumX() * 2);
		// if (MinAndMaxYLineInPlaneXArray.IsEmpty())
		// {
		// 	return;
		// }
		for (auto& Elem : MinAndMaxYLineInPlaneXArray)
		{
			Elem.MinYLineIndex = GridBox.GetGridSize().Y;
			Elem.MaxYLineIndex = -GridBox.GetGridSize().Y;
		}
		// FMemory::Memset(MinAndMaxYLineInPlaneXArray.GetData(), 0,
		//                 sizeof(FMinAndMaxYLineInPlaneX) * GridBox.GetGridHalfNumX() * 2);
	};

public:
	static void DrawDebugGrids(const UWorld* World, const FVoxelBox& VoxelBox, float TimeLength, FColor Color, const TArray<uint8>& Voxels);

	};
