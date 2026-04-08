// Copyright Epic Games, Inc. All Rights Reserved.

#include "TSVoxelGridFuncLib.h"

#include "GeomTools.h"
#include "Kismet/GameplayStatics.h"
#include "PhysicsEngine/BodySetup.h"
#include "PhysicsEngine/PhysicsAsset.h"
#include "PhysicsEngine/SkeletalBodySetup.h"

auto TSVoxelGridFuncLib::QueryVoxelGrids(const FVoxelGridQueryParam& QueryParams, TArray<uint8>& VoxelGrids,
                                         UWorld* InWorld) -> void{
	const auto IgnoredPrimitiveComponents = QueryParams.IgnoredPrimitiveComponents;
	const auto IgnoredSkeletalMeshComponents = QueryParams.IgnoredSkeletalMeshComponents;

	uint16 Aligned_8_GridZNum;

	uint32 GridNumX = QueryParams.GridBox.GetGridHalfNumX() * 2;
	uint32 GridNumY = QueryParams.GridBox.GetGridHalfNumY() * 2;
	uint32 GridNumZ = QueryParams.GridBox.GetGridHalfNumZ() * 2;
	if (GridNumZ % 8 == 0)
	{
		Aligned_8_GridZNum = GridNumZ;
	}
	else
	{
		Aligned_8_GridZNum = (GridNumZ / 8 + 1) * 8;
	}
	auto& VoxelGridsArray = VoxelGrids;
	VoxelGridsArray.SetNumZeroed(Aligned_8_GridZNum * GridNumX * GridNumY / 8);

	TMap<FName, FName> PrimitiveBodySetupMap;
	TArray<TObjectPtr<UPrimitiveComponent>> WorldPrimitiveComponents;
	TArray<TObjectPtr<USkeletalMeshComponent>> WorldSkeletalMeshComponents;
	TMap<TObjectPtr<UBodySetup>, FBox> BodySetupAABBsMap;
	TMap<FName, FBox> SkeletalMeshComponentAABBsMap;
	TMap<FName, FName> SkeletalMeshBodySetupMap;
	decltype(GFrameCounter) CurrentFrame{0};

	FTransform VoxelBoxTransform = QueryParams.GridBox.GetBoxTransform();
	VoxelBoxTransform.RemoveScaling();
	FTransform VoxelBoxInverseTransform = VoxelBoxTransform.Inverse();
	// FTransform VoxelBoxInverseTransform {VoxelBoxTransform.GetRotation().Inverse(), -VoxelBoxTransform.GetLocation()};

	UpdateBodySetupAABBMap(QueryParams.Actors, QueryParams, WorldPrimitiveComponents, WorldSkeletalMeshComponents,
	                       BodySetupAABBsMap,
	                       SkeletalMeshComponentAABBsMap);

	TArray<TObjectPtr<UPrimitiveComponent>> AABBOverlappedPrimitiveComponents;
	AABBOverlappedPrimitiveComponents.Reserve(WorldPrimitiveComponents.Num());
	for (auto Component : WorldPrimitiveComponents)
	{
		if (IgnoredPrimitiveComponents.Contains(Component))
		{
			continue;
		}
		if (!IsValidCollision(Component))
		{
			continue;
		}
		auto BodySetupName = Component->GetBodySetup()->GetFName();
		if (BodySetupAABBsMap.Contains(Component->GetBodySetup()))
		{
			auto ComponentBox = BodySetupAABBsMap.FindRef(Component->GetBodySetup());
			auto ComponentTransform = Component->GetComponentTransform();
			auto BoxTransformInVoxelCoordinate = ComponentTransform * VoxelBoxInverseTransform;


			FBox VoxelBox{-QueryParams.GridBox.GetBoxSize() / 2, QueryParams.GridBox.GetBoxSize() / 2};
			if (AABBOverlap(VoxelBox, ComponentBox,
			                BoxTransformInVoxelCoordinate))
			{
				AABBOverlappedPrimitiveComponents.Add(Component);
			}
		}
	}

	// 获取与体素Box有重叠的SkeletalMeshComponent PhysicsAsset Body Setup AABB
	TArray<TObjectPtr<USkeletalMeshComponent>> AABBOverlappedSkeletalMeshComponents;
	AABBOverlappedSkeletalMeshComponents.Reserve(WorldSkeletalMeshComponents.Num());
	for (auto Component : WorldSkeletalMeshComponents)
	{
		if (IgnoredSkeletalMeshComponents.Contains(Component))
		{
			continue;
		}
		if (!IsValidCollision(Component))
		{
			continue;
		}
		FBox SkeletalMeshBox = Component->GetPhysicsAsset()->CalcAABB(Component, Component->GetComponentTransform());
		if (AABBOverlap(FBox{-QueryParams.GridBox.GetBoxSize() / 2, QueryParams.GridBox.GetBoxSize() / 2},
		                SkeletalMeshBox,
		                VoxelBoxInverseTransform))
		{
			AABBOverlappedSkeletalMeshComponents.Add(Component);
		}
	}


	// PrimitiveMesh的BodySetup逐个更新VoxelGrid
	UE_LOG(LogTemp, Log, TEXT("This Voxel Grids has %d Overlapped Primitive Components."),
	       AABBOverlappedPrimitiveComponents.Num());
	for (auto Component : AABBOverlappedPrimitiveComponents)
	{
		// Fix Grids With BodySetup AggGeom BoxElements
		FTransform ComponentTransformInVoxelBoxSpace = Component->GetComponentTransform() * VoxelBoxInverseTransform;
		if (IsValid(Component->GetBodySetup()))
		{
			FixVoxelGridsWithAggGeom(QueryParams.GridBox, Component->GetBodySetup()->AggGeom,
			                         ComponentTransformInVoxelBoxSpace, VoxelGridsArray, InWorld);
		}
	}

	// SkeletalMesh的SkeletalBodySetup逐个更新VoxelGrid
	for (auto Component : AABBOverlappedSkeletalMeshComponents)
	{
		for (auto BodySetup : Component->GetPhysicsAsset()->SkeletalBodySetups)
		{
			auto BoneIndex = Component->GetBoneIndex(BodySetup->BoneName);
			FTransform BoneTransform = Component->GetBoneTransform(BoneIndex);
			FTransform BoneTransformInVoxelBoxSpace = BoneTransform * VoxelBoxInverseTransform;
			FixVoxelGridsWithAggGeom(QueryParams.GridBox, BodySetup->AggGeom, BoneTransformInVoxelBoxSpace,
			                         VoxelGridsArray);
		}
	}
}

void TSVoxelGridFuncLib::UpdateBodySetupAABBMap(const TArray<AActor*>& Actors, const FVoxelGridQueryParam& QueryParam,
                                                TArray<TObjectPtr<UPrimitiveComponent>>& WorldPrimitiveComponents,
                                                TArray<TObjectPtr<USkeletalMeshComponent>>&
                                                WorldSkeletalMeshComponents,
                                                TMap<TObjectPtr<UBodySetup>, FBox>& BodySetupAABBsMap,
                                                TMap<FName, FBox>& SkeletalMeshComponentAABBsMap){
	WorldPrimitiveComponents.Empty();

	TArray<TObjectPtr<UPrimitiveComponent>> ActorPrimitiveComponents;
	for (auto Actor : Actors)
	{
		if (IsValid(Actor))
		{
			Actor->GetComponents(ActorPrimitiveComponents);
			WorldPrimitiveComponents.Append(ActorPrimitiveComponents);
		}
	}

	for (auto PrimitiveComponent : WorldPrimitiveComponents)
	{
		if (!IsValidCollision(PrimitiveComponent))
		{
			continue;
		}
		UBodySetup* BodySetup = PrimitiveComponent->GetBodySetup();
		if (!IsValid(BodySetup))
		{
			continue;
		}
		// auto BodySetupName = BodySetup->GetFName();
		if (!BodySetupAABBsMap.Contains(BodySetup))
		{
			BodySetupAABBsMap.Add(BodySetup, BodySetup->AggGeom.CalcAABB(FTransform::Identity));
		}
	}

	WorldSkeletalMeshComponents.Empty();
	SkeletalMeshComponentAABBsMap.Empty();
	TArray<TObjectPtr<USkeletalMeshComponent>> ActorSkeletalMeshComponents;
	for (auto Actor : Actors)
	{
		Actor->GetComponents(ActorSkeletalMeshComponents);
		WorldSkeletalMeshComponents.Append(ActorSkeletalMeshComponents);
	}

	for (auto SkeletalMeshComponent : WorldSkeletalMeshComponents)
	{
		if (!IsValidCollision(SkeletalMeshComponent))
		{
			continue;
		}
		auto ComponentName = SkeletalMeshComponent->GetFName();
		auto AABB = SkeletalMeshComponent->GetSkeletalMeshAsset()->GetPhysicsAsset()->CalcAABB(
			SkeletalMeshComponent, FTransform::Identity);
		SkeletalMeshComponentAABBsMap.Add(ComponentName, AABB);
	}
}

TArray<FVector> TSVoxelGridFuncLib::GetBoxEdges(const FBox& Box){
	TArray<FVector> Edges;
	Edges.Reserve(24);

	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Max.Z});

	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Min.Z});

	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Min.Z});


	Edges.Add(FVector{Box.Max.X, Box.Min.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Max.Z});

	Edges.Add(FVector{Box.Max.X, Box.Min.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Max.Z});

	Edges.Add(FVector{Box.Max.X, Box.Min.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Max.X, Box.Min.Y, Box.Min.Z});


	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Max.X, Box.Min.Y, Box.Min.Z});

	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Min.Z});

	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Min.Z});
	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Max.Z});


	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Min.Z});

	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Max.X, Box.Max.Y, Box.Max.Z});

	Edges.Add(FVector{Box.Min.X, Box.Max.Y, Box.Max.Z});
	Edges.Add(FVector{Box.Min.X, Box.Min.Y, Box.Max.Z});

	return Edges;
}

bool TSVoxelGridFuncLib::AABBOverlap(const FBox& A, const FBox& B, const FTransform& BTransform){

	auto BBoxEdges=GetBoxEdges(B);

	for (auto& Point:BBoxEdges)
	{
		Point=BTransform.TransformPosition(Point);
	}

	for (int i=0; i<BBoxEdges.Num()/2;i++)
	{
		const FVector& Point1=BBoxEdges[i*2];
		const FVector& Point2=BBoxEdges[i*2+1];
		if (FMath::LineBoxIntersection(A, Point1, Point2, Point2-Point1))
		{
			return true;
		}
	}

	auto ABoxEdges=GetBoxEdges(A);

	for (auto& Point:ABoxEdges)
	{
		Point=BTransform.InverseTransformPosition(Point);
	}

	for (int i=0; i<ABoxEdges.Num()/2;i++)
	{
		const FVector& Point1=ABoxEdges[i*2];
		const FVector& Point2=ABoxEdges[i*2+1];
		if (FMath::LineBoxIntersection(B, Point1, Point2, Point2-Point1))
		{
			return true;
		}
	}
	return false;
	// FVector APoints[8], BPoints[8];
	// double DeltaScale = 1.001;
	// A.GetVertices(APoints);
	// B.GetVertices(BPoints);
	//
	// FTransform BInverseTransform = BTransform.Inverse();
	// FVector TransformedAVertices[8];
	//
	// for (int i = 0; i < 8; i++)
	// {
	// 	TransformedAVertices[i] = BInverseTransform.TransformPosition(APoints[i]);
	// }
	// FBox TransformedABox;
	// TransformedABox.Min = TransformedAVertices[0];
	// TransformedABox.Max = TransformedAVertices[0];
	// for (int i = 1; i < 8; i++)
	// {
	// 	TransformedABox.Min.X = FMath::Min(TransformedABox.Min.X, TransformedAVertices[i].X);
	// 	TransformedABox.Min.Y = FMath::Min(TransformedABox.Min.Y, TransformedAVertices[i].Y);
	// 	TransformedABox.Min.Z = FMath::Min(TransformedABox.Min.Z, TransformedAVertices[i].Z);
	// 	TransformedABox.Max.X = FMath::Max(TransformedABox.Max.X, TransformedAVertices[i].X);
	// 	TransformedABox.Max.Y = FMath::Max(TransformedABox.Max.Y, TransformedAVertices[i].Y);
	// 	TransformedABox.Max.Z = FMath::Max(TransformedABox.Max.Z, TransformedAVertices[i].Z);
	// }
	// TransformedABox.Min *= DeltaScale;
	// TransformedABox.Max *= DeltaScale;
	// for (int i = 0; i < 8; i++)
	// {
	// 	if (FMath::PointBoxIntersection(BPoints[i], TransformedABox))
	// 	{
	// 		return true;
	// 	}
	// }
	//
	// FVector TransformedBVertices[8];
	// for (int i = 0; i < 8; i++)
	// {
	// 	TransformedBVertices[i] = BTransform.TransformPosition(BPoints[i]);
	// }
	// FBox TransformedBBox;
	// TransformedBBox.Min = TransformedBVertices[0];
	// TransformedBBox.Max = TransformedBVertices[0];
	// for (int i = 1; i < 8; i++)
	// {
	// 	TransformedBBox.Min.X = FMath::Min(TransformedBBox.Min.X, TransformedBVertices[i].X);
	// 	TransformedBBox.Min.Y = FMath::Min(TransformedBBox.Min.Y, TransformedBVertices[i].Y);
	// 	TransformedBBox.Min.Z = FMath::Min(TransformedBBox.Min.Z, TransformedBVertices[i].Z);
	// 	TransformedBBox.Max.X = FMath::Max(TransformedBBox.Max.X, TransformedBVertices[i].X);
	// 	TransformedBBox.Max.Y = FMath::Max(TransformedBBox.Max.Y, TransformedBVertices[i].Y);
	// 	TransformedBBox.Max.Z = FMath::Max(TransformedBBox.Max.Z, TransformedBVertices[i].Z);
	// }
	// TransformedBBox.Min *= DeltaScale;
	// TransformedBBox.Max *= DeltaScale;
	// for (int i = 0; i < 8; i++)
	// {
	// 	if (FMath::PointBoxIntersection(APoints[i], TransformedBBox))
	// 		FMath::LineExtentBoxIntersection()
	// 	{
	// 		return true;
	// 	}
	// }
	// return false;
}

void TSVoxelGridFuncLib::FixVoxelGridsWithAggGeom(const FVoxelBox& GridBox, const FKAggregateGeom& AggGeom,
                                                  const FTransform& AggGeomTransformInVoxelBoxSpace,
                                                  TArray<uint8>& VoxelGridsArray,
                                                  UWorld* InWorld){
	// Fix Grids With BodySetup AggGeom BoxElements
	for (auto BoxElem : AggGeom.BoxElems)
	{
		FTransform ElemTransform = BoxElem.GetTransform() * AggGeomTransformInVoxelBoxSpace;
		FVector BoxExtent{BoxElem.X / 2, BoxElem.Y / 2, BoxElem.Z / 2};
		FixVoxelGridsWithBox(GridBox, BoxExtent, ElemTransform, VoxelGridsArray, InWorld);
	}

	// 当前的球形碰撞体，只支持等轴缩放
	// Fix Grids With BodySetup AggGeom SphereElements
	float TransformAbsScale = AggGeomTransformInVoxelBoxSpace.GetScale3D().X;
	for (auto SphereElem : AggGeom.SphereElems)
	{
		float TransformedRadius = SphereElem.Radius * TransformAbsScale;
		FVector TransformedCenter = AggGeomTransformInVoxelBoxSpace.TransformPosition(SphereElem.Center);
		FixVoxelGridsWithSphere(GridBox, TransformedCenter, TransformedRadius, VoxelGridsArray);
	}

	// Fix Grids With BodySetup AggGeom CapsuleElements
	for (auto Capsule : AggGeom.SphylElems)
	{
		// CS means in Capsule Space
		FTransform CapsuleTransform = Capsule.GetTransform() * AggGeomTransformInVoxelBoxSpace;
		FVector Center1CS = Capsule.Center + FVector(0, 0, Capsule.Length / 2);
		FVector Center2CS = Capsule.Center + FVector(0, 0, -Capsule.Length / 2);

		FVector TransformedCenter1 = CapsuleTransform.TransformPosition(Center1CS);
		FVector TransformedCenter2 = CapsuleTransform.TransformPosition(Center2CS);


		float TransformedRadius = TransformAbsScale * Capsule.Radius;
		FixVoxelGridsWithCapsule(GridBox, TransformedCenter1, TransformedCenter2, TransformedRadius, VoxelGridsArray);
	}

	// Fix Grids With BodySetup AggGeom ConvexElements
	for (auto Convex : AggGeom.ConvexElems)
	{
		FixVoxelGridsWithConvexMesh(GridBox, Convex, AggGeomTransformInVoxelBoxSpace, VoxelGridsArray);
	}
}

void TSVoxelGridFuncLib::FixVoxelGridsWithCapsule(const FVoxelBox& GridBox, const FVector& Center1,
                                                  const FVector& Center2, float Radius, TArray<uint8>& VoxelGridsArray){
	double r = FMath::Abs(Radius);
	FVector MinZCenter, MaxZCenter;
	if (Center1.Z < Center2.Z)
	{
		MinZCenter = Center1;
		MaxZCenter = Center2;
	}
	else
	{
		MinZCenter = Center2;
		MaxZCenter = Center1;
	}

	FVector Axis = MaxZCenter - MinZCenter;
	FVector AxisNorm = Axis.GetSafeNormal();
	float AxisLength = Axis.Size();
	double a = AxisNorm.X;
	double b = AxisNorm.Y;
	double c = AxisNorm.Z;

	double MinPlaneX = FMath::Min(Center1.X, Center2.X) - r;
	double MaxPlaneX = FMath::Max(Center1.X, Center2.X) + r;

	int PlaneMinXIndex, PlaneMaxXIndex;
	GetXIndexRegionFromXRegion(GridBox, MinPlaneX, MaxPlaneX, PlaneMinXIndex, PlaneMaxXIndex);

	for (int PlaneXIndex = PlaneMinXIndex; PlaneXIndex <= PlaneMaxXIndex; PlaneXIndex++)
	{
		double PlaneX = GetXFromXIndex(GridBox, PlaneXIndex);
		double MinLineY = GridBox.GetBoxSize().Y / 2;
		double MaxLineY = -GridBox.GetBoxSize().Y / 2;
		int MinLineYIndex, MaxLineYIndex;
		double X = PlaneX - MinZCenter.X;
		// 胶囊体轴线与X平面平行的情况: 可直接计算出直线x=X, y=Y的边界情况(即MinLineY和MaxLineY)
		if (FMath::Abs(a) < KINDA_SMALL_NUMBER)
		{
			double SquaredRyz = r * r - X * X;
			if (SquaredRyz > KINDA_SMALL_NUMBER)
			{
				MinLineY = FMath::Min(Center1.Y, Center2.Y) - FMath::Sqrt(r * r - X * X);
				MaxLineY = FMath::Max(Center1.Y, Center2.Y) + FMath::Sqrt(r * r - X * X);
			}
		}
		// 胶囊体轴线与X平面不平行的情况: 将胶囊体分解为一个圆柱体及两个半圆
		// 首先根据公式计算出平面x=X与的圆柱面(轴线为MAxisNorm，且两端无限延长)相交所形成曲线的y方向的极值点
		// 得到y方向上的两个极值点后，判断极值点是否在圆柱两端平面之内(利用向量(交点-MinZCenter)与轴向向量的点乘结果判断是否在两端平面之内)
		else
		{
			double AY = b * b + c * c - 1;
			double BY = a * b * X;
			double CY;
			double Y1, Y2;
			double Z1, Z2;

			CY = X * X * (a * a + c * c - 1) - r * r * (c * c - 1);
			Y1 = (-BY + FMath::Sqrt(FMath::Square(BY) - AY * CY)) / (AY);
			Y2 = (-BY - FMath::Sqrt(FMath::Square(BY) - AY * CY)) / (AY);
			Z1 = (a * X + b * Y1) * c / (1 - c * c);
			Z2 = (a * X + b * Y2) * c / (1 - c * c);
			// Z2 = BY * Y2 * c / (1 - c * c);

			// 两个交点的三维坐标
			FVector P1{X, Y1, Z1};
			FVector P2{X, Y2, Z2};
			// Y1在圆柱体下端面以下, 则计算平面x=X与圆柱体下端半球相交曲线(平面圆形曲线)的y轴极值点
			if (P1.Dot(AxisNorm) < 0)
			{
				MinLineY = MinZCenter.Y - FMath::Sqrt(r * r - X * X);
			}
			// Y1在圆柱体上端面以上, 则计算平面x=X与圆柱体上端半球相交曲线(平面圆形曲线)的y轴极值点
			else if (P1.Dot(AxisNorm) > Axis.Size())
			{
				MinLineY = MaxZCenter.Y - FMath::Sqrt(r * r - FMath::Square(PlaneX - MaxZCenter.X));
			}
			else
			{
				MinLineY = Y1 + MinZCenter.Y;
			}

			// 同理，对Y2做以上计算
			if (P2.Dot(AxisNorm) < 0)
			{
				MaxLineY = MinZCenter.Y + FMath::Sqrt(r * r - X * X);
			}
			// Y1在圆柱体上端面以上, 则计算平面x=X与圆柱体上端半球相交曲线(平面圆形曲线)的y轴极值点
			else if (P2.Dot(AxisNorm) > Axis.Size())
			{
				MaxLineY = MaxZCenter.Y + FMath::Sqrt(r * r - FMath::Square(PlaneX - MaxZCenter.X));
			}
			else
			{
				MaxLineY = Y2 + MinZCenter.Y;
			}
		}

		GetYIndexRegionFromYRegion(GridBox, MinLineY, MaxLineY, MinLineYIndex, MaxLineYIndex);
		for (int LineYIndex = MinLineYIndex; LineYIndex <= MaxLineYIndex; LineYIndex++)
		{
			double LineY = GetYFromYIndex(GridBox, LineYIndex);
			double Y = LineY - MinZCenter.Y;
			double MinZ{GridBox.GetBoxSize().Z / 2};
			double MaxZ{-GridBox.GetBoxSize().Z / 2};

			do
			{
				// 胶囊体轴线与Z轴平行的情况，直接计算直线x=X, y=Y与胶囊体上下两半球的交点
				if (FMath::Abs(c) > 1 - KINDA_SMALL_NUMBER)
				{
					double SquaredRz = r * r - X * X - Y * Y;
					if (SquaredRz > 0)
					{
						MinZ = MinZCenter.Z - FMath::Sqrt(SquaredRz);
						MaxZ = MaxZCenter.Z + FMath::Sqrt(SquaredRz);
					}
					break;
				}
				// 否则分别计算直线与胶囊体两个半球的交点，以及与胶囊体圆柱体的交点，并进行比较
				bool bMinZOnSphere = false;
				bool bMaxZOnSphere = false;
				do
				{
					// 如果相交: 分别判断上下两个交点是否在下部半球表面上，如果在(交点坐标向量与Axis向量点乘<0)，则设置MaxZ/MinZ为ZUpper和ZLower
					// 判断直线x=X, y=Y是否与下半球表面相交
					double SquaredZMinCenter = r * r - X * X - Y * Y;
					if (SquaredZMinCenter > KINDA_SMALL_NUMBER)
					{
						// 直线x=X, y=Y与下球表面的两个交点
						// Z1 < Z2
						float Z1 = -FMath::Sqrt(SquaredZMinCenter);
						float Z2 = +FMath::Sqrt(SquaredZMinCenter);

						FVector P1{X, Y, Z1};
						FVector P2{X, Y, Z2};
						// P1与下半球表面相交
						if (P1.Dot(AxisNorm) < 0)
						{
							MinZ = Z1 + MinZCenter.Z;
							bMinZOnSphere = true;
						}
						// P2与下半球表面相交
						if (P2.Dot(AxisNorm) < 0)
						{
							MaxZ = Z2 + MinZCenter.Z;
							bMaxZOnSphere = true;
						}
						// 如果P1和P2都与下半球表面相交，则跳出While
						if (bMinZOnSphere && bMaxZOnSphere)
						{
							break;
						}
					}

					// 判断直线x=X, y=Y是否与上半球表面相交
					double SquaredZMaxCenter = r * r - FMath::Square(PlaneX - MaxZCenter.X) - FMath::Square(
						LineY - MaxZCenter.Y);
					if (SquaredZMaxCenter > KINDA_SMALL_NUMBER)
					{
						// 直线x=X, y=Y与上球表面的两个交点
						// Z1 < Z2
						float Z1 = -FMath::Sqrt(SquaredZMaxCenter);
						float Z2 = +FMath::Sqrt(SquaredZMaxCenter);

						FVector P1{PlaneX - MaxZCenter.X, LineY - MaxZCenter.Y, Z1};
						FVector P2{P1.X, P1.Y, Z2};
						// P1与上半球表面相交
						if (P1.Dot(AxisNorm) > 0)
						{
							MinZ = Z1 + MaxZCenter.Z;
							bMinZOnSphere = true;
						}
						// P2与上半球表面相交
						if (P2.Dot(AxisNorm) > 0)
						{
							MaxZ = Z2 + MaxZCenter.Z;
							bMaxZOnSphere = true;
						}
						// 如果P1和P2都与上半球表面相交，则跳出While
						if (bMinZOnSphere && bMaxZOnSphere)
						{
							break;
						}
					}

					// 计算直线x=X, y=Y与胶囊体圆柱体的交点
					if (!(bMinZOnSphere && bMaxZOnSphere))
					{
						double A = 1 - c * c;
						double B = -c * (X * a + Y * b);
						double C = X * X * (1 - a * a) + Y * Y * (1 - b * b) - r * r - 2 * X * Y * a * b;
						double CylinderMinZ = MinZCenter.Z + (-B - FMath::Sqrt(FMath::Square(B) - A * C)) /
							A;
						double CylinderMaxZ = MinZCenter.Z + (-B + FMath::Sqrt(FMath::Square(B) - A * C)) /
							A;

						if (!bMinZOnSphere)
						{
							MinZ = CylinderMinZ;
						}
						if (!bMaxZOnSphere)
						{
							MaxZ = CylinderMaxZ;
						}
					}
				}
				while (false);
			}
			while (false);
			FixVoxelGridsWithSegment(GridBox, PlaneXIndex, LineYIndex, MinZ, MaxZ, VoxelGridsArray);
		}
	}
}

void TSVoxelGridFuncLib::FixVoxelGridsWithBox(const FVoxelBox& GridBox, const FVector& BoxExtent,
                                              const FTransform& TransformBS2VS, TArray<uint8>& VoxelGridsArray,
                                              UWorld* InWorld, bool bIsDrawDebug){
	const FTransform& TransformVS2BS = TransformBS2VS.Inverse();

	// VS means in Voxel Space
	// BS means in Body Space
	// Box AggGeom
	if (BoxExtent.X <= 0 || BoxExtent.Y <= 0 || BoxExtent.Z <= 0)
	{
		return;
	}

	// Box局部坐标系中的6个平面
	// FPlane BoxPlanesInVoxelSpace[6];
	FPlane BoxPlanesBS[6];
	BoxPlanesBS[0] = FPlane(FVector::XAxisVector, -BoxExtent.X);
	BoxPlanesBS[1] = FPlane(-FVector::XAxisVector, -BoxExtent.X);

	BoxPlanesBS[2] = FPlane(FVector::YAxisVector, -BoxExtent.Y);
	BoxPlanesBS[3] = FPlane(-FVector::YAxisVector, -BoxExtent.Y);

	BoxPlanesBS[4] = FPlane(FVector::ZAxisVector, -BoxExtent.Z);
	BoxPlanesBS[5] = FPlane(-FVector::ZAxisVector, -BoxExtent.Z);

	FPlane VoxelPlanesVS[6];
	VoxelPlanesVS[0] = FPlane(FVector::XAxisVector, -GridBox.GetBoxSize().X / 2);
	VoxelPlanesVS[1] = FPlane(-FVector::XAxisVector, -GridBox.GetBoxSize().X / 2);

	VoxelPlanesVS[2] = FPlane(FVector::YAxisVector, -GridBox.GetBoxSize().Y / 2);
	VoxelPlanesVS[3] = FPlane(-FVector::YAxisVector, -GridBox.GetBoxSize().Y / 2);

	VoxelPlanesVS[4] = FPlane(FVector::ZAxisVector, -GridBox.GetBoxSize().Z / 2);
	VoxelPlanesVS[5] = FPlane(-FVector::ZAxisVector, -GridBox.GetBoxSize().Z / 2);

	FPlane VoxelPlanesBS[6];
	for (int i = 0; i < 6; i++)
	{
		VoxelPlanesBS[i] = VoxelPlanesVS[i].TransformBy(TransformVS2BS.ToMatrixWithScale());
	}

	// VoxelGrid局部坐标系中Box的8个顶点
	FBox BoxBS{-BoxExtent, BoxExtent};
	// VoxelGrid局部坐标系下，从Box上选取四个点，四个点两两不在一条边上, 四个点存放在DiagonalPoints[0][0], DiagonalPoints[1][0], DiagonalPoints[2][0], DiagonalPoints[3][0]中
	// DiagonalPoints[i][1], DiagonalPoints[i][2], DiagonalPoints[i][3]分别存放与点DiagonalPoints[i][0]边相连的三个Box顶点
	// 从而构造出Box的12条边
	FVector BoxDiagonalPointsVS[4][4];
	BoxDiagonalPointsVS[0][0] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Min.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[0][1] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Min.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[0][2] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Max.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[0][3] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Min.Y, BoxBS.Min.Z});

	BoxDiagonalPointsVS[1][0] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Min.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[1][1] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Min.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[1][2] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Max.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[1][3] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Min.Y, BoxBS.Min.Z});

	BoxDiagonalPointsVS[2][0] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Max.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[2][1] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Min.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[2][2] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Max.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[2][3] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Max.Y, BoxBS.Max.Z});

	BoxDiagonalPointsVS[3][0] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Max.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[3][1] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Max.Y, BoxBS.Min.Z});
	BoxDiagonalPointsVS[3][2] = TransformBS2VS.TransformPosition(FVector{BoxBS.Max.X, BoxBS.Max.Y, BoxBS.Max.Z});
	BoxDiagonalPointsVS[3][3] = TransformBS2VS.TransformPosition(FVector{BoxBS.Min.X, BoxBS.Min.Y, BoxBS.Max.Z});

	const FTransform& BoxTransformDrawDebugLine = GridBox.GetBoxTransform();
	if (InWorld && bIsDrawDebug)
	{
		for (int i = 0; i < 4; i++)
		{
			FVector Point0 = BoxTransformDrawDebugLine.TransformPositionNoScale(BoxDiagonalPointsVS[i][0]);
			for (int j = 1; j < 4; j++)
			{
				FVector Pointj = BoxTransformDrawDebugLine.TransformPositionNoScale(BoxDiagonalPointsVS[i][j]);
				DrawDebugLine(InWorld, Point0, Pointj, FColor::Green, true, -1, 1, 2);
			}
		}
	}

	const FBox VoxelBoxVS{-GridBox.GetBoxSize() / 2, GridBox.GetBoxSize() / 2};
	TArray<FVector> PointsInVoxelBoxVS;

	// 最多有10个点与VoxelBox相交
	PointsInVoxelBoxVS.Reserve(24);

	// 分别判断每个点DiagonalPoints[i][0]是否在VoxelBox中，if true，则将该点加入PointsInVoxelBox中，else，与该点连接的三条边与VoxelBox的所有面的交点加入PointsInVoxelBox中
	for (int i = 0; i < 4; i++)
	{
		if (VoxelBoxVS.IsInside(BoxDiagonalPointsVS[i][0]))
		{
			PointsInVoxelBoxVS.Add(BoxDiagonalPointsVS[i][0]);
		}
		else
		{
			for (int j = 1; j < 4; j++)
			{
				int count = 0;
				FVector LineNormVS{BoxDiagonalPointsVS[i][0] - BoxDiagonalPointsVS[i][j]};
				LineNormVS.Normalize();
				FVector IntersectionPointVS;
				for (int k = 0; k < 6; k++)
				{
					if (FMath::Abs(LineNormVS.Dot(VoxelPlanesVS[k].GetNormal())) < KINDA_SMALL_NUMBER)
					{
						// 线段与平面平行则
						continue;
					}

					// 线段与Box的面相交
					if (FMath::SegmentPlaneIntersection(BoxDiagonalPointsVS[i][0], BoxDiagonalPointsVS[i][j],
					                                    VoxelPlanesVS[k], IntersectionPointVS))
					{
						// float Delta = UE_KINDA_SMALL_NUMBER;
						// if (IntersectionPointVS.X > -GridBox.GetBoxSize().X / 2 - UE_KINDA_SMALL_NUMBER &&
						// 	IntersectionPointVS.X < GridBox.GetBoxSize().X / 2 + UE_KINDA_SMALL_NUMBER &&
						// 	IntersectionPointVS.Y > -GridBox.GetBoxSize().Y / 2 - UE_KINDA_SMALL_NUMBER &&
						// 	IntersectionPointVS.Y < GridBox.GetBoxSize().Y / 2 + UE_KINDA_SMALL_NUMBER &&
						// 	IntersectionPointVS.Z > -GridBox.GetBoxSize().Z / 2 - UE_KINDA_SMALL_NUMBER &&
						// 	IntersectionPointVS.Z < GridBox.GetBoxSize().Z / 2 + UE_KINDA_SMALL_NUMBER)
						// {
						// 	count++;
						// 	PointsInVoxelBoxVS.Add(IntersectionPointVS);
						// }
						count++;
						PointsInVoxelBoxVS.Add(IntersectionPointVS);
					}
					// 一条线段最多与Box的6个面相交两个点
					if (count >= 2)
					{
						break;
					}
				}
			}
		}
	}

	if (InWorld && bIsDrawDebug)
	{
		for (const auto& Point : PointsInVoxelBoxVS)
		{
			const FVector PointInWorldSpace = BoxTransformDrawDebugLine.TransformPositionNoScale(Point);
			// DrawDebugPoint(InWorld, PointInWorldSpace, 5, FColor::Green, true, -1, 0);
			DrawDebugPoint(InWorld, BoxTransformDrawDebugLine.TransformPositionNoScale(Point), 2, FColor::Red, true);
		}
	}

	// 根据PointsInVoxelBox，计算Box与VoxelBox重叠体的X方向近平面与远平面
	// 如果Box与VoxelBox不相交，则跳出函数
	if (PointsInVoxelBoxVS.IsEmpty())
	{
		return;
	}

	double MinXVS = PointsInVoxelBoxVS[0].X;
	double MaxXVS = MinXVS;
	for (const auto& Point : PointsInVoxelBoxVS)
	{
		if (Point.X < MinXVS)
		{
			MinXVS = Point.X;
		}
		if (Point.X > MaxXVS)
		{
			MaxXVS = Point.X;
		}
	}

	MinXVS = FMath::Max(-GridBox.GetBoxSize().X / 2, MinXVS);
	MaxXVS = FMath::Min(GridBox.GetBoxSize().X / 2, MaxXVS);

	// 遍历VoxelBox中的平面（MinX<X<MaxX）
	TArray<FVector> BoxEdgePlaneXIntersectionPointsVS;
	BoxEdgePlaneXIntersectionPointsVS.Reserve(4);

	TArray<FVector> BoxLineYIntersectionPointsBS;
	BoxLineYIntersectionPointsBS.Reserve(2);
	int MinPlaneXIndex, MaxPlaneXIndex;
	GetXIndexRegionFromXRegion(GridBox, MinXVS, MaxXVS, MinPlaneXIndex, MaxPlaneXIndex);
	for (int PlaneXIndex = MinPlaneXIndex; PlaneXIndex <= MaxPlaneXIndex; PlaneXIndex++)
	{
		float PlaneX = GetXFromXIndex(GridBox, PlaneXIndex);
		FPlane XPlane{FVector::XAxisVector, PlaneX};
		// 求取平面x=PlaneX与Box各条边的交点
		BoxEdgePlaneXIntersectionPointsVS.Empty();
		int IntersectionPointCount = 0;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 1; j < 4; j++)
			{
				FVector LineNormVS = BoxDiagonalPointsVS[i][0] - BoxDiagonalPointsVS[i][j];
				LineNormVS.Normalize();
				if (FMath::Abs(LineNormVS.Dot(FVector::XAxisVector)) < KINDA_SMALL_NUMBER)
				{
					continue;
				}
				// 线段与平面平行则
				FVector IntersectionPointVS;
				if (FMath::SegmentPlaneIntersection(BoxDiagonalPointsVS[i][0], BoxDiagonalPointsVS[i][j],
				                                    XPlane, IntersectionPointVS))
				{
					IntersectionPointCount++;
					BoxEdgePlaneXIntersectionPointsVS.Add(IntersectionPointVS);
				}
				// 对于任意平面，Box中至多有4条边线段与其相交
				if (IntersectionPointCount >= 4)
				{
					break;
				}
			}
			// 对于任意平面，Box中至多有4条边线段与其相交
			if (IntersectionPointCount >= 4)
			{
				break;
			}
		}

		// 求取直线x=PlaneX, y=Y与Box的交点
		if (BoxEdgePlaneXIntersectionPointsVS.IsEmpty())
		{
			continue;
		}
		float MinYVS = BoxEdgePlaneXIntersectionPointsVS[0].Y;
		float MaxYVS = BoxEdgePlaneXIntersectionPointsVS[0].Y;
		for (const auto& Point : BoxEdgePlaneXIntersectionPointsVS)
		{
			if (Point.Y < MinYVS)
			{
				MinYVS = Point.Y;
			}
			if (Point.Y > MaxYVS)
			{
				MaxYVS = Point.Y;
			}
		}
		MinYVS = FMath::Max(MinYVS, -GridBox.GetBoxSize().Y / 2);
		MaxYVS = FMath::Min(MaxYVS, GridBox.GetBoxSize().Y / 2);

		int MinYIndex, MaxYIndex;
		GetYIndexRegionFromYRegion(GridBox, MinYVS, MaxYVS, MinYIndex, MaxYIndex);

		for (int LineYIndex = MinYIndex; LineYIndex <= MaxYIndex; LineYIndex++)
		{
			int count = 0;
			float LineY = GetYFromYIndex(GridBox, LineYIndex);
			FVector IntersectionPointBS;
			// FVector VoxelPoint1BS = TransformVS2BS.
			// 	TransformPosition(FVector(PlaneX, LineY, -GridBox.GetBoxSize().Z / 2));
			// FVector VoxelPoint2BS = TransformVS2BS.
			// 	TransformPosition(FVector(PlaneX, LineY, GridBox.GetBoxSize().Z / 2));
			// FVector VoxelPoint1BS=TransformVS2BS.TransformPositionNoScale(FVector(PlaneX, LineY, -GridBox.GetBoxSize().Z / 2));
			FVector VoxelPoint1BS=TransformBS2VS.InverseTransformPositionNoScale(FVector(PlaneX, LineY, -GridBox.GetBoxSize().Z / 2));
			// VoxelPoint1BS = TransformVS2BS.GetRotation().RotateVector(VoxelPoint1BS);
			// VoxelPoint1BS=TransformVS2BS.GetLocation()+VoxelPoint1BS;
			VoxelPoint1BS=TransformVS2BS.GetScale3D()*VoxelPoint1BS;


			FVector VoxelPoint2BS=TransformBS2VS.InverseTransformPositionNoScale(FVector(PlaneX, LineY, GridBox.GetBoxSize().Z / 2));
			// VoxelPoint2BS = TransformVS2BS.GetRotation().RotateVector(VoxelPoint2BS);
			VoxelPoint2BS=TransformVS2BS.GetScale3D()*VoxelPoint2BS;

			// FVector VoxelPoint2BS=TransformVS2BS.GetScale3D()*FVector(PlaneX, LineY, GridBox.GetBoxSize().Z / 2);
			// VoxelPoint2BS = TransformVS2BS.GetRotation().RotateVector(VoxelPoint2BS);
			// VoxelPoint2BS=TransformVS2BS.GetLocation()+VoxelPoint2BS;

			FVector VoxelLineNormBS = VoxelPoint2BS - VoxelPoint1BS;
			VoxelLineNormBS.Normalize();

			BoxLineYIntersectionPointsBS.Empty();
			for (int i = 0; i < 6; i++)
			{
				FVector BoxPlaneNormBS = BoxPlanesBS[i].GetNormal();
				if (FMath::Abs(VoxelLineNormBS.Dot(BoxPlaneNormBS)) < KINDA_SMALL_NUMBER)
				{
					continue;
				}
				IntersectionPointBS = FMath::LinePlaneIntersection(VoxelPoint1BS, VoxelPoint2BS,
				                                                   BoxPlanesBS[i].GetOrigin(),
				                                                   BoxPlanesBS[i].GetNormal());
				{
					float BoxXExtent = BoxBS.GetExtent().X + KINDA_SMALL_NUMBER;
					float BoxYExtent = BoxBS.GetExtent().Y + KINDA_SMALL_NUMBER;
					float BoxZExtent = BoxBS.GetExtent().Z + KINDA_SMALL_NUMBER;
					if (FMath::Abs(IntersectionPointBS.X) > BoxXExtent
						|| FMath::Abs(IntersectionPointBS.Y) > BoxYExtent
						|| FMath::Abs(IntersectionPointBS.Z) > BoxZExtent)
					{
						continue;
					}
					BoxLineYIntersectionPointsBS.Add(IntersectionPointBS);
					count++;
					if (count >= 2)
					{
						break;
					}
				}
			}

			if (BoxLineYIntersectionPointsBS.Num() < 2)
			{
				continue;
			}


			FVector BoxLineYIntersectionPointsVS[2];

			for (int i = 0; i < 2; i++)
			{
				BoxLineYIntersectionPointsVS[i] = TransformBS2VS.TransformPosition(BoxLineYIntersectionPointsBS[i]);
			}

			double MinZVS = FMath::Min(BoxLineYIntersectionPointsVS[0].Z, BoxLineYIntersectionPointsVS[1].Z);
			double MaxZVS = FMath::Max(BoxLineYIntersectionPointsVS[0].Z, BoxLineYIntersectionPointsVS[1].Z);
			FixVoxelGridsWithSegment(GridBox, PlaneXIndex, LineYIndex, MinZVS, MaxZVS, VoxelGridsArray);

			FTransform BoxTransform = GridBox.GetBoxTransform();
			FVector StartPointGS = BoxTransform.TransformPosition(FVector(PlaneX, LineY, MinZVS));
			FVector EndPointGS = BoxTransform.TransformPosition(FVector(PlaneX, LineY, MaxZVS));

			// Debug Draw Point
			// DrawDebugSolidBox(GetWorld(), StartPointGS, FVector{1, 1, 1}, FColor::Blue, true, -1, 0);
			// DrawDebugSolidBox(GetWorld(), EndPointGS, FVector{1, 1, 1}, FColor::Blue, true, -1, 0);
		}
	}
}

void TSVoxelGridFuncLib::FixVoxelGridsWithSphere(const FVoxelBox& GridBox, const FVector& Center, float Radius,
                                                 TArray<uint8>& VoxelGridsArray){
	float BackPlaneX = Center.X - Radius;
	float FrontPlaneX = Center.X + Radius;

	int BackPlaneXI;
	int FrontPlaneXI;
	GetXIndexRegionFromXRegion(GridBox, BackPlaneX, FrontPlaneX, BackPlaneXI, FrontPlaneXI);

	for (int XPlane_i = BackPlaneXI; XPlane_i <= FrontPlaneXI; XPlane_i++)
	{
		float PlaneX = GetXFromXIndex(GridBox, XPlane_i);
		float XCenterDist = Center.X - PlaneX;
		float LeftLineY = Center.Y - FMath::Sqrt(Radius * Radius - XCenterDist * XCenterDist);
		float RightLineY = Center.Y + FMath::Sqrt(Radius * Radius - XCenterDist * XCenterDist);
		int LeftLineYI;
		int RightLineYI;
		GetYIndexRegionFromYRegion(GridBox, LeftLineY, RightLineY, LeftLineYI, RightLineYI);
		for (int YLine_i = LeftLineYI; YLine_i <= RightLineYI; YLine_i++)
		{
			float Y = GetYFromYIndex(GridBox, YLine_i);
			float YCenterDist = Center.Y - Y;
			float ZSqr = Radius * Radius - XCenterDist * XCenterDist - YCenterDist * YCenterDist;
			if (ZSqr > 0.f)
			{
				float ZSqrt = FMath::Sqrt(ZSqr);
				float ZUp = Center.Z + ZSqrt;
				float ZDown = Center.Z - ZSqrt;
				FixVoxelGridsWithSegment(GridBox, XPlane_i, YLine_i, ZDown, ZUp, VoxelGridsArray);
			}
		}
	}
}

void TSVoxelGridFuncLib::FixVoxelGridsWithConvexMesh(const FVoxelBox& GridBox, const FKConvexElem& Convex,
                                                     const FTransform& ConvexTransformInVoxelBoxSpace,
                                                     TArray<uint8>& VoxelGridsArray){
	const double MinZ = -GridBox.GetBoxSize().Z / 2;
	const double MaxZ = GridBox.GetBoxSize().Z / 2;
	TArray<FZAxisSegment> ZAxisSegmentArray;
	TArray<FMinAndMaxYLineInPlaneX> MinAndMaxYLineInPlaneXArray;
	ResetZAxisSegmentArray(GridBox, ZAxisSegmentArray);
	ResetMinAndMaxYLineInPlaneXArray(GridBox, MinAndMaxYLineInPlaneXArray);
	TArray<double> FaceEdgePlaneIntersectionsYArray;
	FaceEdgePlaneIntersectionsYArray.Reserve(3);
	const FTransform ConvexMeshTransformInVoxelBoxSpace = Convex.GetTransform() * ConvexTransformInVoxelBoxSpace;
	int ConvexMinXPlaneIndex = GridBox.GetGridHalfNumX() * 2 - 1;
	int ConvexMaxXPlaneIndex = 0;
	uint32 GridYNum = GridBox.GetGridHalfNumY() * 2;
	for (int FaceIndex = 0; FaceIndex < Convex.IndexData.Num(); FaceIndex += 3)
	{
		const FVector Point1 = Convex.VertexData[Convex.IndexData[FaceIndex]];
		const FVector Point2 = Convex.VertexData[Convex.IndexData[FaceIndex + 1]];
		const FVector Point3 = Convex.VertexData[Convex.IndexData[FaceIndex + 2]];
		const FPlane FacePlane{Point1, Point2, Point3};
		const FVector FacePlaneNormal = FacePlane.GetNormal();
		const FVector FacePlaneOrigin = FacePlane.GetOrigin();
		const FVector Point1InBoxSpace = ConvexMeshTransformInVoxelBoxSpace.TransformPosition(Point1);
		const FVector Point2InBoxSpace = ConvexMeshTransformInVoxelBoxSpace.TransformPosition(Point2);
		const FVector Point3InBoxSpace = ConvexMeshTransformInVoxelBoxSpace.TransformPosition(Point3);

		FVector GridBoxHalfSize = GridBox.GetBoxSize() / 2;
		if (Point1InBoxSpace.X < -GridBoxHalfSize.X
			&& Point2InBoxSpace.X < -GridBoxHalfSize.X
			&& Point3InBoxSpace.X < -GridBoxHalfSize.X)
		{
			continue;
		}
		if (Point1InBoxSpace.X > GridBoxHalfSize.X
			&& Point2InBoxSpace.X > GridBoxHalfSize.X
			&& Point3InBoxSpace.X > GridBoxHalfSize.X)
		{
			continue;
		}
		double MinX = FMath::Min(FMath::Min(Point1InBoxSpace.X, Point2InBoxSpace.X), Point3InBoxSpace.X);
		MinX = FMath::Max(MinX, -GridBoxHalfSize.X);

		double MaxX = FMath::Max(FMath::Max(Point1InBoxSpace.X, Point2InBoxSpace.X), Point3InBoxSpace.X);
		MaxX = FMath::Min(MaxX, GridBoxHalfSize.X);

		int FaceMinXIndex, FaceMaxXIndex;
		GetXIndexRegionFromXRegion(GridBox, MinX, MaxX, FaceMinXIndex, FaceMaxXIndex);

		ConvexMinXPlaneIndex = FMath::Min(ConvexMinXPlaneIndex, FaceMinXIndex);
		ConvexMaxXPlaneIndex = FMath::Max(ConvexMaxXPlaneIndex, FaceMaxXIndex);


		int XIndex = FaceMinXIndex;
		for (; XIndex <= FaceMaxXIndex; XIndex++)
		{
			double X = GetXFromXIndex(GridBox, XIndex);
			FaceEdgePlaneIntersectionsYArray.Empty();
			FPlane XPlane{1.0, 0.0, 0.0, X};
			FVector XPlaneFaceEdgeIntersectionPoint;
			if (FMath::SegmentPlaneIntersection(Point1InBoxSpace, Point2InBoxSpace, XPlane,
			                                    XPlaneFaceEdgeIntersectionPoint))
			{
				FaceEdgePlaneIntersectionsYArray.Add(XPlaneFaceEdgeIntersectionPoint.Y);
			}
			if (FMath::SegmentPlaneIntersection(Point1InBoxSpace, Point3InBoxSpace, XPlane,
			                                    XPlaneFaceEdgeIntersectionPoint))
			{
				FaceEdgePlaneIntersectionsYArray.Add(XPlaneFaceEdgeIntersectionPoint.Y);
			}
			if (FMath::SegmentPlaneIntersection(Point2InBoxSpace, Point3InBoxSpace, XPlane,
			                                    XPlaneFaceEdgeIntersectionPoint))
			{
				FaceEdgePlaneIntersectionsYArray.Add(XPlaneFaceEdgeIntersectionPoint.Y);
			}
			if (FaceEdgePlaneIntersectionsYArray.IsEmpty())
			{
				continue;
			}
			double MinY = FaceEdgePlaneIntersectionsYArray[0];
			double MaxY = FaceEdgePlaneIntersectionsYArray[0];

			for (int i = 1; i < FaceEdgePlaneIntersectionsYArray.Num(); i++)
			{
				if (FaceEdgePlaneIntersectionsYArray[i] < MinY)
				{
					MinY = FaceEdgePlaneIntersectionsYArray[i];
				}
				else
				{
					MaxY = FaceEdgePlaneIntersectionsYArray[i];
				}
			}
			MinY = FMath::Max(MinY, -GridBoxHalfSize.Y);
			MaxY = FMath::Min(MaxY, GridBoxHalfSize.Y);

			int MinYIndex, MaxYIndex;
			GetYIndexRegionFromYRegion(GridBox, MinY, MaxY, MinYIndex, MaxYIndex);
			MinAndMaxYLineInPlaneXArray[XIndex].MinYLineIndex = FMath::Min(
				MinAndMaxYLineInPlaneXArray[XIndex].MinYLineIndex, MinYIndex);
			MinAndMaxYLineInPlaneXArray[XIndex].MaxYLineIndex = FMath::Max(
				MinAndMaxYLineInPlaneXArray[XIndex].MaxYLineIndex, MaxYIndex);

			int YIndex = MinYIndex;
			for (; YIndex <= MaxYIndex; YIndex++)
			{
				if (XIndex == 10 && YIndex == 10)
				{
					int i = 1;
				}
				if (ZAxisSegmentArray[XIndex * GridBox.GetGridHalfNumY() * 2 + YIndex].IntersectionPointNum == 2)
				{
					continue;
				}

				double Y = GetYFromYIndex(GridBox, YIndex);
				FVector IntersectionPoint;
				FVector IntersectionNorm;
				bool bIntersected = false;
				FPlane YPlane{0.0, 1.0, 0.0, Y};
				if (ZAxisSegmentArray[XIndex * GridYNum + YIndex].IntersectionPointNum < 2)
				{
					constexpr double MaxZExtent = 1000000.0;
					constexpr double MinZExtent = -MaxZExtent;
					bIntersected = FMath::SegmentTriangleIntersection(FVector{X, Y, MinZExtent},
					                                                  FVector{X, Y, MaxZExtent}, Point1InBoxSpace,
					                                                  Point2InBoxSpace, Point3InBoxSpace,
					                                                  IntersectionPoint, IntersectionNorm);
				}

				if (bIntersected)
				{
					switch (ZAxisSegmentArray[XIndex * GridYNum + YIndex].IntersectionPointNum)
					{
					case 0:
						ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[0] = IntersectionPoint.Z;
						ZAxisSegmentArray[XIndex * GridYNum + YIndex].IntersectionPointNum = 1;
						break;
					case 1:
						ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[1] = IntersectionPoint.Z;
						ZAxisSegmentArray[XIndex * GridYNum + YIndex].IntersectionPointNum = 2;
						break;
					default:
						break;
					}
				}
			}
		}
	}

	for (int XIndex = ConvexMinXPlaneIndex; XIndex <= ConvexMaxXPlaneIndex; XIndex++)
	{
		for (int YIndex = MinAndMaxYLineInPlaneXArray[XIndex].MinYLineIndex; YIndex <= MinAndMaxYLineInPlaneXArray[
			     XIndex].MaxYLineIndex; YIndex++)
		{
			if (ZAxisSegmentArray[XIndex * GridYNum + YIndex].IntersectionPointNum == 2)
			{
				float ZStart = FMath::Min(ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[0],
				                          ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[1]);
				float ZEnd = FMath::Max(ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[0],
				                        ZAxisSegmentArray[XIndex * GridYNum + YIndex].Z[1]);
				float PlaneX = GetXFromXIndex(GridBox, XIndex);
				float LineY = GetYFromYIndex(GridBox, YIndex);
				FixVoxelGridsWithSegment(GridBox, XIndex, YIndex, ZStart, ZEnd, VoxelGridsArray);

				FVector StartPoint = GridBox.GetBoxTransform().
				                             TransformPosition(FVector(PlaneX, LineY, ZStart));
				FVector EndPoint = GridBox.GetBoxTransform().TransformPosition(FVector(PlaneX, LineY, ZEnd));

				// Debug Draw Point
				// DrawDebugSolidBox(GetWorld(), StartPoint, FVector{1, 1, 1}, FColor::Blue, true, -1, 0);
				// DrawDebugSolidBox(GetWorld(), EndPoint, FVector{1, 1, 1}, FColor::Blue, true, -1, 0);
			}
		}
	}
}

bool TSVoxelGridFuncLib::FixVoxelGridsWithSegment(const FVoxelBox& GridBox, int PlaneXIndex, int LineYIndex,
                                                  float ZMin, float ZMax, TArray<uint8>& VoxelGridsArray){
	int GridNumX = GridBox.GetGridHalfNumX() * 2;
	int GridNumY = GridBox.GetGridHalfNumY() * 2;
	int GridNumZ = GridBox.GetGridHalfNumZ() * 2;
	if (PlaneXIndex < 0 || PlaneXIndex > GridNumX - 1
		|| LineYIndex < 0 || LineYIndex > GridNumY - 1
		|| ZMin > ZMax)
	{
		return false;
	}

	int ZMaxGridIndex, ZMinGridIndex;
	GetZIndexRegionFromZRegion(GridBox, ZMin, ZMax, ZMinGridIndex, ZMaxGridIndex);
	if (ZMaxGridIndex < ZMinGridIndex)
	{
		return false;
	}

	if (ZMinGridIndex > GridNumZ - 1 || ZMaxGridIndex < 0)
	{
		return false;
	}

	int ZMemCpyStartByte = ZMinGridIndex / 8 + 1;
	int ZMemCpyEndByte = ZMaxGridIndex / 8;
	int ZStartBit = ZMinGridIndex % 8;
	int ZEndBit = ZMaxGridIndex % 8;

	uint16 Aligned_8_GridZNum;
	// ZMinGridIndex 和 ZMaxGridIndex不在一个字节内
	if (GridNumZ % 8 == 0)
	{
		Aligned_8_GridZNum = GridNumZ;
	}
	else
	{
		Aligned_8_GridZNum = (GridNumZ / 8 + 1) * 8;
	}
	int ByteStart = (PlaneXIndex * GridNumY + LineYIndex) * Aligned_8_GridZNum / 8 + ZMemCpyStartByte;
	int ByteEnd = (PlaneXIndex * GridNumY + LineYIndex) * Aligned_8_GridZNum / 8 + ZMemCpyEndByte;

	for (int Byte = ByteStart; Byte < ByteEnd; Byte++)
	{
		VoxelGridsArray[Byte] = 0XFF;
	}

	if (ByteStart <= ByteEnd)
	{
		switch (ZStartBit)
		{
		case 0:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X00);
			break;
		case 1:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X01);
			break;
		case 2:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X03);
			break;
		case 3:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X07);
			break;
		case 4:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X0F);
			break;
		case 5:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X1F);
			break;
		case 6:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X3F);
			break;
		case 7:
			VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | (0XFF - 0X7F);
			break;
		default:
			break;
		}

		switch (ZEndBit)
		{
		case 0:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X01;
			break;
		case 1:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X03;
			break;
		case 2:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X07;
			break;
		case 3:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X0F;
			break;
		case 4:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X1F;
			break;
		case 5:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X3F;
			break;
		case 6:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0X7F;
			break;
		case 7:
			VoxelGridsArray[ByteEnd] = VoxelGridsArray[ByteEnd] | 0XFF;
			break;
		default:
			break;
		}
	}
	// ZMinGridIndex和ZMaxGridIndex在一个字节内
	else
	{
		uint8 TempByte = 0XFF;

		switch (ZStartBit)
		{
		case 0:
			TempByte = TempByte & (0XFF - 0X00);
			break;
		case 1:
			TempByte = TempByte & (0XFF - 0X01);
			break;
		case 2:
			TempByte = TempByte & (0XFF - 0X03);
			break;
		case 3:
			TempByte = TempByte & (0XFF - 0X07);
			break;
		case 4:
			TempByte = TempByte & (0XFF - 0X0F);
			break;
		case 5:
			TempByte = TempByte & (0XFF - 0X1F);
			break;
		case 6:
			TempByte = TempByte & (0XFF - 0X3F);
			break;
		case 7:
			TempByte = TempByte & (0XFF - 0X7F);
			break;
		default:
			break;
		}
		switch (ZEndBit)
		{
		case 0:
			TempByte = TempByte & 0x01;
			break;
		case 1:
			TempByte = TempByte & 0x03;
			break;
		case 2:
			TempByte = TempByte & 0x07;
			break;
		case 3:
			TempByte = TempByte & 0x0F;
			break;
		case 4:
			TempByte = TempByte & 0x1F;
			break;
		case 5:
			TempByte = TempByte & 0x3F;
			break;
		case 6:
			TempByte = TempByte & 0x7F;
			break;
		case 7:
			TempByte = TempByte & 0xFF;
			break;
		default:
			break;
		}
		VoxelGridsArray[ByteStart - 1] = VoxelGridsArray[ByteStart - 1] | TempByte;
	}

	return true;
}

void TSVoxelGridFuncLib::DrawDebugGrids(const UWorld* World, const FVoxelBox& VoxelBox, float TimeLength,
                                        FColor Color, const TArray<uint8>& Voxels){
	FTransform BoxTransform = VoxelBox.GetBoxTransform();
	BoxTransform.RemoveScaling();
	FQuat BoxRotation = BoxTransform.GetRotation();

	FVector GirdExtent = VoxelBox.GetGridSize() / 4;

	int GridNumX = VoxelBox.GetGridHalfNumX() * 2;
	int GridNumY = VoxelBox.GetGridHalfNumY() * 2;
	int GridNumZ = VoxelBox.GetGridHalfNumZ() * 2;
	int Aligned_8_GridNumZ = (GridNumZ + 1) / 8 * 8;

	for (int x_i = 0; x_i < VoxelBox.GetGridHalfNumX() * 2; x_i++)
	{
		for (int y_i = 0; y_i < VoxelBox.GetGridHalfNumY() * 2; y_i++)
		{
			for (int zbit_i = 0; zbit_i < VoxelBox.GetGridHalfNumZ() * 2; zbit_i++)
			{
				int ByteIndex = zbit_i / 8;
				int BitIndex = zbit_i % 8;
				const auto& Byte = Voxels[(x_i * GridNumY + y_i) * Aligned_8_GridNumZ / 8 + ByteIndex];
				bool bShouldDraw{false};
				switch (BitIndex)
				{
				case 0:
					bShouldDraw = Byte & 0X01;
					break;
				case 1:
					bShouldDraw = Byte & 0X02;
					break;
				case 2:
					bShouldDraw = Byte & 0X04;
					break;
				case 3:
					bShouldDraw = Byte & 0X08;
					break;
				case 4:
					bShouldDraw = Byte & 0X10;
					break;
				case 5:
					bShouldDraw = Byte & 0X20;
					break;
				case 6:
					bShouldDraw = Byte & 0X40;
					break;
				case 7:
					bShouldDraw = Byte & 0X80;
					break;
				default:
					break;
				}

				if (bShouldDraw)
				{
					FVector Center;
					Center.X = (x_i - GridNumX / 2.f + 0.5f) * VoxelBox.GetGridSize().X;
					Center.Y = (y_i - GridNumY / 2.f + 0.5f) * VoxelBox.GetGridSize().Y;
					Center.Z = (zbit_i - GridNumZ / 2.f + 0.5f) * VoxelBox.GetGridSize().Z;
					FVector GridCenterLocation = BoxTransform.TransformPosition(Center);

					DrawDebugBox(World, GridCenterLocation + FVector(0, 0, 2000), GirdExtent, BoxRotation, Color, false,
					             TimeLength);
				}
			}
		}
	}
}
